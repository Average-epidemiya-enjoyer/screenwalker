"""Анализатор паттернов — исследует журналы выполнения и предлагает улучшения.

Предоставляются две взаимодополняющие возможности:

**Онлайн-обучение** (:meth:`PatternLearner.observe`)
    Вызывается движком после каждого успешного OCR-совпадения. Когда один
    и тот же наблюдаемый текст встречается :attr:`~PatternLearner.min_observations`
    раз для одной канонической метки, он автоматически добавляется в реестр синонимов.

**Офлайн-анализ логов** (:meth:`PatternLearner.analyze_logs`)
    Читает все файлы ``steps.jsonl`` в директории логов и создаёт
    :class:`LearningReport`, содержащий:

    * Какие шаги чаще всего завершаются неудачей (и почему).
    * OCR-несоответствия — кандидаты для новых записей синонимов.
    * Среднее время выполнения каждого шага (для оптимизации таймаутов).
    * Статистику успеха/неудач по методам vision.
    * Рекомендуемые значения таймаутов (p95 × 1.2).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from screenwalker.matching.synonyms import SynonymRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Типы данных отчёта
# ---------------------------------------------------------------------------


@dataclass
class StepStats:
    """Агрегированная статистика выполнения для одного идентификатора шага.

    Attributes:
        count: Общее количество наблюдённых выполнений.
        success_count: Успешные выполнения.
        failure_count: Неудачные выполнения.
        durations_ms: Все наблюдённые длительности выполнения в миллисекундах.
        errors: Сообщения об ошибках из неудачных выполнений (дедуплицированные).
        methods: Количество использований каждого метода vision.
    """

    count: int = 0
    success_count: int = 0
    failure_count: int = 0
    durations_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    methods: dict[str, int] = field(default_factory=dict)


@dataclass
class OcrMismatch:
    """Фиксирует OCR-считывание, отличающееся от ожидаемой метки.

    Attributes:
        find_target: Текст, по которому выполнялся поиск.
        found_text: Фактический текст, возвращённый OCR.
        step_id: Шаг, в котором наблюдалось данное несоответствие.
        count: Количество наблюдений данной конкретной пары.
    """

    find_target: str
    found_text: str
    step_id: str
    count: int = 1


@dataclass
class LearningReport:
    """Сводка, созданная методом :meth:`PatternLearner.analyze_logs`.

    Attributes:
        run_count: Количество проанализированных отдельных запусков сценария.
        total_steps: Общее количество выполнений шагов по всем запускам.
        step_stats: Агрегированная статистика по каждому шагу.
        ocr_mismatches: OCR-считывания, отличавшиеся от целевого текста поиска.
        method_stats: Счётчики успеха/неудачи по каждому методу.
        suggested_timeouts: Рекомендуемые значения таймаутов (мс) по каждому шагу.
    """

    run_count: int
    total_steps: int
    step_stats: dict[str, StepStats]
    ocr_mismatches: list[OcrMismatch]
    method_stats: dict[str, dict[str, int]]
    suggested_timeouts: dict[str, int]

    @property
    def failed_step_ids(self) -> list[str]:
        """Отсортированный список идентификаторов шагов, имеющих хотя бы одну неудачу."""
        return sorted(
            sid for sid, s in self.step_stats.items() if s.failure_count > 0
        )

    def format_text(self) -> str:
        """Сформировать отчёт в виде читаемого текстового резюме.

        Returns:
            Многострочная строка, готовая для вывода через ``click.echo()``.
        """
        lines: list[str] = []
        lines.append("=== Learning Report ===")
        lines.append(f"Runs: {self.run_count}  |  Total steps: {self.total_steps}")

        # Неудавшиеся шаги
        failed = self.failed_step_ids
        if failed:
            lines.append(f"\n--- Failed Steps ({len(failed)}) ---")
            for sid in failed:
                s = self.step_stats[sid]
                pct = s.failure_count / s.count * 100 if s.count else 0
                top_err = s.errors[0][:80] if s.errors else "unknown error"
                lines.append(
                    f"  {sid}: {s.failure_count}/{s.count} failures "
                    f"({pct:.0f}%)  — {top_err}"
                )
        else:
            lines.append("\n  No failed steps.")

        # OCR-несоответствия
        if self.ocr_mismatches:
            lines.append(
                f"\n--- OCR Mismatches — synonym candidates "
                f"({len(self.ocr_mismatches)}) ---"
            )
            for m in self.ocr_mismatches[:20]:
                lines.append(
                    f"  '{m.find_target}'  →  OCR read '{m.found_text}'  "
                    f"({m.count}× in step '{m.step_id}')"
                )

        # Статистика по методам
        if self.method_stats:
            lines.append("\n--- Method Effectiveness ---")
            for method, stats in sorted(self.method_stats.items()):
                ok = stats.get("success", 0)
                total = ok + stats.get("failure", 0)
                pct = ok / total * 100 if total else 0
                lines.append(f"  {method}: {ok}/{total} ({pct:.0f}% success)")

        # Предложения по таймаутам
        if self.suggested_timeouts:
            lines.append("\n--- Timeout Suggestions (p95 × 1.2) ---")
            for sid, ms in sorted(self.suggested_timeouts.items()):
                lines.append(f"  {sid}: {ms} ms")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PatternLearner
# ---------------------------------------------------------------------------


_EMPTY_REPORT = LearningReport(
    run_count=0,
    total_steps=0,
    step_stats={},
    ocr_mismatches=[],
    method_stats={},
    suggested_timeouts={},
)


class PatternLearner:
    """Наблюдает успешные совпадения и предлагает новые расширения синонимов.

    Также выполняет офлайн-анализ логов через :meth:`analyze_logs`.

    Attributes:
        registry: :class:`~screenwalker.matching.synonyms.SynonymRegistry`,
            пополняемый выученными паттернами.
        min_observations: Количество наблюдений паттерна, необходимое для
            автоматического продвижения.
        evidence_path: Опциональный путь для сохранения счётчиков наблюдений.
    """

    def __init__(
        self,
        registry: SynonymRegistry,
        min_observations: int = 3,
        evidence_path: Path | str | None = None,
    ) -> None:
        """Инициализировать PatternLearner.

        Args:
            registry: Реестр синонимов для расширения выученными паттернами.
            min_observations: Порог доказательств для автоматического продвижения.
            evidence_path: YAML-файл для сохранения счётчиков наблюдений между запусками.
        """
        self.registry = registry
        self.min_observations = min_observations
        self.evidence_path = Path(evidence_path) if evidence_path else None
        # canonical_label → {observed_text → count}
        self._observations: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._last_report: LearningReport = _EMPTY_REPORT

    # ------------------------------------------------------------------
    # Онлайн-обучение
    # ------------------------------------------------------------------

    def observe(self, canonical_label: str, observed_text: str) -> None:
        """Зафиксировать, что *observed_text* был сопоставлен с *canonical_label*.

        Если счётчик наблюдений достигает :attr:`min_observations`, текст
        автоматически добавляется в реестр синонимов.

        Args:
            canonical_label: Метка элемента, к которой было разрешено совпадение.
            observed_text: Необработанный OCR-текст, который был сопоставлен.
        """
        norm_text = observed_text.lower().strip()
        norm_label = canonical_label.lower().strip()

        if norm_text in self.registry.expand(norm_label):
            return

        self._observations[norm_label][norm_text] += 1
        count = self._observations[norm_label][norm_text]

        logger.debug(
            "pattern_observation",
            label=norm_label,
            text=norm_text,
            count=count,
        )

        if count >= self.min_observations:
            self._promote(norm_label, norm_text)

    def _promote(self, canonical_label: str, observed_text: str) -> None:
        """Добавить *observed_text* в реестр синонимов.

        Args:
            canonical_label: Каноническая метка элемента.
            observed_text: Текст для добавления в качестве синонима.
        """
        logger.info(
            "promoting_synonym",
            label=canonical_label,
            synonym=observed_text,
        )
        self.registry.add_group(canonical_label, [observed_text])

    def pending_proposals(self) -> dict[str, list[tuple[str, int]]]:
        """Вернуть наблюдённые паттерны, ещё не достигшие порога.

        Returns:
            Словарь ``canonical_label`` → ``[(text, count)]``, отсортированный
            по убыванию счётчика, для паттернов ниже порога.
        """
        proposals: dict[str, list[tuple[str, int]]] = {}
        for label, texts in self._observations.items():
            pending = [
                (text, count)
                for text, count in texts.items()
                if count < self.min_observations
                and text not in self.registry.expand(label)
            ]
            if pending:
                proposals[label] = sorted(pending, key=lambda x: x[1], reverse=True)
        return proposals

    # ------------------------------------------------------------------
    # Офлайн-анализ логов
    # ------------------------------------------------------------------

    def analyze_logs(self, log_dir: Path | str) -> LearningReport:
        """Проанализировать все файлы ``steps.jsonl`` в директории *log_dir*.

        Ищет файлы по маске ``*/steps.jsonl`` (один уровень вложенности) и
        ``steps.jsonl`` в корне *log_dir*.

        Args:
            log_dir: Корневая директория, содержащая поддиректории отдельных запусков.

        Returns:
            :class:`LearningReport` с агрегированными данными всех наблюдённых выполнений.
        """
        log_dir = Path(log_dir)

        jsonl_files: list[Path] = list(log_dir.glob("*/steps.jsonl"))
        direct = log_dir / "steps.jsonl"
        if direct.exists() and direct not in jsonl_files:
            jsonl_files.append(direct)

        if not jsonl_files:
            logger.info("analyze_logs: no steps.jsonl files found", dir=str(log_dir))
            self._last_report = _EMPTY_REPORT
            return _EMPTY_REPORT

        run_count = len({f.parent for f in jsonl_files})

        step_stats: dict[str, StepStats] = {}
        method_stats: dict[str, dict[str, int]] = {}
        # (find_target, found_text) → {step_id, count}
        mismatch_counts: dict[tuple[str, str], int] = {}
        mismatch_step: dict[tuple[str, str], str] = {}
        total_steps = 0

        for jsonl_path in jsonl_files:
            try:
                with jsonl_path.open("r", encoding="utf-8") as fh:
                    lines = fh.readlines()
            except OSError as exc:
                logger.warning("analyze_logs: cannot read file", path=str(jsonl_path), error=str(exc))
                continue

            for raw_line in lines:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry: dict[str, Any] = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                # Пропускаем граничные события сценария
                if entry.get("type") in ("scenario_start", "scenario_end"):
                    continue

                step_id: str = entry.get("step_id") or "unknown"
                success: bool = bool(entry.get("success", False))
                duration_ms: float = float(entry.get("duration_ms") or 0.0)
                method: str = entry.get("method_used") or "unknown"
                error: str | None = entry.get("error")
                find_target: str | None = entry.get("find_target")
                found_text: str | None = entry.get("found_text")

                total_steps += 1

                # Накапливаем статистику шагов
                if step_id not in step_stats:
                    step_stats[step_id] = StepStats()
                ss = step_stats[step_id]
                ss.count += 1
                ss.durations_ms.append(duration_ms)
                if success:
                    ss.success_count += 1
                else:
                    ss.failure_count += 1
                    if error and error not in ss.errors:
                        ss.errors.append(error)
                ss.methods[method] = ss.methods.get(method, 0) + 1

                # Статистика по методам
                if method not in method_stats:
                    method_stats[method] = {"success": 0, "failure": 0}
                method_stats[method]["success" if success else "failure"] += 1

                # Обнаружение OCR-несоответствий
                if (
                    find_target
                    and found_text
                    and found_text.lower().strip() != find_target.lower().strip()
                ):
                    mk = (find_target, found_text)
                    mismatch_counts[mk] = mismatch_counts.get(mk, 0) + 1
                    mismatch_step[mk] = step_id

        # Строим список несоответствий, отсортированный по частоте
        ocr_mismatches: list[OcrMismatch] = [
            OcrMismatch(
                find_target=key[0],
                found_text=key[1],
                step_id=mismatch_step[key],
                count=cnt,
            )
            for key, cnt in sorted(mismatch_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        # Передаём несоответствия в онлайн-обучение
        for m in ocr_mismatches:
            for _ in range(m.count):
                self.observe(m.find_target, m.found_text)

        # Рекомендуемые таймауты: p95 × 1.2 для шагов с ≥ 3 точками данных
        suggested_timeouts: dict[str, int] = {}
        for sid, ss in step_stats.items():
            if len(ss.durations_ms) >= 3:
                p95 = _percentile(ss.durations_ms, 95)
                suggested_timeouts[sid] = max(1, int(p95 * 1.2))

        self._last_report = LearningReport(
            run_count=run_count,
            total_steps=total_steps,
            step_stats=step_stats,
            ocr_mismatches=ocr_mismatches,
            method_stats=method_stats,
            suggested_timeouts=suggested_timeouts,
        )
        logger.info(
            "analyze_logs complete",
            runs=run_count,
            steps=total_steps,
            mismatches=len(ocr_mismatches),
        )
        return self._last_report

    # ------------------------------------------------------------------
    # Производные предложения
    # ------------------------------------------------------------------

    def suggest_synonyms(self) -> dict[str, list[str]]:
        """Вернуть кандидатов в синонимы, полученных из наблюдений.

        Каждый ключ — это find_target / каноническая метка; значения — OCR-считывания,
        которые были сопоставлены, но ещё не добавлены в реестр синонимов.

        Returns:
            Словарь метка → список строк-кандидатов в синонимы.
        """
        proposals: dict[str, list[str]] = {}
        for label, texts in self._observations.items():
            candidates = [
                text
                for text in texts
                if text not in self.registry.expand(label)
            ]
            if candidates:
                proposals[label] = sorted(candidates)
        return proposals

    def optimize_timeouts(self) -> dict[str, int]:
        """Вернуть предложения по таймаутам из последнего вызова :meth:`analyze_logs`.

        Returns:
            Словарь step_id → рекомендуемый таймаут в миллисекундах
            (p95 наблюдённых длительностей × 1.2). Пустой, если логи
            ещё не анализировались.
        """
        return dict(self._last_report.suggested_timeouts)

    # ------------------------------------------------------------------
    # Сохранение доказательств
    # ------------------------------------------------------------------

    def save_evidence(self) -> None:
        """Сохранить счётчики наблюдений в :attr:`evidence_path`.

        Не выполняет никаких действий, если :attr:`evidence_path` равно ``None``.
        """
        if not self.evidence_path:
            return
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {label: dict(texts) for label, texts in self._observations.items()}
        with self.evidence_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True)
        logger.debug("Evidence saved", path=str(self.evidence_path))

    def load_evidence(self) -> None:
        """Загрузить сохранённые счётчики наблюдений из :attr:`evidence_path`.

        Не выполняет никаких действий, если файл не существует.
        """
        if not self.evidence_path or not self.evidence_path.exists():
            return
        with self.evidence_path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            logger.warning("Evidence file has unexpected format", path=str(self.evidence_path))
            return
        for label, texts in raw.items():
            if isinstance(texts, dict):
                for text, count in texts.items():
                    self._observations[str(label)][str(text)] = int(count)
        logger.debug("Evidence loaded", path=str(self.evidence_path))


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: int) -> float:
    """Вернуть *pct*-й перцентиль значений *values* (метод ближайшего ранга).

    Args:
        values: Непустой список числовых значений.
        pct: Перцентиль в диапазоне ``[0, 100]``.

    Returns:
        Значение перцентиля.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * pct / 100) - 1)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]
