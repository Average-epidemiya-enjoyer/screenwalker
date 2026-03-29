"""Генератор HTML-отчётов о запуске сценариев.

Создаёт standalone HTML-файл (все стили и изображения встроены inline),
который открывается в любом браузере без веб-сервера.

Пример использования::

    from screenwalker.reporting.html_report import RunReportGenerator, RunResult

    result = RunResult.from_log_dir(Path("logs/my_run"))
    generator = RunReportGenerator()
    report_path = generator.generate(result, Path("report.html"))
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _h(s: str | None) -> str:
    """HTML-экранирование строки (None → пустая строка)."""
    return html.escape(str(s)) if s is not None else ""


def _parse_dt(raw: str | None) -> datetime:
    """Разобрать ISO-8601 дату; вернуть текущее время при ошибке."""
    if not raw:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _confidence_color(conf: float) -> str:
    """Вернуть CSS-цвет для полоски уверенности."""
    if conf >= 0.9:
        return "#3fb950"
    if conf >= 0.7:
        return "#f0883e"
    return "#f85149"


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------


class StepReportRecord(BaseModel):
    """Данные одного шага для HTML-отчёта.

    Attributes:
        index: Порядковый номер шага (1-based).
        step_id: Идентификатор шага из YAML.
        action: Тип действия (click, type, wait и т.д.).
        description: Человекочитаемое описание шага.
        success: Успешно ли выполнен шаг.
        elapsed_ms: Время выполнения в миллисекундах.
        find_target: Искомый текст или имя шаблона.
        method_used: Метод поиска (ocr / template / yolo / cache).
        confidence: Уверенность совпадения [0.0, 1.0].
        bbox: Координаты найденного элемента (x, y, w, h).
        found_text: Фактический текст, найденный OCR.
        error: Текст ошибки, если шаг упал.
        screenshot_path: Путь к скриншоту этого шага.
    """

    index: int
    step_id: str
    action: str
    description: str = ""
    success: bool
    elapsed_ms: float
    find_target: str | None = None
    method_used: str | None = None
    confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    found_text: str | None = None
    error: str | None = None
    screenshot_path: Path | None = None

    model_config = {"arbitrary_types_allowed": True}


class RunResult(BaseModel):
    """Полный результат запуска сценария для генерации отчёта.

    Attributes:
        scenario_name: Название сценария.
        run_id: Уникальный идентификатор запуска (обычно временна́я метка).
        started_at: Время начала запуска.
        finished_at: Время завершения запуска.
        success: True, если все шаги прошли успешно.
        steps: Список записей по каждому шагу.
    """

    scenario_name: str
    run_id: str
    started_at: datetime
    finished_at: datetime
    success: bool
    steps: list[StepReportRecord]

    # ------------------------------------------------------------------
    # Вычисляемые свойства
    # ------------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        """Общая длительность запуска в секундах."""
        return max(0.0, (self.finished_at - self.started_at).total_seconds())

    @property
    def pass_rate(self) -> float:
        """Доля успешных шагов [0.0, 1.0]."""
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.success) / len(self.steps)

    @property
    def avg_step_ms(self) -> float:
        """Среднее время выполнения шага в миллисекундах."""
        if not self.steps:
            return 0.0
        return sum(s.elapsed_ms for s in self.steps) / len(self.steps)

    @property
    def slowest_step(self) -> StepReportRecord | None:
        """Самый медленный шаг или None, если шагов нет."""
        if not self.steps:
            return None
        return max(self.steps, key=lambda s: s.elapsed_ms)

    # ------------------------------------------------------------------
    # Фабричные методы
    # ------------------------------------------------------------------

    @classmethod
    def from_context(
        cls,
        context: Any,
        started_at: datetime,
        finished_at: datetime,
    ) -> "RunResult":
        """Создать RunResult из живого RunContext после выполнения сценария.

        Args:
            context: Экземпляр RunContext с историей шагов.
            started_at: Момент начала запуска.
            finished_at: Момент завершения запуска.

        Returns:
            Заполненный RunResult.
        """
        screenshots_dir = context.output_dir / "screenshots"
        steps: list[StepReportRecord] = []

        for idx, record in enumerate(context.history, start=1):
            screenshot_path = _find_screenshot(screenshots_dir, idx)
            meta: dict[str, Any] = record.metadata or {}
            bbox = _extract_bbox(meta.get("bbox"))

            steps.append(StepReportRecord(
                index=idx,
                step_id=record.step_id,
                action=record.action,
                success=record.success,
                elapsed_ms=record.elapsed * 1000.0,
                find_target=meta.get("find_target"),
                method_used=meta.get("method_used"),
                confidence=meta.get("confidence"),
                bbox=bbox,
                found_text=meta.get("found_text"),
                error=record.error,
                screenshot_path=screenshot_path,
            ))

        return cls(
            scenario_name=context.scenario_name,
            run_id=context.run_id,
            started_at=started_at,
            finished_at=finished_at,
            success=len(context.failed_steps) == 0,
            steps=steps,
        )

    @classmethod
    def from_log_dir(cls, log_dir: Path) -> "RunResult":
        """Загрузить RunResult из директории с логами (steps.jsonl + screenshots/).

        Args:
            log_dir: Директория, содержащая steps.jsonl и папку screenshots/.

        Returns:
            Заполненный RunResult.

        Raises:
            FileNotFoundError: Если steps.jsonl не найден.
        """
        jsonl_path = log_dir / "steps.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Файл steps.jsonl не найден в {log_dir}")

        entries: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        scenario_start = next((e for e in entries if e.get("type") == "scenario_start"), None)
        scenario_end = next((e for e in entries if e.get("type") == "scenario_end"), None)
        step_entries = [e for e in entries if "step_id" in e and e.get("type") is None]

        scenario_name = (
            scenario_start.get("scenario", log_dir.name)
            if scenario_start else log_dir.name
        )
        now = datetime.now(timezone.utc)
        started_at = _parse_dt(scenario_start.get("timestamp")) if scenario_start else now
        finished_at = _parse_dt(scenario_end.get("timestamp")) if scenario_end else now
        success = bool(scenario_end.get("success", False)) if scenario_end else False

        screenshots_dir = log_dir / "screenshots"
        steps: list[StepReportRecord] = []

        for idx, entry in enumerate(step_entries, start=1):
            steps.append(StepReportRecord(
                index=idx,
                step_id=entry.get("step_id", f"step_{idx}"),
                action=entry.get("action", ""),
                success=bool(entry.get("success", False)),
                elapsed_ms=float(entry.get("duration_ms", 0.0)),
                find_target=entry.get("find_target"),
                method_used=entry.get("method_used"),
                confidence=entry.get("confidence"),
                bbox=_extract_bbox(entry.get("bbox")),
                found_text=entry.get("found_text"),
                error=entry.get("error"),
                screenshot_path=_find_screenshot(screenshots_dir, idx),
            ))

        return cls(
            scenario_name=scenario_name,
            run_id=log_dir.name,
            started_at=started_at,
            finished_at=finished_at,
            success=success,
            steps=steps,
        )


# ---------------------------------------------------------------------------
# Генератор отчётов
# ---------------------------------------------------------------------------


class RunReportGenerator:
    """Генерирует standalone HTML-отчёт из RunResult.

    Все стили и изображения встраиваются inline — файл открывается
    в любом браузере без веб-сервера и внешних зависимостей.

    Args:
        thumbnail_max_width: Максимальная ширина миниатюры скриншота в пикселях.
        jpeg_quality: Качество JPEG для миниатюр (10–100).
        include_screenshots: Включать ли скриншоты в отчёт.
    """

    def __init__(
        self,
        thumbnail_max_width: int = 400,
        jpeg_quality: int = 72,
        include_screenshots: bool = True,
    ) -> None:
        self._thumb_width = thumbnail_max_width
        self._jpeg_quality = jpeg_quality
        self._include_screenshots = include_screenshots

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def generate(self, run_result: RunResult, output_path: Path) -> Path:
        """Создать HTML-отчёт и записать в файл.

        Args:
            run_result: Данные о запуске сценария.
            output_path: Путь к выходному HTML-файлу.

        Returns:
            Абсолютный путь к сохранённому файлу.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self._render_html(run_result)
        output_path.write_text(content, encoding="utf-8")

        log.info(
            "report.generated",
            path=str(output_path),
            scenario=run_result.scenario_name,
            steps=len(run_result.steps),
            success=run_result.success,
        )
        return output_path.resolve()

    # ------------------------------------------------------------------
    # Рендеринг HTML
    # ------------------------------------------------------------------

    def _render_html(self, result: RunResult) -> str:
        title = _h(result.scenario_name)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="ru">\n'
            "<head>\n"
            '  <meta charset="UTF-8">\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"  <title>ScreenWalker — {title}</title>\n"
            f"  <style>{self._css()}</style>\n"
            "</head>\n"
            "<body>\n"
            '  <div class="container">\n'
            f"    {self._render_header(result)}\n"
            f"    {self._render_stats(result)}\n"
            f"    {self._render_steps(result)}\n"
            '    <div class="footer">Создано ScreenWalker &mdash; '
            f"{_h(result.started_at.strftime('%d.%m.%Y %H:%M:%S'))}</div>\n"
            "  </div>\n"
            '  <div id="modal" onclick="closeModal()">\n'
            '    <img id="modal-img" src="" alt="скриншот">\n'
            '    <span class="modal-close" onclick="closeModal()">&#x2715;</span>\n'
            "  </div>\n"
            f"  <script>{self._js()}</script>\n"
            "</body>\n"
            "</html>"
        )

    def _render_header(self, result: RunResult) -> str:
        badge_cls = "badge-pass" if result.success else "badge-fail"
        badge_text = "PASS" if result.success else "FAIL"
        dt_str = result.started_at.strftime("%d.%m.%Y %H:%M:%S")
        dur = f"{result.duration_s:.1f} с"
        return (
            '<div class="header">\n'
            '  <div class="header-main">\n'
            f'    <div class="scenario-name">{_h(result.scenario_name)}</div>\n'
            '    <div class="run-meta">\n'
            f'      <span>&#128197; {_h(dt_str)}</span>\n'
            f'      <span>&#9201; {_h(dur)}</span>\n'
            f'      <span>&#128190; {_h(result.run_id)}</span>\n'
            "    </div>\n"
            "  </div>\n"
            f'  <span class="badge {badge_cls}">{badge_text}</span>\n'
            "</div>"
        )

    def _render_stats(self, result: RunResult) -> str:
        total = len(result.steps)
        passed = sum(1 for s in result.steps if s.success)
        failed = total - passed
        pass_pct = f"{result.pass_rate * 100:.0f}%"
        avg_ms = f"{result.avg_step_ms:.0f} мс"
        slowest = result.slowest_step
        slowest_str = (
            f"{_h(slowest.step_id)} ({slowest.elapsed_ms:.0f} мс)"
            if slowest else "—"
        )

        pass_cls = "pass" if failed == 0 else "fail"

        return (
            '<div class="stats">\n'
            f'  {self._stat_card(pass_pct, "% успеха", pass_cls)}\n'
            f'  {self._stat_card(str(total), "Всего шагов", "accent")}\n'
            f'  {self._stat_card(str(passed), "Прошло", "pass")}\n'
            f'  {self._stat_card(str(failed), "Упало", "fail" if failed else "accent")}\n'
            f'  {self._stat_card(avg_ms, "Среднее время", "accent")}\n'
            f'  {self._stat_card_wide(slowest_str, "Самый медленный шаг")}\n'
            "</div>"
        )

    @staticmethod
    def _stat_card(value: str, label: str, cls: str = "accent") -> str:
        return (
            '<div class="stat-card">\n'
            f'  <div class="stat-value {cls}">{_h(value)}</div>\n'
            f'  <div class="stat-label">{_h(label)}</div>\n'
            "</div>"
        )

    @staticmethod
    def _stat_card_wide(value: str, label: str) -> str:
        return (
            '<div class="stat-card stat-wide">\n'
            f'  <div class="stat-value-wide">{value}</div>\n'
            f'  <div class="stat-label">{_h(label)}</div>\n'
            "</div>"
        )

    def _render_steps(self, result: RunResult) -> str:
        rows = "\n".join(self._render_step_row(step) for step in result.steps)
        return (
            '<p class="section-title">Шаги выполнения</p>\n'
            '<div class="table-wrap">\n'
            "  <table>\n"
            "    <thead>\n"
            "      <tr>\n"
            '        <th style="width:36px">#</th>\n'
            '        <th style="width:180px">ID шага</th>\n'
            '        <th style="width:110px">Действие</th>\n'
            "        <th>Цель / Метод</th>\n"
            '        <th style="width:110px">Уверенность</th>\n'
            '        <th style="width:80px">Время</th>\n'
            '        <th style="width:50px">Статус</th>\n'
            '        <th style="width:220px">Скриншот</th>\n'
            "      </tr>\n"
            "    </thead>\n"
            "    <tbody>\n"
            f"{rows}\n"
            "    </tbody>\n"
            "  </table>\n"
            "</div>"
        )

    def _render_step_row(self, step: StepReportRecord) -> str:
        row_cls = "step-pass" if step.success else "step-fail"
        status_icon = '<span class="status-ok">&#10003;</span>' if step.success else '<span class="status-fail">&#10007;</span>'

        # Цель поиска + метод
        target_html = ""
        if step.find_target:
            target_html += f'<div class="find-target">{_h(step.find_target)}</div>'
        if step.method_used:
            target_html += f'<div class="find-method">&#128269; {_h(step.method_used)}'
            if step.found_text and step.found_text != step.find_target:
                target_html += f' &rarr; <em>{_h(step.found_text)}</em>'
            target_html += "</div>"

        # Полоска уверенности
        conf_html = ""
        if step.confidence is not None:
            pct = int(step.confidence * 100)
            color = _confidence_color(step.confidence)
            conf_html = (
                f'<span style="font-size:12px">{pct}%</span>'
                '<div class="confidence-bar">'
                f'<div class="confidence-fill" style="width:{pct}%;background:{color}"></div>'
                "</div>"
            )

        # Время (оранжевый, если > 3× среднего)
        elapsed_cls = "elapsed"
        elapsed_str = f"{step.elapsed_ms:.0f} мс"

        # Ошибка
        error_html = ""
        if step.error:
            error_html = f'<div class="error-box">{_h(step.error)}</div>'

        # Скриншот
        screenshot_html = ""
        if self._include_screenshots and step.screenshot_path:
            b64 = self._load_and_annotate(step)
            if b64:
                thumb_cls = "thumb" if step.success else "thumb failed-thumb"
                img_id = f"img-{step.index}"
                screenshot_html = (
                    f'<div class="thumb-wrap">'
                    f'<img id="{img_id}" class="{thumb_cls}" '
                    f'src="data:image/jpeg;base64,{b64}" '
                    f'alt="шаг {step.index}" '
                    f'onclick="openModal(this.src)">'
                    f"</div>"
                )

        # Собрать ячейки
        cells = (
            f'      <tr class="{row_cls}">\n'
            f'        <td class="step-index">{step.index}</td>\n'
            f'        <td><span class="step-id">{_h(step.step_id)}</span>'
            f'{"<br><span class=\"step-desc\">" + _h(step.description) + "</span>" if step.description else ""}'
            f"</td>\n"
            f'        <td><span class="step-action">{_h(step.action)}</span></td>\n'
            f'        <td>{target_html}{error_html}</td>\n'
            f'        <td>{conf_html}</td>\n'
            f'        <td class="{elapsed_cls}">{_h(elapsed_str)}</td>\n'
            f"        <td>{status_icon}</td>\n"
            f"        <td>{screenshot_html}</td>\n"
            "      </tr>"
        )
        return cells

    # ------------------------------------------------------------------
    # Обработка изображений
    # ------------------------------------------------------------------

    def _load_and_annotate(self, step: StepReportRecord) -> str | None:
        """Загрузить скриншот, нарисовать bbox и вернуть base64 JPEG.

        Args:
            step: Запись шага с путём к скриншоту.

        Returns:
            Base64-строка или None при ошибке.
        """
        if step.screenshot_path is None or not step.screenshot_path.exists():
            return None
        try:
            image = Image.open(step.screenshot_path)
            image = _annotate_screenshot(image, step.bbox, step.success)
            return _image_to_base64_jpeg(image, self._thumb_width, self._jpeg_quality)
        except Exception as exc:
            log.warning("report.screenshot_load_failed", path=str(step.screenshot_path), error=str(exc))
            return None

    # ------------------------------------------------------------------
    # CSS и JavaScript
    # ------------------------------------------------------------------

    @staticmethod
    def _css() -> str:
        return """
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--pass:#3fb950;--fail:#f85149;--accent:#58a6ff;--warn:#f0883e;--font:'Cascadia Code','Fira Code','JetBrains Mono','Courier New',monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;line-height:1.6}
.container{max-width:1440px;margin:0 auto;padding:32px 24px}
.header{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:28px 32px;margin-bottom:24px;display:flex;flex-wrap:wrap;align-items:center;gap:24px}
.header-main{flex:1;min-width:200px}
.scenario-name{font-size:20px;font-weight:700;color:var(--text);margin-bottom:8px;letter-spacing:-.02em}
.run-meta{color:var(--muted);font-size:12px}
.run-meta span{margin-right:20px}
.badge{padding:7px 18px;border-radius:24px;font-weight:700;font-size:13px;letter-spacing:.05em}
.badge-pass{background:rgba(63,185,80,.1);color:var(--pass);border:1.5px solid var(--pass)}
.badge-fail{background:rgba(248,81,73,.1);color:var(--fail);border:1.5px solid var(--fail)}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-bottom:28px}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 20px}
.stat-wide{grid-column:span 2}
.stat-value{font-size:24px;font-weight:700;margin-bottom:4px}
.stat-value.pass{color:var(--pass)}
.stat-value.fail{color:var(--fail)}
.stat-value.accent{color:var(--accent)}
.stat-value-wide{font-size:14px;font-weight:600;color:var(--warn);margin-bottom:4px;word-break:break-all}
.stat-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.section-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px}
.table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:24px}
table{width:100%;border-collapse:collapse}
th{background:rgba(255,255,255,.025);color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:9px 14px;text-align:left;border-bottom:1px solid var(--border)}
td{padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.035);vertical-align:top}
tr:last-child td{border-bottom:none}
tr.step-pass:hover td{background:rgba(255,255,255,.018)}
tr.step-fail{background:rgba(248,81,73,.04)}
tr.step-fail:hover td{background:rgba(248,81,73,.08)}
.step-index{color:var(--muted);font-size:12px;text-align:center}
.step-id{font-weight:700;color:var(--text)}
.step-desc{font-size:11px;color:var(--muted)}
.step-action{display:inline-block;padding:2px 8px;background:rgba(88,166,255,.1);color:var(--accent);border-radius:4px;font-size:11px}
.find-target{color:var(--text)}
.find-method{font-size:11px;color:var(--muted);margin-top:2px}
.find-method em{color:var(--accent);font-style:normal}
.confidence-bar{height:4px;border-radius:2px;background:var(--border);margin-top:5px;overflow:hidden}
.confidence-fill{height:100%;border-radius:2px}
.elapsed{color:var(--text);text-align:right;white-space:nowrap}
.status-ok{color:var(--pass);font-size:16px}
.status-fail{color:var(--fail);font-size:16px}
.error-box{margin-top:7px;padding:7px 10px;background:rgba(248,81,73,.1);border-left:3px solid var(--fail);border-radius:0 4px 4px 0;color:var(--fail);font-size:11px;white-space:pre-wrap;word-break:break-word;max-width:420px}
.thumb-wrap{display:inline-block}
.thumb{max-width:200px;max-height:120px;border:1px solid var(--border);border-radius:4px;cursor:zoom-in;display:block;transition:border-color .15s,transform .15s}
.thumb:hover{border-color:var(--accent);transform:scale(1.03)}
.failed-thumb{border:2px solid var(--fail) !important}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.93);z-index:9999;cursor:zoom-out;align-items:center;justify-content:center}
#modal.open{display:flex}
#modal-img{max-width:94vw;max-height:92vh;border-radius:6px;box-shadow:0 8px 60px rgba(0,0,0,.9);pointer-events:none}
.modal-close{position:fixed;top:18px;right:22px;color:#fff;font-size:22px;cursor:pointer;opacity:.6;transition:opacity .15s;line-height:1;user-select:none}
.modal-close:hover{opacity:1}
.footer{margin-top:28px;text-align:center;color:var(--muted);font-size:11px;padding-bottom:16px}
@media(max-width:768px){.stats{grid-template-columns:1fr 1fr}.stat-wide{grid-column:span 1}.thumb{max-width:120px;max-height:80px}}
"""

    @staticmethod
    def _js() -> str:
        return """
function openModal(src){
  document.getElementById('modal-img').src=src;
  document.getElementById('modal').classList.add('open');
}
function closeModal(){
  document.getElementById('modal').classList.remove('open');
  document.getElementById('modal-img').src='';
}
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeModal();});
"""


# ---------------------------------------------------------------------------
# Вспомогательные функции для изображений
# ---------------------------------------------------------------------------


def _annotate_screenshot(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    success: bool,
) -> Image.Image:
    """Нарисовать прямоугольник bbox на копии скриншота.

    Args:
        image: Исходное изображение (не изменяется).
        bbox: Координаты (x, y, w, h) найденного элемента.
        success: True → зелёная рамка, False → красная.

    Returns:
        Новое изображение с нарисованным прямоугольником.
    """
    img = image.copy()
    if bbox is not None:
        x, y, w, h = bbox
        color = (63, 185, 80) if success else (248, 81, 73)
        draw = ImageDraw.Draw(img)
        # Основная рамка
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        # Полупрозрачная заливка через альфа-слой
        if img.mode == "RGBA":
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)
            fill = (63, 185, 80, 30) if success else (248, 81, 73, 30)
            ov_draw.rectangle([x, y, x + w, y + h], fill=fill)
            img = Image.alpha_composite(img, overlay)
    return img


def _image_to_base64_jpeg(
    image: Image.Image,
    max_width: int,
    quality: int,
) -> str:
    """Изменить размер и закодировать изображение в base64 JPEG.

    Args:
        image: PIL-изображение.
        max_width: Максимальная ширина в пикселях (высота масштабируется пропорционально).
        quality: Качество JPEG (10–100).

    Returns:
        Base64-строка без переносов строк.
    """
    if image.width > max_width:
        ratio = max_width / image.width
        new_h = max(1, int(image.height * ratio))
        image = image.resize((max_width, new_h), Image.LANCZOS)

    # JPEG не поддерживает прозрачность
    if image.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", image.size, (13, 17, 23))
        if image.mode == "P":
            image = image.convert("RGBA")
        if image.mode in ("RGBA", "LA"):
            bg.paste(image, mask=image.split()[-1])
        else:
            bg.paste(image)
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _find_screenshot(screenshots_dir: Path, step_index: int) -> Path | None:
    """Найти скриншот шага в директории screenshots/.

    Args:
        screenshots_dir: Директория с PNG-файлами скриншотов.
        step_index: Порядковый номер шага (1-based).

    Returns:
        Путь к файлу или None, если не найден.
    """
    if not screenshots_dir.exists():
        return None
    exact = screenshots_dir / f"step_{step_index:03d}_after.png"
    if exact.exists():
        return exact
    candidates = sorted(screenshots_dir.glob(f"step_{step_index:03d}_*.png"))
    return candidates[-1] if candidates else None


def _extract_bbox(
    raw: Any,
) -> tuple[int, int, int, int] | None:
    """Преобразовать сырые данные bbox в типизированный кортеж.

    Args:
        raw: Список/кортеж из 4 чисел или None.

    Returns:
        Кортеж (x, y, w, h) или None.
    """
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        except (TypeError, ValueError):
            pass
    return None
