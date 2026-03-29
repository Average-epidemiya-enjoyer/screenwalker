"""Нечёткое сопоставление текста с использованием RapidFuzz.

Предоставляет вспомогательные функции нормализации и класс :class:`FuzzyMatcher`,
который оценивает строки запроса относительно корпуса токенов, извлечённых OCR.

Соглашения по оценкам
---------------------
* Функции уровня модуля :func:`fuzzy_score` и :func:`fuzzy_match` используют
  **float**-оценки в ``[0.0, 1.0]`` для обратной совместимости с пайплайном vision.
* Новые методы :class:`FuzzyMatcher` (:meth:`~FuzzyMatcher.match`,
  :meth:`~FuzzyMatcher.best_match`, :meth:`~FuzzyMatcher.contains_fuzzy`) используют
  **int**-оценки в ``[0, 100]`` в соответствии с соглашением RapidFuzz.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz as _fuzz
from rapidfuzz import process as _process


# ---------------------------------------------------------------------------
# Тип результата
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzyResult:
    """Единственный результат нечёткого сопоставления.

    Attributes:
        text: Исходная строка-кандидат (до какой-либо нормализации).
        score: Оценка сходства RapidFuzz в диапазоне ``[0, 100]``.
        original_index: Позиция кандидата во входном списке.
    """

    text: str
    score: int
    original_index: int


# ---------------------------------------------------------------------------
# Нормализация строк
# ---------------------------------------------------------------------------


def normalise(text: str, lowercase: bool = True, strip_whitespace: bool = True) -> str:
    """Нормализовать строку для нечёткого сравнения.

    Применяет Unicode NFC-нормализацию, опциональное приведение к нижнему регистру,
    свёртку пробелов и удаление распространённых несемантических символов.

    Args:
        text: Необработанная входная строка.
        lowercase: Привести к нижнему регистру при значении True.
        strip_whitespace: Свернуть последовательности пробелов в одиночный пробел.

    Returns:
        Нормализованная строка.

    Example:
        >>> normalise("  Hello\\tWorld!  ")
        'hello world!'
    """
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    if strip_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# Псевдоним с английским написанием, чтобы оба имени можно было импортировать
normalize = normalise


# ---------------------------------------------------------------------------
# Вспомогательные функции уровня модуля (float-оценки 0–1, обратная совместимость)
# ---------------------------------------------------------------------------


def fuzzy_score(query: str, candidate: str, normalise_inputs: bool = True) -> float:
    """Вычислить оценку нечёткого сходства между *query* и *candidate*.

    Использует ``rapidfuzz.fuzz.WRatio`` (взвешенное соотношение), которое
    объединяет partial ratio, token sort ratio и token set ratio для наилучших
    результатов на практике.

    Args:
        query: Строка поиска (например, из YAML сценария).
        candidate: Строка-кандидат (например, из OCR-вывода).
        normalise_inputs: Применить :func:`normalise` к обеим строкам перед оценкой.

    Returns:
        Оценка в диапазоне [0.0, 1.0].
    """
    q = normalise(query) if normalise_inputs else query
    c = normalise(candidate) if normalise_inputs else candidate
    return _fuzz.WRatio(q, c) / 100.0


def fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.80,
    normalise_inputs: bool = True,
) -> list[tuple[str, float]]:
    """Найти все кандидаты, нечётко совпадающие с *query* выше *threshold*.

    Args:
        query: Строка поиска.
        candidates: Список строк для сравнения.
        threshold: Минимальная оценка (0.0–1.0) для включения в результаты.
        normalise_inputs: Нормализовать строки перед оценкой.

    Returns:
        Список кортежей ``(candidate, score)``, отсортированных по убыванию оценки,
        отфильтрованных по условию score >= threshold.
    """
    if not candidates:
        return []

    q = normalise(query) if normalise_inputs else query
    normed = [normalise(c) if normalise_inputs else c for c in candidates]

    cutoff = int(threshold * 100)
    raw = _process.extract(
        q,
        normed,
        scorer=_fuzz.WRatio,
        score_cutoff=cutoff,
        limit=None,
    )
    # элементы raw: (matched_text, score, index)
    return sorted(
        [(candidates[idx], score / 100.0) for _, score, idx in raw],
        key=lambda t: t[1],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Класс FuzzyMatcher
# ---------------------------------------------------------------------------


class FuzzyMatcher:
    """Состоятельный нечёткий матчер с настраиваемыми порогами.

    Новые методы (:meth:`match`, :meth:`best_match`, :meth:`contains_fuzzy`)
    используют **int**-оценки 0–100. Устаревшие методы (:meth:`score`, :meth:`all_matches`)
    используют float-оценки 0.0–1.0 для обратной совместимости с пайплайном vision.

    Attributes:
        threshold: Минимальная оценка по умолчанию для новых методов (0–100).
        normalise_inputs: Нормализовать ли строки перед оценкой.
    """

    def __init__(self, threshold: int = 80, normalise_inputs: bool = True) -> None:
        """Инициализировать FuzzyMatcher.

        Args:
            threshold: Пороговая оценка по умолчанию (0–100) для :meth:`match` и
                :meth:`best_match`.
            normalise_inputs: Применять нормализацию строк перед оценкой.
        """
        self.threshold = threshold
        self.normalise_inputs = normalise_inputs

    # ------------------------------------------------------------------
    # Статический вспомогательный метод
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(text: str) -> str:
        """Нормализовать *text* для сравнения (нижний регистр, свёртка пробелов).

        Args:
            text: Необработанная входная строка.

        Returns:
            Нормализованная строка.
        """
        return normalise(text)

    # ------------------------------------------------------------------
    # Новые методы — int-оценки 0–100
    # ------------------------------------------------------------------

    def match(
        self,
        query: str,
        candidates: list[str],
        threshold: int | None = None,
    ) -> list[FuzzyResult]:
        """Найти все кандидаты, нечётко совпадающие с *query* выше *threshold*.

        Использует ``rapidfuzz.fuzz.token_sort_ratio`` как основную оценку и
        ``rapidfuzz.process.extract`` для эффективной пакетной обработки.

        Args:
            query: Строка поиска.
            candidates: Пул строк-кандидатов.
            threshold: Минимальная оценка 0–100 (по умолчанию :attr:`threshold`).

        Returns:
            Список :class:`FuzzyResult`, отсортированных по убыванию оценки.
        """
        if not candidates:
            return []

        cutoff = threshold if threshold is not None else self.threshold
        q = normalise(query) if self.normalise_inputs else query
        normed = [normalise(c) if self.normalise_inputs else c for c in candidates]

        raw = _process.extract(
            q,
            normed,
            scorer=_fuzz.token_sort_ratio,
            score_cutoff=cutoff,
            limit=None,
        )
        results = [
            FuzzyResult(text=candidates[idx], score=int(score), original_index=idx)
            for _, score, idx in raw
        ]
        return sorted(results, key=lambda r: r.score, reverse=True)

    def best_match(
        self,
        query: str,
        candidates: list[str],
        threshold: int | None = None,
    ) -> FuzzyResult | None:
        """Вернуть наилучшего кандидата, превышающего *threshold*.

        Args:
            query: Строка поиска.
            candidates: Пул строк-кандидатов.
            threshold: Минимальная оценка 0–100 (по умолчанию :attr:`threshold`).

        Returns:
            :class:`FuzzyResult` для наилучшего совпадения, или None, если ни один
            не превысил порог.
        """
        if not candidates:
            return None

        cutoff = threshold if threshold is not None else self.threshold
        q = normalise(query) if self.normalise_inputs else query
        normed = {i: (normalise(c) if self.normalise_inputs else c) for i, c in enumerate(candidates)}

        result = _process.extractOne(
            q,
            normed,
            scorer=_fuzz.token_sort_ratio,
            score_cutoff=cutoff,
        )
        if result is None:
            return None
        _matched_text, score, idx = result
        return FuzzyResult(text=candidates[idx], score=int(score), original_index=idx)

    def contains_fuzzy(
        self,
        text: str,
        substring: str,
        threshold: int = 85,
    ) -> bool:
        """Проверить, содержится ли *substring* внутри *text* с нечётким сопоставлением.

        Использует ``rapidfuzz.fuzz.partial_ratio``, который находит наилучшее
        выравнивание более короткой строки внутри более длинной.

        Args:
            text: Более длинная строка, в которой выполняется поиск.
            substring: Более короткая строка для поиска.
            threshold: Минимальная оценка ``partial_ratio`` (0–100).

        Returns:
            True, если оценка наилучшего частичного совпадения >= *threshold*.
        """
        t = normalise(text) if self.normalise_inputs else text
        s = normalise(substring) if self.normalise_inputs else substring
        return int(_fuzz.partial_ratio(s, t)) >= threshold

    # ------------------------------------------------------------------
    # Устаревшие методы — float-оценки 0.0–1.0
    # ------------------------------------------------------------------

    def score(self, query: str, candidate: str) -> float:
        """Оценить *query* относительно одного *candidate*.

        Args:
            query: Строка поиска.
            candidate: Кандидат для оценки.

        Returns:
            Оценка сходства в [0.0, 1.0] (WRatio).
        """
        return fuzzy_score(query, candidate, normalise_inputs=self.normalise_inputs)

    def all_matches(
        self, query: str, candidates: list[str], threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """Вернуть всех кандидатов, совпадающих с *query* выше порога.

        Args:
            query: Строка поиска.
            candidates: Пул строк-кандидатов.
            threshold: Переопределить порог экземпляра для этого вызова (0.0–1.0).

        Returns:
            Отсортированный список кортежей ``(candidate, score)``, оценки в [0.0, 1.0].
        """
        t = threshold if threshold is not None else self.threshold / 100.0
        return fuzzy_match(
            query, candidates, threshold=t, normalise_inputs=self.normalise_inputs
        )
