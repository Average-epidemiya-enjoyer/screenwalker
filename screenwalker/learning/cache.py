"""Кэш действий — сохраняет успешно найденные местоположения элементов для быстрого повторного использования.

При попадании в кэш движок может пропустить дорогостоящий vision-конвейер и
сразу перейти к действию, используя сохранённые экранные координаты.

Ключ кэша: ``"{screen_id}::{element_label}"``

Формат файла кэша (JSON)::

    {
      "screen:main_menu/find:text:Создать заявку": {
        "screen_id": "main_menu",
        "element_label": "создать заявку",
        "bbox_x": 450, "bbox_y": 230, "bbox_w": 180, "bbox_h": 35,
        "method": "ocr",
        "confidence": 0.92,
        "timestamp": 1705329000.0,
        "hit_count": 15,
        "last_used": "2024-01-15T14:30:00+00:00",
        "screen_resolution": [1920, 1080]
      }
    }

Обнаружение изменения разрешения экрана
----------------------------------------
При первой загрузке кэша сохранённое ``screen_resolution`` сравнивается
с текущим разрешением дисплея.  Если они отличаются, весь кэш инвалидируется,
чтобы устаревшие координаты BBox не вызывали промахи при кликах.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from screenwalker.vision.screen_state import BBox, FindResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Единичная запись кэша с местоположением элемента.

    Attributes:
        screen_id: Состояние экрана, на котором был найден элемент.
        element_label: Каноническая метка / поисковый текст элемента.
        bbox_x: Левый край BBox (пиксели).
        bbox_y: Верхний край BBox (пиксели).
        bbox_w: Ширина BBox (пиксели).
        bbox_h: Высота BBox (пиксели).
        method: Vision-метод, давший этот результат.
        confidence: Оценка уверенности на момент кэширования.
        timestamp: Unix-временная метка записи.
        hit_count: Количество успешных повторных использований этой записи.
        last_used: ISO-8601 UTC-метка последнего попадания в кэш.
        screen_resolution: Разрешение дисплея ``[ширина, высота]`` на момент кэширования.
    """

    screen_id: str
    element_label: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    method: str
    confidence: float
    timestamp: float
    hit_count: int = 0
    last_used: str = ""
    screen_resolution: list[int] = field(default_factory=list)

    @property
    def bbox(self) -> BBox:
        """Восстановить :class:`~screenwalker.vision.screen_state.BBox`."""
        return BBox(self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h)

    def to_find_result(self) -> FindResult:
        """Преобразовать в :class:`~screenwalker.vision.screen_state.FindResult`.

        Returns:
            FindResult, эквивалентный данной записи кэша.
        """
        return FindResult(
            element=self.element_label,
            confidence=self.confidence,
            bbox=self.bbox,
            method=f"cache:{self.method}",
            metadata={
                "cached_at": self.timestamp,
                "hit_count": self.hit_count,
            },
        )

    def is_fresh(self, ttl: float) -> bool:
        """Вернуть ``True``, если запись находится в пределах окна TTL.

        Args:
            ttl: Максимальный возраст в секундах.
        """
        return (time.time() - self.timestamp) < ttl


# ---------------------------------------------------------------------------
# ActionCache
# ---------------------------------------------------------------------------


class ActionCache:
    """Персистентный кэш местоположений элементов на основе JSON-файла.

    Файл кэша читается лениво при первом обращении и сбрасывается на диск после
    каждой записи.  Если разрешение дисплея изменилось между запусками,
    весь кэш инвалидируется.

    Attributes:
        path: Путь к JSON-файлу кэша.
        ttl: Максимальный возраст записи кэша в секундах (по умолчанию 24 ч).
        enabled: При значении ``False`` все операции становятся заглушками.
    """

    def __init__(
        self,
        path: Path | str = Path("logs/cache.json"),
        ttl: float = 86400.0,
        enabled: bool = True,
    ) -> None:
        """Инициализировать ActionCache.

        Args:
            path: Путь к резервному JSON-файлу.
            ttl: Время жизни записи в секундах.
            enabled: Установить ``False`` для полного обхода кэша.
        """
        self.path = Path(path)
        self.ttl = ttl
        self.enabled = enabled
        self._store: dict[str, CacheEntry] | None = None
        self._current_resolution: list[int] = []

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _get_resolution() -> list[int]:
        """Вернуть текущее разрешение дисплея в виде ``[ширина, высота]``.

        Возвращает пустой список, если PyAutoGUI недоступен (тесты, CI).
        """
        try:
            import pyautogui
            w, h = pyautogui.size()
            return [int(w), int(h)]
        except Exception:
            return []

    def _key(self, screen_id: str, element_label: str) -> str:
        """Построить составной ключ кэша.

        Args:
            screen_id: Идентификатор состояния экрана.
            element_label: Метка элемента (нормализуется до нижнего регистра).

        Returns:
            Строка составного ключа.
        """
        return f"{screen_id}::{element_label.lower().strip()}"

    def _ensure_loaded(self) -> None:
        """Лениво загрузить файл кэша; инвалидировать при изменении разрешения."""
        if self._store is not None:
            return

        current_res = self._get_resolution()
        self._current_resolution = current_res

        if not self.path.exists():
            self._store = {}
            return

        try:
            with self.path.open("r", encoding="utf-8") as fh:
                raw: dict[str, Any] = json.load(fh)
        except Exception as exc:
            logger.warning("Cache load failed, starting empty", error=str(exc))
            self._store = {}
            return

        # Проверка разрешения — сравниваем с первой записью, у которой оно есть
        stored_res: list[int] = []
        for v in raw.values():
            if isinstance(v, dict):
                r = v.get("screen_resolution") or []
                if r:
                    stored_res = r
                    break

        if stored_res and current_res and stored_res != current_res:
            logger.info(
                "Screen resolution changed — cache invalidated",
                old=stored_res,
                new=current_res,
            )
            self._store = {}
            return

        store: dict[str, CacheEntry] = {}
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            try:
                store[k] = CacheEntry(
                    screen_id=v["screen_id"],
                    element_label=v["element_label"],
                    bbox_x=int(v["bbox_x"]),
                    bbox_y=int(v["bbox_y"]),
                    bbox_w=int(v["bbox_w"]),
                    bbox_h=int(v["bbox_h"]),
                    method=v["method"],
                    confidence=float(v["confidence"]),
                    timestamp=float(v["timestamp"]),
                    hit_count=int(v.get("hit_count", 0)),
                    last_used=str(v.get("last_used", "")),
                    screen_resolution=list(v.get("screen_resolution") or []),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed cache entry", key=k)

        self._store = store
        logger.debug("Cache loaded", entries=len(self._store), path=str(self.path))

    # ------------------------------------------------------------------
    # Новый интерфейс: get / put
    # ------------------------------------------------------------------

    def get(self, screen_id: str, find_target: str) -> CacheEntry | None:
        """Вернуть свежую :class:`CacheEntry` или ``None``.

        При попадании увеличивает :attr:`CacheEntry.hit_count` и обновляет
        :attr:`CacheEntry.last_used`.  Устаревшие записи вытесняются.

        Args:
            screen_id: Идентификатор текущего состояния экрана.
            find_target: Метка элемента для поиска.

        Returns:
            :class:`CacheEntry`, если запись существует и ещё свежая,
            иначе ``None``.
        """
        if not self.enabled:
            return None
        self._ensure_loaded()
        assert self._store is not None

        key = self._key(screen_id, find_target)
        entry = self._store.get(key)
        if entry is None:
            return None
        if not entry.is_fresh(self.ttl):
            del self._store[key]
            logger.debug("Cache entry evicted (stale)", key=key)
            return None

        entry.hit_count += 1
        entry.last_used = datetime.now(timezone.utc).isoformat()
        logger.debug("Cache hit", key=key, method=entry.method, hits=entry.hit_count)
        return entry

    def put(
        self,
        screen_id: str,
        find_target: str,
        bbox: tuple[int, int, int, int],
        method: str,
        confidence: float,
    ) -> None:
        """Сохранить местоположение элемента в кэше.

        Сохраняет существующий :attr:`CacheEntry.hit_count`, если запись
        для этого ключа уже существует.

        Args:
            screen_id: Идентификатор текущего состояния экрана.
            find_target: Метка / поисковый текст элемента.
            bbox: BBox ``(x, y, w, h)`` в экранных координатах.
            method: Vision-метод (``ocr``, ``template``, ``yolo``).
            confidence: Уверенность совпадения в диапазоне ``[0.0, 1.0]``.
        """
        if not self.enabled:
            return
        self._ensure_loaded()
        assert self._store is not None

        key = self._key(screen_id, find_target)
        existing = self._store.get(key)
        entry = CacheEntry(
            screen_id=screen_id,
            element_label=find_target,
            bbox_x=bbox[0],
            bbox_y=bbox[1],
            bbox_w=bbox[2],
            bbox_h=bbox[3],
            method=method,
            confidence=confidence,
            timestamp=time.time(),
            hit_count=existing.hit_count if existing else 0,
            last_used=datetime.now(timezone.utc).isoformat(),
            screen_resolution=list(self._current_resolution),
        )
        self._store[key] = entry
        logger.debug("Cache put", key=key, method=method, confidence=confidence)
        self.flush()

    # ------------------------------------------------------------------
    # Интерфейс обратной совместимости: lookup / store
    # ------------------------------------------------------------------

    def lookup(self, screen_id: str, element_label: str) -> FindResult | None:
        """Вернуть кэшированный :class:`FindResult` или ``None`` (обратная совместимость).

        Делегирует вызов :meth:`get` и преобразует результат в :class:`FindResult`.

        Args:
            screen_id: Идентификатор текущего состояния экрана.
            element_label: Искомый элемент.

        Returns:
            :class:`~screenwalker.vision.screen_state.FindResult` или ``None``.
        """
        entry = self.get(screen_id, element_label)
        return entry.to_find_result() if entry is not None else None

    def store(self, screen_id: str, result: FindResult) -> None:
        """Сохранить :class:`FindResult` в кэше (обратная совместимость).

        Делегирует вызов :meth:`put`.

        Args:
            screen_id: Идентификатор текущего состояния экрана.
            result: Экземпляр :class:`FindResult` для кэширования.
        """
        self.put(
            screen_id=screen_id,
            find_target=result.element,
            bbox=(result.bbox.x, result.bbox.y, result.bbox.w, result.bbox.h),
            method=result.method,
            confidence=result.confidence,
        )

    # ------------------------------------------------------------------
    # Инвалидация / обслуживание
    # ------------------------------------------------------------------

    def invalidate(self, screen_id: str, element_label: str | None = None) -> None:
        """Удалить одну или все записи для заданного экрана.

        Args:
            screen_id: Состояние экрана для инвалидации.
            element_label: Если задан, инвалидировать только этот элемент;
                иначе инвалидировать все записи для *screen_id*.
        """
        self._ensure_loaded()
        assert self._store is not None
        if element_label is not None:
            self._store.pop(self._key(screen_id, element_label), None)
        else:
            to_remove = [k for k in self._store if k.startswith(f"{screen_id}::")]
            for k in to_remove:
                del self._store[k]

    def flush(self) -> None:
        """Записать хранилище из памяти в JSON-файл кэша."""
        if self._store is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {k: asdict(v) for k, v in self._store.items()},
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as exc:
            logger.warning("Cache flush failed", error=str(exc))

    def evict_stale(self) -> int:
        """Удалить все записи старше :attr:`ttl`.

        Returns:
            Количество вытесненных записей.
        """
        self._ensure_loaded()
        assert self._store is not None
        stale = [k for k, v in self._store.items() if not v.is_fresh(self.ttl)]
        for k in stale:
            del self._store[k]
        if stale:
            self.flush()
        logger.debug("Evicted stale cache entries", count=len(stale))
        return len(stale)
