"""Action cache — persists successful element locations for fast reuse.

On a cache hit the engine can skip the expensive vision pipeline and go
straight to the action using stored screen coordinates.

Cache key: ``"{screen_id}::{element_label}"``

Cache file format (JSON)::

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

Resolution change detection
----------------------------
When the cache is first loaded, the stored ``screen_resolution`` is compared
against the current display resolution.  If they differ, the entire cache is
invalidated to prevent stale bounding-box coordinates from causing mis-clicks.
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
    """A single cached element location.

    Attributes:
        screen_id: Screen state where the element was found.
        element_label: Canonical label / search text for the element.
        bbox_x: Bounding box left edge (pixels).
        bbox_y: Bounding box top edge (pixels).
        bbox_w: Bounding box width (pixels).
        bbox_h: Bounding box height (pixels).
        method: Vision method that produced this result.
        confidence: Confidence score at time of caching.
        timestamp: Unix timestamp when the entry was written.
        hit_count: Number of times this entry has been successfully reused.
        last_used: ISO-8601 UTC timestamp of the last cache hit.
        screen_resolution: Display resolution ``[width, height]`` at cache time.
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
        """Reconstruct the :class:`~screenwalker.vision.screen_state.BBox`."""
        return BBox(self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h)

    def to_find_result(self) -> FindResult:
        """Convert to a :class:`~screenwalker.vision.screen_state.FindResult`.

        Returns:
            FindResult equivalent of this cache entry.
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
        """Return ``True`` if the entry is within the TTL window.

        Args:
            ttl: Maximum age in seconds.
        """
        return (time.time() - self.timestamp) < ttl


# ---------------------------------------------------------------------------
# ActionCache
# ---------------------------------------------------------------------------


class ActionCache:
    """JSON-backed persistent cache for element locations.

    The cache file is read lazily on first access and flushed to disk after
    every write.  If the display resolution changes between runs, the whole
    cache is invalidated.

    Attributes:
        path: Path to the JSON cache file.
        ttl: Maximum age of a cache entry in seconds (default 24 h).
        enabled: When ``False``, all operations are no-ops.
    """

    def __init__(
        self,
        path: Path | str = Path("logs/cache.json"),
        ttl: float = 86400.0,
        enabled: bool = True,
    ) -> None:
        """Initialise ActionCache.

        Args:
            path: Path to the backing JSON file.
            ttl: Entry time-to-live in seconds.
            enabled: Set to ``False`` to bypass the cache entirely.
        """
        self.path = Path(path)
        self.ttl = ttl
        self.enabled = enabled
        self._store: dict[str, CacheEntry] | None = None
        self._current_resolution: list[int] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_resolution() -> list[int]:
        """Return the current display resolution as ``[width, height]``.

        Returns an empty list when PyAutoGUI is unavailable (tests, CI).
        """
        try:
            import pyautogui
            w, h = pyautogui.size()
            return [int(w), int(h)]
        except Exception:
            return []

    def _key(self, screen_id: str, element_label: str) -> str:
        """Build the composite cache key.

        Args:
            screen_id: Screen state identifier.
            element_label: Element label (normalised to lowercase).

        Returns:
            Composite key string.
        """
        return f"{screen_id}::{element_label.lower().strip()}"

    def _ensure_loaded(self) -> None:
        """Lazily load the cache file; invalidate if resolution changed."""
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

        # Resolution check — compare against the first entry that has one
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
    # New-style interface: get / put
    # ------------------------------------------------------------------

    def get(self, screen_id: str, find_target: str) -> CacheEntry | None:
        """Return a fresh :class:`CacheEntry` or ``None``.

        Increments :attr:`CacheEntry.hit_count` and updates
        :attr:`CacheEntry.last_used` on a hit.  Evicts stale entries.

        Args:
            screen_id: Current screen state identifier.
            find_target: The element label to look up.

        Returns:
            :class:`CacheEntry` if the entry exists and is still fresh,
            otherwise ``None``.
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
        """Store an element location in the cache.

        Preserves the existing :attr:`CacheEntry.hit_count` if an entry
        for the same key already exists.

        Args:
            screen_id: Current screen state identifier.
            find_target: The element label / search text.
            bbox: Bounding box ``(x, y, w, h)`` in screen coordinates.
            method: Vision method (``ocr``, ``template``, ``yolo``).
            confidence: Match confidence in ``[0.0, 1.0]``.
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
    # Backward-compatible interface: lookup / store
    # ------------------------------------------------------------------

    def lookup(self, screen_id: str, element_label: str) -> FindResult | None:
        """Return a cached :class:`FindResult` or ``None`` (backward compat).

        Delegates to :meth:`get` and converts to :class:`FindResult`.

        Args:
            screen_id: Current screen state identifier.
            element_label: Element being searched.

        Returns:
            :class:`~screenwalker.vision.screen_state.FindResult` or ``None``.
        """
        entry = self.get(screen_id, element_label)
        return entry.to_find_result() if entry is not None else None

    def store(self, screen_id: str, result: FindResult) -> None:
        """Persist a :class:`FindResult` to the cache (backward compat).

        Delegates to :meth:`put`.

        Args:
            screen_id: Current screen state identifier.
            result: The :class:`FindResult` to cache.
        """
        self.put(
            screen_id=screen_id,
            find_target=result.element,
            bbox=(result.bbox.x, result.bbox.y, result.bbox.w, result.bbox.h),
            method=result.method,
            confidence=result.confidence,
        )

    # ------------------------------------------------------------------
    # Invalidation / maintenance
    # ------------------------------------------------------------------

    def invalidate(self, screen_id: str, element_label: str | None = None) -> None:
        """Remove one or all entries for a screen.

        Args:
            screen_id: Screen state to invalidate.
            element_label: If given, invalidate only this element; otherwise
                invalidate all entries for *screen_id*.
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
        """Write the in-memory store to the JSON cache file."""
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
        """Remove all entries older than :attr:`ttl`.

        Returns:
            Number of entries evicted.
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
