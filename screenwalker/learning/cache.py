"""Action cache — persists successful element locations for fast reuse.

On a cache hit the engine can skip the expensive vision pipeline and go
straight to the action using stored screen coordinates.

Cache key: ``(screen_id, element_label)``
Cache value: ``{ bbox, method, confidence, timestamp }``
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog

from screenwalker.vision.screen_state import BBox, FindResult

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """A single cached action result.

    Attributes:
        screen_id: Identifier of the screen state where the element was found.
        element_label: Canonical label of the element.
        bbox_x: Bounding box left edge.
        bbox_y: Bounding box top edge.
        bbox_w: Bounding box width.
        bbox_h: Bounding box height.
        method: Vision method that produced the result (ocr/template/yolo).
        confidence: Confidence at time of caching.
        timestamp: Unix timestamp when the entry was written.
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

    @property
    def bbox(self) -> BBox:
        """Reconstruct the BBox from stored fields."""
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
            metadata={"cached_at": self.timestamp},
        )

    def is_fresh(self, ttl: float) -> bool:
        """Check whether this entry is within the TTL window.

        Args:
            ttl: Maximum age in seconds.

        Returns:
            True if the entry was written less than *ttl* seconds ago.
        """
        return (time.time() - self.timestamp) < ttl


class ActionCache:
    """JSON-backed persistent cache for element locations.

    The cache file is read lazily on the first access and flushed to disk
    after every write.

    Attributes:
        path: Path to the JSON cache file.
        ttl: Maximum age of a cache entry in seconds.
        enabled: When False, all lookups return None and writes are no-ops.
    """

    def __init__(
        self,
        path: Path | str = Path("logs/cache.json"),
        ttl: float = 86400.0,
        enabled: bool = True,
    ) -> None:
        """Initialize ActionCache.

        Args:
            path: Path to the backing JSON file.
            ttl: Entry time-to-live in seconds (default 24 h).
            enabled: Disable the cache entirely when False.
        """
        self.path = Path(path)
        self.ttl = ttl
        self.enabled = enabled
        self._store: dict[str, CacheEntry] | None = None  # lazy load

    def _key(self, screen_id: str, element_label: str) -> str:
        """Build the string cache key.

        Args:
            screen_id: Screen state identifier.
            element_label: Canonical element label.

        Returns:
            Composite key string.
        """
        return f"{screen_id}::{element_label.lower().strip()}"

    def _ensure_loaded(self) -> None:
        """Load the cache file from disk if not already loaded."""
        if self._store is not None:
            return
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    raw: dict[str, Any] = json.load(fh)
                self._store = {k: CacheEntry(**v) for k, v in raw.items()}
                logger.debug("Cache loaded", entries=len(self._store), path=str(self.path))
            except Exception as exc:
                logger.warning("Cache load failed, starting empty", error=str(exc))
                self._store = {}
        else:
            self._store = {}

    def lookup(self, screen_id: str, element_label: str) -> FindResult | None:
        """Return a cached result if one exists and is still fresh.

        Args:
            screen_id: Current screen state identifier.
            element_label: Element being searched.

        Returns:
            :class:`~screenwalker.vision.screen_state.FindResult` or None.
        """
        if not self.enabled:
            return None
        self._ensure_loaded()
        assert self._store is not None
        key = self._key(screen_id, element_label)
        entry = self._store.get(key)
        if entry is None:
            return None
        if not entry.is_fresh(self.ttl):
            del self._store[key]
            logger.debug("Cache entry evicted (stale)", key=key)
            return None
        logger.debug("Cache hit", key=key, method=entry.method)
        return entry.to_find_result()

    def store(self, screen_id: str, result: FindResult) -> None:
        """Persist a successful find result to the cache.

        Args:
            screen_id: Current screen state identifier.
            result: The :class:`FindResult` to cache.
        """
        if not self.enabled:
            return
        self._ensure_loaded()
        assert self._store is not None
        key = self._key(screen_id, result.element)
        entry = CacheEntry(
            screen_id=screen_id,
            element_label=result.element,
            bbox_x=result.bbox.x,
            bbox_y=result.bbox.y,
            bbox_w=result.bbox.w,
            bbox_h=result.bbox.h,
            method=result.method,
            confidence=result.confidence,
            timestamp=time.time(),
        )
        self._store[key] = entry
        logger.debug("Cache store", key=key, method=result.method)
        self.flush()

    def invalidate(self, screen_id: str, element_label: str | None = None) -> None:
        """Remove cache entries for a screen (and optionally a specific element).

        Args:
            screen_id: Screen state to invalidate.
            element_label: If given, invalidate only this element; otherwise
                invalidate all entries for the screen.
        """
        self._ensure_loaded()
        assert self._store is not None
        if element_label is not None:
            key = self._key(screen_id, element_label)
            self._store.pop(key, None)
        else:
            to_remove = [k for k in self._store if k.startswith(f"{screen_id}::")]
            for k in to_remove:
                del self._store[k]

    def flush(self) -> None:
        """Write the current in-memory store to the cache file."""
        if self._store is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serialisable = {k: asdict(v) for k, v in self._store.items()}
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(serialisable, fh, indent=2)
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
