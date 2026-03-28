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
        self._store: dict[str, dict[str, Any]] | None = None  # lazy load

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
            # TODO: read and deserialise JSON
            raise NotImplementedError("TODO: implement cache file loading")
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
        # TODO: load, key lookup, freshness check, return or evict
        raise NotImplementedError("TODO: implement ActionCache.lookup")

    def store(self, screen_id: str, result: FindResult) -> None:
        """Persist a successful find result to the cache.

        Args:
            screen_id: Current screen state identifier.
            result: The :class:`FindResult` to cache.
        """
        if not self.enabled:
            return
        # TODO: build CacheEntry, update store, flush to disk
        raise NotImplementedError("TODO: implement ActionCache.store")

    def invalidate(self, screen_id: str, element_label: str | None = None) -> None:
        """Remove cache entries for a screen (and optionally a specific element).

        Args:
            screen_id: Screen state to invalidate.
            element_label: If given, invalidate only this element; otherwise
                invalidate all entries for the screen.
        """
        # TODO: filter and remove matching keys
        raise NotImplementedError("TODO: implement ActionCache.invalidate")

    def flush(self) -> None:
        """Write the current in-memory store to the cache file."""
        if self._store is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: serialise and write JSON
        raise NotImplementedError("TODO: implement ActionCache.flush")

    def evict_stale(self) -> int:
        """Remove all entries older than :attr:`ttl`.

        Returns:
            Number of entries evicted.
        """
        # TODO: iterate store, remove entries where not entry.is_fresh(self.ttl)
        raise NotImplementedError("TODO: implement ActionCache.evict_stale")
