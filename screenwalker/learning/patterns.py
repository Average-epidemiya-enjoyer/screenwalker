"""Pattern learner — expands synonym dictionaries from runtime observations.

When ``config.learning.pattern_learning`` is enabled, the engine calls
:meth:`PatternLearner.observe` after each successful OCR match.  Over time
the learner accumulates evidence and proposes new synonyms for elements that
are consistently matched under different OCR readings.

This is an **experimental** module and is disabled by default.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog
import yaml

from screenwalker.matching.synonyms import SynonymRegistry

logger = structlog.get_logger(__name__)


class PatternLearner:
    """Observes successful matches and proposes new synonym expansions.

    Attributes:
        registry: The :class:`~screenwalker.matching.synonyms.SynonymRegistry`
            that will be updated with learned patterns.
        min_observations: Minimum times a pattern must be seen before it is
            automatically promoted to the synonym dictionary.
        evidence_path: Optional path to persist observation counts across runs.
    """

    def __init__(
        self,
        registry: SynonymRegistry,
        min_observations: int = 3,
        evidence_path: Path | str | None = None,
    ) -> None:
        """Initialize PatternLearner.

        Args:
            registry: Synonym registry to expand with learned patterns.
            min_observations: Evidence threshold for automatic promotion.
            evidence_path: YAML file to persist observation counts.
        """
        self.registry = registry
        self.min_observations = min_observations
        self.evidence_path = Path(evidence_path) if evidence_path else None
        # canonical_label → {observed_text → count}
        self._observations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, canonical_label: str, observed_text: str) -> None:
        """Record that *observed_text* was matched to *canonical_label*.

        If the observation count for this pair reaches :attr:`min_observations`,
        the text is automatically added to the synonym registry.

        Args:
            canonical_label: The element label the match was resolved to.
            observed_text: The raw OCR text that was matched.
        """
        norm_text = observed_text.lower().strip()
        norm_label = canonical_label.lower().strip()

        # Skip if already a known synonym
        if norm_text in self.registry.expand(norm_label):
            return

        self._observations[norm_label][norm_text] += 1
        count = self._observations[norm_label][norm_text]

        logger.debug(
            "Pattern observation",
            label=norm_label,
            text=norm_text,
            count=count,
        )

        if count >= self.min_observations:
            self._promote(norm_label, norm_text)

    def _promote(self, canonical_label: str, observed_text: str) -> None:
        """Add *observed_text* to the synonym registry.

        Args:
            canonical_label: Canonical element label.
            observed_text: Text to add as a synonym.
        """
        logger.info(
            "Promoting new synonym",
            label=canonical_label,
            synonym=observed_text,
        )
        self.registry.add_group(canonical_label, [observed_text])

    def pending_proposals(self) -> dict[str, list[tuple[str, int]]]:
        """Return observed patterns that have not yet reached the threshold.

        Returns:
            Mapping of canonical_label → list of (text, count) tuples,
            sorted by count descending, for patterns below threshold.
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

    def save_evidence(self) -> None:
        """Persist observation counts to :attr:`evidence_path`.

        No-op if :attr:`evidence_path` is None.
        """
        if not self.evidence_path:
            return
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {label: dict(texts) for label, texts in self._observations.items()}
        with self.evidence_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True)

    def load_evidence(self) -> None:
        """Load persisted observation counts from :attr:`evidence_path`.

        No-op if the file does not exist.
        """
        if not self.evidence_path or not self.evidence_path.exists():
            return
        # TODO: load YAML, merge into self._observations
        raise NotImplementedError("TODO: implement PatternLearner.load_evidence")
