"""Pattern learner — analyses execution logs and suggests improvements.

Two complementary capabilities are provided:

**Online learning** (:meth:`PatternLearner.observe`)
    Called by the engine after each successful OCR match.  When the same
    observed text is seen :attr:`~PatternLearner.min_observations` times for
    the same canonical label, it is automatically promoted to the synonym
    registry.

**Offline log analysis** (:meth:`PatternLearner.analyze_logs`)
    Reads all ``steps.jsonl`` files under a log directory and produces a
    :class:`LearningReport` containing:

    * Which steps fail most often (and why).
    * OCR mismatches — candidates for new synonym entries.
    * Average execution time per step (for timeout optimisation).
    * Method-level success/failure statistics.
    * Suggested timeout values (p95 × 1.2).
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
# Report data types
# ---------------------------------------------------------------------------


@dataclass
class StepStats:
    """Aggregated execution statistics for a single step ID.

    Attributes:
        count: Total number of executions observed.
        success_count: Successful executions.
        failure_count: Failed executions.
        durations_ms: All observed execution durations in milliseconds.
        errors: Error messages from failed executions (deduplicated).
        methods: Counts per vision method used.
    """

    count: int = 0
    success_count: int = 0
    failure_count: int = 0
    durations_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    methods: dict[str, int] = field(default_factory=dict)


@dataclass
class OcrMismatch:
    """Records an OCR reading that differed from the expected label.

    Attributes:
        find_target: The text that was searched for.
        found_text: The actual text OCR returned.
        step_id: Step where this mismatch was observed.
        count: Number of times this exact pair was observed.
    """

    find_target: str
    found_text: str
    step_id: str
    count: int = 1


@dataclass
class LearningReport:
    """Summary produced by :meth:`PatternLearner.analyze_logs`.

    Attributes:
        run_count: Number of distinct scenario runs analysed.
        total_steps: Total step executions across all runs.
        step_stats: Per-step aggregated statistics.
        ocr_mismatches: OCR readings that differed from the search target.
        method_stats: Per-method success/failure counts.
        suggested_timeouts: Recommended timeout values (ms) per step.
    """

    run_count: int
    total_steps: int
    step_stats: dict[str, StepStats]
    ocr_mismatches: list[OcrMismatch]
    method_stats: dict[str, dict[str, int]]
    suggested_timeouts: dict[str, int]

    @property
    def failed_step_ids(self) -> list[str]:
        """Sorted list of step IDs that have at least one failure."""
        return sorted(
            sid for sid, s in self.step_stats.items() if s.failure_count > 0
        )

    def format_text(self) -> str:
        """Render the report as a human-readable text summary.

        Returns:
            Multi-line string ready for ``click.echo()``.
        """
        lines: list[str] = []
        lines.append("=== Learning Report ===")
        lines.append(f"Runs: {self.run_count}  |  Total steps: {self.total_steps}")

        # Failed steps
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

        # OCR mismatches
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

        # Method stats
        if self.method_stats:
            lines.append("\n--- Method Effectiveness ---")
            for method, stats in sorted(self.method_stats.items()):
                ok = stats.get("success", 0)
                total = ok + stats.get("failure", 0)
                pct = ok / total * 100 if total else 0
                lines.append(f"  {method}: {ok}/{total} ({pct:.0f}% success)")

        # Timeout suggestions
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
    """Observes successful matches and proposes new synonym expansions.

    Also performs offline log analysis via :meth:`analyze_logs`.

    Attributes:
        registry: :class:`~screenwalker.matching.synonyms.SynonymRegistry`
            updated with learned patterns.
        min_observations: Times a pattern must be seen before automatic
            promotion.
        evidence_path: Optional path to persist observation counts.
    """

    def __init__(
        self,
        registry: SynonymRegistry,
        min_observations: int = 3,
        evidence_path: Path | str | None = None,
    ) -> None:
        """Initialise PatternLearner.

        Args:
            registry: Synonym registry to expand with learned patterns.
            min_observations: Evidence threshold for automatic promotion.
            evidence_path: YAML file to persist observation counts across runs.
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
    # Online learning
    # ------------------------------------------------------------------

    def observe(self, canonical_label: str, observed_text: str) -> None:
        """Record that *observed_text* was matched to *canonical_label*.

        If the observation count reaches :attr:`min_observations`, the text
        is automatically added to the synonym registry.

        Args:
            canonical_label: The element label the match was resolved to.
            observed_text: The raw OCR text that was matched.
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
        """Add *observed_text* to the synonym registry.

        Args:
            canonical_label: Canonical element label.
            observed_text: Text to add as a synonym.
        """
        logger.info(
            "promoting_synonym",
            label=canonical_label,
            synonym=observed_text,
        )
        self.registry.add_group(canonical_label, [observed_text])

    def pending_proposals(self) -> dict[str, list[tuple[str, int]]]:
        """Return observed patterns that have not yet reached the threshold.

        Returns:
            Mapping of ``canonical_label`` → ``[(text, count)]`` sorted by
            count descending, for patterns still below the threshold.
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
    # Offline log analysis
    # ------------------------------------------------------------------

    def analyze_logs(self, log_dir: Path | str) -> LearningReport:
        """Analyse all ``steps.jsonl`` files under *log_dir*.

        Searches for files matching ``*/steps.jsonl`` (one level deep) and
        ``steps.jsonl`` at the root of *log_dir*.

        Args:
            log_dir: Root directory that contains per-run sub-directories.

        Returns:
            :class:`LearningReport` aggregating all observed executions.
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

                # Skip scenario boundary events
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

                # Accumulate step stats
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

                # Method-level stats
                if method not in method_stats:
                    method_stats[method] = {"success": 0, "failure": 0}
                method_stats[method]["success" if success else "failure"] += 1

                # OCR mismatch detection
                if (
                    find_target
                    and found_text
                    and found_text.lower().strip() != find_target.lower().strip()
                ):
                    mk = (find_target, found_text)
                    mismatch_counts[mk] = mismatch_counts.get(mk, 0) + 1
                    mismatch_step[mk] = step_id

        # Build mismatch list sorted by frequency
        ocr_mismatches: list[OcrMismatch] = [
            OcrMismatch(
                find_target=key[0],
                found_text=key[1],
                step_id=mismatch_step[key],
                count=cnt,
            )
            for key, cnt in sorted(mismatch_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        # Feed mismatches into online learner
        for m in ocr_mismatches:
            for _ in range(m.count):
                self.observe(m.find_target, m.found_text)

        # Suggested timeouts: p95 × 1.2 for steps with ≥ 3 data points
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
    # Derived suggestions
    # ------------------------------------------------------------------

    def suggest_synonyms(self) -> dict[str, list[str]]:
        """Return synonym candidates derived from observations.

        Each key is a find_target / canonical label; values are OCR readings
        that were matched but are not yet in the synonym registry.

        Returns:
            Mapping of label → list of candidate synonym strings.
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
        """Return timeout suggestions from the most recent :meth:`analyze_logs`.

        Returns:
            Mapping of step_id → suggested timeout in milliseconds
            (p95 of observed durations × 1.2).  Empty if no logs have been
            analysed.
        """
        return dict(self._last_report.suggested_timeouts)

    # ------------------------------------------------------------------
    # Evidence persistence
    # ------------------------------------------------------------------

    def save_evidence(self) -> None:
        """Persist observation counts to :attr:`evidence_path`.

        No-op if :attr:`evidence_path` is ``None``.
        """
        if not self.evidence_path:
            return
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {label: dict(texts) for label, texts in self._observations.items()}
        with self.evidence_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True)
        logger.debug("Evidence saved", path=str(self.evidence_path))

    def load_evidence(self) -> None:
        """Load persisted observation counts from :attr:`evidence_path`.

        No-op if the file does not exist.
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
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: int) -> float:
    """Return the *pct*-th percentile of *values* (nearest-rank method).

    Args:
        values: Non-empty list of numeric values.
        pct: Percentile in ``[0, 100]``.

    Returns:
        Percentile value.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * pct / 100) - 1)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]
