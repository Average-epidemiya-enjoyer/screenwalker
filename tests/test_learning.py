"""Tests for the learning module: StepLogger, ActionCache, PatternLearner."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from screenwalker.core.context import StepRecord
from screenwalker.learning.cache import ActionCache, CacheEntry
from screenwalker.learning.logger import StepLogger, StepResult
from screenwalker.learning.patterns import LearningReport, OcrMismatch, PatternLearner, StepStats, _percentile
from screenwalker.matching.synonyms import SynonymRegistry
from screenwalker.vision.screen_state import BBox, FindResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_image(w: int = 10, h: int = 10) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _make_step_result(**kwargs) -> StepResult:
    defaults = dict(
        step_id="click_ok",
        scenario="test_scenario",
        action="click",
        success=True,
        duration_ms=234.5,
        timestamp=datetime.now(timezone.utc).isoformat(),
        find_target="OK",
        method_used="ocr",
        confidence=0.92,
        bbox=(450, 230, 180, 35),
        found_text="OK",
        error=None,
    )
    defaults.update(kwargs)
    return StepResult(**defaults)


def _make_step_record(**kwargs) -> StepRecord:
    defaults = dict(
        step_id="click_ok",
        action="click",
        success=True,
        elapsed=0.234,
        error=None,
        metadata={},
    )
    defaults.update(kwargs)
    return StepRecord(**defaults)


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# StepLogger
# ---------------------------------------------------------------------------


class TestStepLoggerJsonl:
    def test_creates_jsonl_on_first_log(self, tmp_path: Path) -> None:
        logger = StepLogger(tmp_path / "run1")
        logger.log_result(_make_step_result())
        assert (tmp_path / "run1" / "steps.jsonl").exists()

    def test_jsonl_entry_is_valid_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        logger.log_result(_make_step_result())
        line = (run_dir / "steps.jsonl").read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["step_id"] == "click_ok"
        assert data["success"] is True
        assert data["action"] == "click"

    def test_jsonl_contains_all_required_fields(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        logger.log_result(
            _make_step_result(
                find_target="Save",
                method_used="template",
                confidence=0.88,
                bbox=(10, 20, 30, 40),
                found_text="Save",
                error=None,
            )
        )
        data = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8").strip())
        for field in ("timestamp", "step_id", "scenario", "action", "find_target",
                      "method_used", "confidence", "success", "duration_ms", "error", "bbox"):
            assert field in data, f"Missing field: {field}"
        assert data["bbox"] == [10, 20, 30, 40]

    def test_multiple_steps_appended(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        for i in range(3):
            logger.log_result(_make_step_result(step_id=f"step_{i}"))
        lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        ids = [json.loads(l)["step_id"] for l in lines]
        assert ids == ["step_0", "step_1", "step_2"]

    def test_scenario_start_written_to_jsonl(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        logger.log_scenario_start("my_scenario", variable_count=3)
        data = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8").strip())
        assert data["type"] == "scenario_start"
        assert data["scenario"] == "my_scenario"
        assert data["variables"] == 3

    def test_scenario_end_written_to_jsonl(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        logger.log_scenario_end("s", total_steps=5, failed_steps=1, elapsed=2.5)
        data = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8").strip())
        assert data["type"] == "scenario_end"
        assert data["total_steps"] == 5
        assert data["failed_steps"] == 1
        assert data["success"] is False

    def test_failed_step_written_correctly(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        logger.log_result(_make_step_result(success=False, error="Element not found"))
        data = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8").strip())
        assert data["success"] is False
        assert data["error"] == "Element not found"

    def test_scenario_name_captured_from_start(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        logger.log_scenario_start("create_ticket", variable_count=0)
        logger.log_result(_make_step_result(scenario="create_ticket"))
        lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_log_step_from_step_record(self, tmp_path: Path) -> None:
        """log_step (backward compat) writes JSONL from StepRecord."""
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir)
        logger.log_scenario_start("scenario", 0)
        record = _make_step_record(
            step_id="click_save",
            action="click",
            success=True,
            elapsed=0.123,
            metadata={
                "find_target": "Save",
                "method_used": "ocr",
                "confidence": 0.9,
                "bbox": [10, 20, 80, 30],
                "found_text": "Save",
            },
        )
        logger.log_step(record)
        lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
        # First line is scenario_start, second is the step
        data = json.loads(lines[1])
        assert data["step_id"] == "click_save"
        assert data["find_target"] == "Save"
        assert data["method_used"] == "ocr"
        assert data["bbox"] == [10, 20, 80, 30]
        assert pytest.approx(data["duration_ms"], abs=1) == 123.0


class TestStepLoggerScreenshots:
    def test_screenshot_after_saved_when_enabled(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir, save_screenshots=True)
        logger.log_result(_make_step_result(), screenshot_after=_make_image())
        files = list((run_dir / "screenshots").glob("*.png"))
        assert len(files) == 1
        assert "after" in files[0].name

    def test_screenshot_before_saved_when_enabled(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir, save_screenshots=True)
        logger.log_result(
            _make_step_result(),
            screenshot_before=_make_image(),
            screenshot_after=_make_image(),
        )
        files = sorted((run_dir / "screenshots").glob("*.png"))
        names = [f.name for f in files]
        assert any("before" in n for n in names)
        assert any("after" in n for n in names)

    def test_screenshot_not_saved_when_disabled(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir, save_screenshots=False, save_on_failure=False)
        logger.log_result(_make_step_result(), screenshot_after=_make_image())
        ss_dir = run_dir / "screenshots"
        assert not ss_dir.exists() or len(list(ss_dir.glob("*.png"))) == 0

    def test_screenshot_saved_on_failure_even_if_disabled(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir, save_screenshots=False, save_on_failure=True)
        logger.log_result(
            _make_step_result(success=False, error="oops"),
            screenshot_after=_make_image(),
        )
        files = list((run_dir / "screenshots").glob("*.png"))
        assert len(files) == 1

    def test_step_index_increments_in_filenames(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        logger = StepLogger(run_dir, save_screenshots=True)
        for _ in range(3):
            logger.log_result(_make_step_result(), screenshot_after=_make_image())
        names = sorted(f.name for f in (run_dir / "screenshots").glob("*.png"))
        assert names[0].startswith("step_001_")
        assert names[1].startswith("step_002_")
        assert names[2].startswith("step_003_")


# ---------------------------------------------------------------------------
# ActionCache
# ---------------------------------------------------------------------------


class TestActionCacheGetPut:
    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        assert cache.get("screen_a", "button_ok") is None

    def test_put_and_get_returns_entry(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("screen_a", "button_ok", (10, 20, 80, 30), "ocr", 0.95)
        entry = cache.get("screen_a", "button_ok")
        assert entry is not None
        assert entry.bbox_x == 10
        assert entry.bbox_y == 20
        assert entry.method == "ocr"
        assert entry.confidence == pytest.approx(0.95, abs=0.001)

    def test_hit_count_increments(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)
        cache.get("s", "btn")
        cache.get("s", "btn")
        entry = cache.get("s", "btn")
        assert entry is not None
        assert entry.hit_count == 3

    def test_last_used_updated_on_hit(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)
        entry = cache.get("s", "btn")
        assert entry is not None
        assert entry.last_used != ""

    def test_case_insensitive_key(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("screen", "Button OK", (5, 5, 50, 20), "ocr", 0.8)
        assert cache.get("screen", "button ok") is not None
        assert cache.get("screen", "BUTTON OK") is not None

    def test_disabled_cache_returns_none(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json", enabled=False)
        cache.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)
        assert cache.get("s", "btn") is None

    def test_lookup_returns_find_result(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("screen_a", "ok_button", (10, 20, 80, 30), "template", 0.88)
        result = cache.lookup("screen_a", "ok_button")
        assert result is not None
        assert isinstance(result, FindResult)
        assert "cache" in result.method

    def test_store_and_lookup(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        find_result = FindResult(
            element="save_btn",
            confidence=0.91,
            bbox=BBox(x=100, y=200, w=60, h=25),
            method="template",
        )
        cache.store("main_screen", find_result)
        r = cache.lookup("main_screen", "save_btn")
        assert r is not None
        assert r.bbox.x == 100


class TestActionCacheTtl:
    def test_ttl_expired_entry_evicted(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json", ttl=0.001)  # 1 ms
        cache.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)
        time.sleep(0.01)
        assert cache.get("s", "btn") is None

    def test_fresh_entry_not_evicted(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json", ttl=3600)
        cache.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)
        assert cache.get("s", "btn") is not None

    def test_evict_stale_removes_expired(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json", ttl=0.001)
        cache.put("s", "btn1", (0, 0, 10, 10), "ocr", 0.9)
        cache.put("s", "btn2", (0, 0, 10, 10), "ocr", 0.9)
        time.sleep(0.01)
        count = cache.evict_stale()
        assert count == 2

    def test_evict_stale_keeps_fresh(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json", ttl=3600)
        cache.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)
        count = cache.evict_stale()
        assert count == 0
        assert cache.get("s", "btn") is not None


class TestActionCacheInvalidate:
    def test_invalidate_specific_element(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("scr", "btn1", (0, 0, 10, 10), "ocr", 0.9)
        cache.put("scr", "btn2", (0, 0, 10, 10), "ocr", 0.9)
        cache.invalidate("scr", "btn1")
        assert cache.get("scr", "btn1") is None
        assert cache.get("scr", "btn2") is not None

    def test_invalidate_all_for_screen(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("scr", "btn1", (0, 0, 10, 10), "ocr", 0.9)
        cache.put("scr", "btn2", (0, 0, 10, 10), "ocr", 0.9)
        cache.put("other", "btn3", (0, 0, 10, 10), "ocr", 0.9)
        cache.invalidate("scr")
        assert cache.get("scr", "btn1") is None
        assert cache.get("scr", "btn2") is None
        assert cache.get("other", "btn3") is not None


class TestActionCachePersistence:
    def test_flush_creates_json_file(self, tmp_path: Path) -> None:
        cache = ActionCache(tmp_path / "cache.json")
        cache.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)
        assert (tmp_path / "cache.json").exists()

    def test_reload_persists_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache1 = ActionCache(path)
        cache1.put("s", "btn", (5, 5, 50, 20), "template", 0.88)

        cache2 = ActionCache(path)
        entry = cache2.get("s", "btn")
        assert entry is not None
        assert entry.bbox_x == 5

    def test_resolution_change_invalidates_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"

        # Save cache with resolution [1920, 1080]
        with patch.object(ActionCache, "_get_resolution", return_value=[1920, 1080]):
            cache1 = ActionCache(path)
            cache1.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)

        # Load with different resolution [1280, 720]
        with patch.object(ActionCache, "_get_resolution", return_value=[1280, 720]):
            cache2 = ActionCache(path)
            assert cache2.get("s", "btn") is None

    def test_same_resolution_preserves_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        with patch.object(ActionCache, "_get_resolution", return_value=[1920, 1080]):
            cache1 = ActionCache(path)
            cache1.put("s", "btn", (0, 0, 10, 10), "ocr", 0.9)

        with patch.object(ActionCache, "_get_resolution", return_value=[1920, 1080]):
            cache2 = ActionCache(path)
            assert cache2.get("s", "btn") is not None

    def test_corrupt_json_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("not valid json", encoding="utf-8")
        cache = ActionCache(path)
        assert cache.get("s", "btn") is None  # no crash


# ---------------------------------------------------------------------------
# PatternLearner — online learning
# ---------------------------------------------------------------------------


class TestPatternLearnerObserve:
    def test_observe_increments_count(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("ok", ["ok", "confirm"])
        learner = PatternLearner(reg, min_observations=3)
        learner.observe("ok", "new_alias")
        assert learner._observations["ok"]["new_alias"] == 1

    def test_observe_skips_known_synonym(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("ok", ["ok", "confirm"])
        learner = PatternLearner(reg, min_observations=3)
        learner.observe("ok", "confirm")  # already known
        assert "confirm" not in learner._observations.get("ok", {})

    def test_observe_promotes_at_threshold(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("ok", ["ok"])
        learner = PatternLearner(reg, min_observations=2)
        learner.observe("ok", "jawohl")
        learner.observe("ok", "jawohl")  # threshold reached
        assert "jawohl" in reg.expand("ok")

    def test_pending_proposals_below_threshold(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("ok", ["ok"])
        learner = PatternLearner(reg, min_observations=5)
        learner.observe("ok", "sure")
        learner.observe("ok", "sure")
        proposals = learner.pending_proposals()
        assert "ok" in proposals
        assert any(t == "sure" for t, _ in proposals["ok"])


class TestPatternLearnerEvidence:
    def test_save_and_load_evidence(self, tmp_path: Path) -> None:
        path = tmp_path / "evidence.yaml"
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("ok", ["ok"])
        learner = PatternLearner(reg, min_observations=10, evidence_path=path)
        learner.observe("ok", "jawohl")
        learner.observe("ok", "jawohl")
        learner.save_evidence()
        assert path.exists()

        reg2 = SynonymRegistry(load_defaults=False)
        reg2.add_group("ok", ["ok"])
        learner2 = PatternLearner(reg2, min_observations=10, evidence_path=path)
        learner2.load_evidence()
        assert learner2._observations["ok"]["jawohl"] == 2

    def test_load_evidence_no_file_is_noop(self, tmp_path: Path) -> None:
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg, evidence_path=tmp_path / "missing.yaml")
        learner.load_evidence()  # should not raise


# ---------------------------------------------------------------------------
# PatternLearner — offline log analysis
# ---------------------------------------------------------------------------


def _fake_jsonl(step_id: str, success: bool, duration_ms: float,
                method: str = "ocr", find_target: str | None = None,
                found_text: str | None = None, error: str | None = None) -> dict:
    return {
        "timestamp": "2024-01-15T14:30:00+00:00",
        "step_id": step_id,
        "scenario": "test",
        "action": "click",
        "find_target": find_target,
        "method_used": method,
        "confidence": 0.9 if success else None,
        "success": success,
        "duration_ms": duration_ms,
        "error": error,
        "bbox": [10, 20, 80, 30] if success else None,
        "found_text": found_text,
    }


class TestAnalyzeLogs:
    def test_empty_dir_returns_empty_report(self, tmp_path: Path) -> None:
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        assert report.run_count == 0
        assert report.total_steps == 0
        assert report.step_stats == {}

    def test_counts_runs_and_steps(self, tmp_path: Path) -> None:
        for run in ("run1", "run2"):
            _write_jsonl(
                tmp_path / run / "steps.jsonl",
                [_fake_jsonl("click_ok", True, 200.0)],
            )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        assert report.run_count == 2
        assert report.total_steps == 2

    def test_identifies_failed_steps(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [
                _fake_jsonl("click_ok", True, 100.0),
                _fake_jsonl("click_save", False, 3000.0, error="Timeout"),
                _fake_jsonl("click_save", False, 3100.0, error="Timeout"),
            ],
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        assert "click_save" in report.failed_step_ids
        assert "click_ok" not in report.failed_step_ids
        assert report.step_stats["click_save"].failure_count == 2

    def test_avg_duration_computed(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [
                _fake_jsonl("step_a", True, 100.0),
                _fake_jsonl("step_a", True, 200.0),
                _fake_jsonl("step_a", True, 300.0),
            ],
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        durations = report.step_stats["step_a"].durations_ms
        assert len(durations) == 3
        assert sum(durations) / len(durations) == pytest.approx(200.0, abs=0.1)

    def test_method_stats_aggregated(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [
                _fake_jsonl("s1", True, 100.0, method="ocr"),
                _fake_jsonl("s2", True, 100.0, method="ocr"),
                _fake_jsonl("s3", False, 100.0, method="template"),
            ],
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        assert report.method_stats["ocr"]["success"] == 2
        assert report.method_stats["template"]["failure"] == 1

    def test_ocr_mismatch_detected(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [
                _fake_jsonl(
                    "click_close", True, 150.0,
                    find_target="Закрыть", found_text="3акрыть",
                ),
                _fake_jsonl(
                    "click_close", True, 150.0,
                    find_target="Закрыть", found_text="3акрыть",
                ),
            ],
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        assert len(report.ocr_mismatches) == 1
        m = report.ocr_mismatches[0]
        assert m.find_target == "Закрыть"
        assert m.found_text == "3акрыть"
        assert m.count == 2

    def test_no_mismatch_when_texts_equal(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [_fake_jsonl("s1", True, 100.0, find_target="OK", found_text="OK")],
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        assert report.ocr_mismatches == []

    def test_suggests_timeout_for_step_with_enough_data(self, tmp_path: Path) -> None:
        entries = [_fake_jsonl("step_a", True, float(d)) for d in [100, 110, 120, 130, 140, 2000]]
        _write_jsonl(tmp_path / "run1" / "steps.jsonl", entries)
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        # p95 of [100,110,120,130,140,2000] = 2000; suggest = 2400
        assert "step_a" in report.suggested_timeouts
        assert report.suggested_timeouts["step_a"] > 0

    def test_no_timeout_suggestion_below_min_data_points(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [_fake_jsonl("step_a", True, 100.0), _fake_jsonl("step_a", True, 200.0)],
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        # only 2 data points < 3 minimum
        assert "step_a" not in report.suggested_timeouts

    def test_skips_scenario_boundary_lines(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [
                {"type": "scenario_start", "scenario": "s", "variables": 0, "timestamp": "ts"},
                _fake_jsonl("step_a", True, 100.0),
                {"type": "scenario_end", "total_steps": 1, "failed_steps": 0, "timestamp": "ts"},
            ],
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        assert report.total_steps == 1

    def test_handles_malformed_jsonl_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "run1" / "steps.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            '{"step_id":"s1","success":true,"duration_ms":100.0,"action":"click"}\n'
            "not valid json\n"
            '{"step_id":"s2","success":false,"duration_ms":200.0,"action":"click"}\n',
            encoding="utf-8",
        )
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        report = learner.analyze_logs(tmp_path)
        # malformed line skipped, 2 valid steps counted
        assert report.total_steps == 2


class TestSuggestSynonyms:
    def test_suggest_synonyms_from_observations(self, tmp_path: Path) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("ok", ["ok"])
        # min_observations=5 keeps "jawohl" pending (not promoted yet)
        learner = PatternLearner(reg, min_observations=5)
        learner.observe("ok", "jawohl")
        suggestions = learner.suggest_synonyms()
        assert "ok" in suggestions
        assert "jawohl" in suggestions["ok"]

    def test_no_suggestion_for_known_synonym(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("ok", ["ok", "confirm"])
        learner = PatternLearner(reg, min_observations=1)
        learner.observe("ok", "confirm")  # already known
        suggestions = learner.suggest_synonyms()
        ok_cands = suggestions.get("ok", [])
        assert "confirm" not in ok_cands

    def test_suggest_synonyms_from_analyze_logs(self, tmp_path: Path) -> None:
        _write_jsonl(
            tmp_path / "run1" / "steps.jsonl",
            [
                _fake_jsonl("close_btn", True, 100.0,
                            find_target="Закрыть", found_text="3акрыть"),
            ],
        )
        reg = SynonymRegistry(load_defaults=False)
        # min_observations=5 keeps "3акрыть" pending (mismatch count is 1)
        learner = PatternLearner(reg, min_observations=5)
        learner.analyze_logs(tmp_path)
        suggestions = learner.suggest_synonyms()
        # "закрыть" (normalized from "Закрыть") should have "3акрыть" as candidate
        assert any(
            "3акрыть" in cands
            for cands in suggestions.values()
        )


class TestOptimizeTimeouts:
    def test_returns_empty_before_analyze(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        assert learner.optimize_timeouts() == {}

    def test_returns_suggestions_after_analyze(self, tmp_path: Path) -> None:
        entries = [_fake_jsonl("step_a", True, float(d)) for d in [100, 200, 300, 400, 500]]
        _write_jsonl(tmp_path / "run1" / "steps.jsonl", entries)
        reg = SynonymRegistry(load_defaults=False)
        learner = PatternLearner(reg)
        learner.analyze_logs(tmp_path)
        timeouts = learner.optimize_timeouts()
        assert "step_a" in timeouts
        assert timeouts["step_a"] > 0


class TestLearningReportFormat:
    def test_format_text_no_failures(self) -> None:
        report = LearningReport(
            run_count=2,
            total_steps=10,
            step_stats={"s1": StepStats(count=10, success_count=10)},
            ocr_mismatches=[],
            method_stats={"ocr": {"success": 10, "failure": 0}},
            suggested_timeouts={},
        )
        text = report.format_text()
        assert "No failed steps" in text
        assert "Runs: 2" in text

    def test_format_text_shows_failures(self) -> None:
        ss = StepStats(count=5, success_count=3, failure_count=2, errors=["Timeout"])
        report = LearningReport(
            run_count=1,
            total_steps=5,
            step_stats={"click_save": ss},
            ocr_mismatches=[],
            method_stats={},
            suggested_timeouts={},
        )
        text = report.format_text()
        assert "click_save" in text
        assert "Timeout" in text

    def test_format_text_shows_ocr_mismatches(self) -> None:
        report = LearningReport(
            run_count=1,
            total_steps=3,
            step_stats={},
            ocr_mismatches=[OcrMismatch("Закрыть", "3акрыть", "step1", 5)],
            method_stats={},
            suggested_timeouts={},
        )
        text = report.format_text()
        assert "Закрыть" in text
        assert "3акрыть" in text

    def test_failed_step_ids_sorted(self) -> None:
        report = LearningReport(
            run_count=1, total_steps=10,
            step_stats={
                "step_c": StepStats(failure_count=1),
                "step_a": StepStats(failure_count=1),
                "step_b": StepStats(failure_count=1),
            },
            ocr_mismatches=[], method_stats={}, suggested_timeouts={},
        )
        assert report.failed_step_ids == ["step_a", "step_b", "step_c"]


class TestPercentileHelper:
    def test_p50_median(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(2.0, abs=0.1)

    def test_p100_max(self) -> None:
        vals = [10.0, 20.0, 30.0]
        assert _percentile(vals, 100) == pytest.approx(30.0, abs=0.1)

    def test_empty_returns_zero(self) -> None:
        assert _percentile([], 95) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([42.0], 95) == pytest.approx(42.0, abs=0.1)
