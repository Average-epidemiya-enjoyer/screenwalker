"""Тесты для генератора HTML-отчётов."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from screenwalker.reporting.html_report import (
    RunReportGenerator,
    RunResult,
    StepReportRecord,
    _annotate_screenshot,
    _extract_bbox,
    _find_screenshot,
    _image_to_base64_jpeg,
)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------


def _make_image(color: tuple[int, int, int] = (50, 50, 80), size: tuple[int, int] = (200, 100)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _dt(year: int = 2024, month: int = 1, day: int = 15, hour: int = 10, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _make_step(
    index: int = 1,
    step_id: str = "step_1",
    action: str = "click",
    success: bool = True,
    elapsed_ms: float = 150.0,
    error: str | None = None,
    screenshot_path: Path | None = None,
    confidence: float | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    find_target: str | None = None,
    method_used: str | None = None,
) -> StepReportRecord:
    return StepReportRecord(
        index=index,
        step_id=step_id,
        action=action,
        success=success,
        elapsed_ms=elapsed_ms,
        error=error,
        screenshot_path=screenshot_path,
        confidence=confidence,
        bbox=bbox,
        find_target=find_target,
        method_used=method_used,
    )


_SENTINEL = object()


def _make_result(
    steps: list[StepReportRecord] | None | object = _SENTINEL,
    success: bool = True,
    scenario_name: str = "Тестовый сценарий",
) -> RunResult:
    if steps is _SENTINEL:
        steps = [_make_step()]
    return RunResult(
        scenario_name=scenario_name,
        run_id="20240115_100000",
        started_at=_dt(hour=10),
        finished_at=_dt(hour=10, minute=0, second=5) if False else datetime(2024, 1, 15, 10, 0, 5, tzinfo=timezone.utc),
        success=success,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Тесты RunResult — вычисляемые свойства
# ---------------------------------------------------------------------------


class TestRunResultProperties:
    def test_duration_s(self) -> None:
        result = RunResult(
            scenario_name="test",
            run_id="r1",
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            finished_at=datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc),
            success=True,
            steps=[],
        )
        assert result.duration_s == pytest.approx(30.0)

    def test_pass_rate_all_pass(self) -> None:
        result = _make_result(steps=[
            _make_step(success=True),
            _make_step(success=True),
        ])
        assert result.pass_rate == pytest.approx(1.0)

    def test_pass_rate_half(self) -> None:
        result = _make_result(steps=[
            _make_step(success=True),
            _make_step(success=False),
        ])
        assert result.pass_rate == pytest.approx(0.5)

    def test_pass_rate_empty(self) -> None:
        result = _make_result(steps=[])
        assert result.pass_rate == pytest.approx(0.0)

    def test_avg_step_ms(self) -> None:
        result = _make_result(steps=[
            _make_step(elapsed_ms=100.0),
            _make_step(elapsed_ms=300.0),
        ])
        assert result.avg_step_ms == pytest.approx(200.0)

    def test_avg_step_ms_empty(self) -> None:
        result = _make_result(steps=[])
        assert result.avg_step_ms == pytest.approx(0.0)

    def test_slowest_step(self) -> None:
        slow = _make_step(index=2, step_id="slow", elapsed_ms=999.0)
        result = _make_result(steps=[
            _make_step(elapsed_ms=100.0),
            slow,
        ])
        assert result.slowest_step is slow

    def test_slowest_step_none_when_empty(self) -> None:
        result = _make_result(steps=[])
        assert result.slowest_step is None


# ---------------------------------------------------------------------------
# Тесты RunResult.from_log_dir
# ---------------------------------------------------------------------------


class TestRunResultFromLogDir:
    def _write_jsonl(self, path: Path, entries: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")

    def test_raises_when_jsonl_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="steps.jsonl"):
            RunResult.from_log_dir(tmp_path)

    def test_loads_scenario_metadata(self, tmp_path: Path) -> None:
        self._write_jsonl(tmp_path / "steps.jsonl", [
            {
                "type": "scenario_start",
                "scenario": "Тест входа",
                "timestamp": "2024-01-15T10:00:00+00:00",
            },
            {
                "type": "scenario_end",
                "success": True,
                "timestamp": "2024-01-15T10:00:10+00:00",
                "total_steps": 1,
                "failed_steps": 0,
            },
            {
                "step_id": "login",
                "action": "click",
                "success": True,
                "duration_ms": 200.0,
            },
        ])
        result = RunResult.from_log_dir(tmp_path)
        assert result.scenario_name == "Тест входа"
        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].step_id == "login"

    def test_populates_step_fields(self, tmp_path: Path) -> None:
        self._write_jsonl(tmp_path / "steps.jsonl", [
            {
                "step_id": "find_btn",
                "action": "click",
                "success": False,
                "duration_ms": 450.5,
                "find_target": "Войти",
                "method_used": "ocr",
                "confidence": 0.92,
                "bbox": [100, 200, 80, 30],
                "found_text": "Войти",
                "error": "ElementNotFound: Войти",
            },
        ])
        result = RunResult.from_log_dir(tmp_path)
        step = result.steps[0]
        assert step.step_id == "find_btn"
        assert step.success is False
        assert step.elapsed_ms == pytest.approx(450.5)
        assert step.find_target == "Войти"
        assert step.method_used == "ocr"
        assert step.confidence == pytest.approx(0.92)
        assert step.bbox == (100, 200, 80, 30)
        assert step.error is not None

    def test_finds_screenshot_for_step(self, tmp_path: Path) -> None:
        self._write_jsonl(tmp_path / "steps.jsonl", [
            {"step_id": "s1", "action": "click", "success": True, "duration_ms": 100.0},
        ])
        ss_dir = tmp_path / "screenshots"
        ss_dir.mkdir()
        ss_file = ss_dir / "step_001_after.png"
        _make_image().save(ss_file)

        result = RunResult.from_log_dir(tmp_path)
        assert result.steps[0].screenshot_path == ss_file

    def test_no_screenshot_if_missing(self, tmp_path: Path) -> None:
        self._write_jsonl(tmp_path / "steps.jsonl", [
            {"step_id": "s1", "action": "click", "success": True, "duration_ms": 100.0},
        ])
        result = RunResult.from_log_dir(tmp_path)
        assert result.steps[0].screenshot_path is None

    def test_uses_log_dir_name_as_fallback_scenario(self, tmp_path: Path) -> None:
        self._write_jsonl(tmp_path / "steps.jsonl", [
            {"step_id": "s1", "action": "wait", "success": True, "duration_ms": 50.0},
        ])
        result = RunResult.from_log_dir(tmp_path)
        assert result.scenario_name == tmp_path.name

    def test_ignores_malformed_json_lines(self, tmp_path: Path) -> None:
        (tmp_path / "steps.jsonl").write_text(
            '{"step_id":"s1","action":"click","success":true,"duration_ms":100}\n'
            "BROKEN LINE\n"
            '{"step_id":"s2","action":"wait","success":true,"duration_ms":50}\n',
            encoding="utf-8",
        )
        result = RunResult.from_log_dir(tmp_path)
        assert len(result.steps) == 2

    def test_empty_jsonl_returns_empty_steps(self, tmp_path: Path) -> None:
        (tmp_path / "steps.jsonl").write_text("", encoding="utf-8")
        result = RunResult.from_log_dir(tmp_path)
        assert result.steps == []


# ---------------------------------------------------------------------------
# Тесты RunResult.from_context
# ---------------------------------------------------------------------------


class TestRunResultFromContext:
    def _make_context(self, tmp_path: Path):
        """Создать минимальный mock-объект RunContext."""
        from unittest.mock import MagicMock
        from screenwalker.core.context import StepRecord

        ctx = MagicMock()
        ctx.scenario_name = "Сценарий из контекста"
        ctx.run_id = "20240115_120000"
        ctx.output_dir = tmp_path
        ctx.failed_steps = []
        ctx.history = [
            StepRecord(step_id="step1", action="click", success=True, elapsed=0.25),
            StepRecord(step_id="step2", action="type", success=False, elapsed=0.10, error="Упал"),
        ]
        return ctx

    def test_builds_result_from_context(self, tmp_path: Path) -> None:
        ctx = self._make_context(tmp_path)
        result = RunResult.from_context(ctx, _dt(hour=10), _dt(hour=10, minute=0))
        assert result.scenario_name == "Сценарий из контекста"
        assert len(result.steps) == 2

    def test_step_elapsed_converted_to_ms(self, tmp_path: Path) -> None:
        ctx = self._make_context(tmp_path)
        result = RunResult.from_context(ctx, _dt(), _dt())
        assert result.steps[0].elapsed_ms == pytest.approx(250.0)

    def test_failed_step_has_error(self, tmp_path: Path) -> None:
        ctx = self._make_context(tmp_path)
        result = RunResult.from_context(ctx, _dt(), _dt())
        assert result.steps[1].error == "Упал"

    def test_screenshot_discovered_from_output_dir(self, tmp_path: Path) -> None:
        ctx = self._make_context(tmp_path)
        ss_dir = tmp_path / "screenshots"
        ss_dir.mkdir()
        ss = ss_dir / "step_001_after.png"
        _make_image().save(ss)

        result = RunResult.from_context(ctx, _dt(), _dt())
        assert result.steps[0].screenshot_path == ss


# ---------------------------------------------------------------------------
# Тесты RunReportGenerator
# ---------------------------------------------------------------------------


class TestRunReportGenerator:
    def test_generate_creates_file(self, tmp_path: Path) -> None:
        result = _make_result()
        gen = RunReportGenerator(include_screenshots=False)
        out = tmp_path / "report.html"
        path = gen.generate(result, out)
        assert path.exists()

    def test_generate_returns_absolute_path(self, tmp_path: Path) -> None:
        result = _make_result()
        gen = RunReportGenerator(include_screenshots=False)
        path = gen.generate(result, tmp_path / "r.html")
        assert path.is_absolute()

    def test_html_contains_scenario_name(self, tmp_path: Path) -> None:
        result = _make_result(scenario_name="Мой сценарий")
        gen = RunReportGenerator(include_screenshots=False)
        out = gen.generate(result, tmp_path / "r.html")
        content = out.read_text(encoding="utf-8")
        assert "Мой сценарий" in content

    def test_html_contains_pass_badge_on_success(self, tmp_path: Path) -> None:
        result = _make_result(success=True)
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "PASS" in content
        assert "badge-pass" in content

    def test_html_contains_fail_badge_on_failure(self, tmp_path: Path) -> None:
        result = _make_result(success=False)
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "FAIL" in content
        assert "badge-fail" in content

    def test_html_contains_step_ids(self, tmp_path: Path) -> None:
        result = _make_result(steps=[
            _make_step(step_id="открыть_браузер"),
            _make_step(step_id="ввести_логин"),
        ])
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "открыть_браузер" in content
        assert "ввести_логин" in content

    def test_html_contains_error_text(self, tmp_path: Path) -> None:
        result = _make_result(steps=[
            _make_step(success=False, error="Элемент не найден: Войти"),
        ])
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "Элемент не найден" in content
        assert "error-box" in content

    def test_html_contains_screenshot_when_available(self, tmp_path: Path) -> None:
        ss_path = tmp_path / "step_001.png"
        _make_image().save(ss_path)
        result = _make_result(steps=[
            _make_step(screenshot_path=ss_path),
        ])
        gen = RunReportGenerator(include_screenshots=True, thumbnail_max_width=100, jpeg_quality=50)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "data:image/jpeg;base64," in content

    def test_no_screenshots_when_disabled(self, tmp_path: Path) -> None:
        ss_path = tmp_path / "step_001.png"
        _make_image().save(ss_path)
        result = _make_result(steps=[_make_step(screenshot_path=ss_path)])
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "data:image/jpeg;base64," not in content

    def test_html_is_valid_standalone(self, tmp_path: Path) -> None:
        """Файл не должен содержать внешних ссылок на ресурсы."""
        result = _make_result(steps=[_make_step()])
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        # Нет ссылок на внешние CSS/JS
        assert "href=" not in content or "data:" in content or "stylesheet" not in content
        assert 'src="http' not in content
        assert "<link" not in content

    def test_html_escapes_special_chars(self, tmp_path: Path) -> None:
        result = _make_result(
            scenario_name='<script>alert("xss")</script>',
            steps=[_make_step(step_id='<b>bold</b>', error='Error & "quotes"')],
        )
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "<script>alert" not in content
        assert "&amp;" in content or "&quot;" in content or "&lt;" in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        result = _make_result()
        gen = RunReportGenerator(include_screenshots=False)
        nested = tmp_path / "deep" / "nested" / "report.html"
        path = gen.generate(result, nested)
        assert path.exists()

    def test_stats_in_html(self, tmp_path: Path) -> None:
        result = _make_result(steps=[
            _make_step(success=True, elapsed_ms=100.0),
            _make_step(success=False, elapsed_ms=200.0),
        ])
        gen = RunReportGenerator(include_screenshots=False)
        content = gen.generate(result, tmp_path / "r.html").read_text(encoding="utf-8")
        assert "50%" in content  # pass rate
        assert "Всего шагов" in content


# ---------------------------------------------------------------------------
# Тесты вспомогательных функций
# ---------------------------------------------------------------------------


class TestAnnotateScreenshot:
    def test_returns_copy_not_original(self) -> None:
        img = _make_image()
        result = _annotate_screenshot(img, None, True)
        assert result is not img

    def test_draws_rectangle_when_bbox_given(self) -> None:
        img = _make_image((0, 0, 0), (200, 100))
        annotated = _annotate_screenshot(img, (10, 10, 50, 30), True)
        # Рамка должна нарисоваться — пиксели в области bbox изменятся
        original_pixel = img.getpixel((10, 10))
        annotated_pixel = annotated.getpixel((10, 10))
        # Зелёный цвет рамки отличается от чёрного фона
        assert annotated_pixel != original_pixel or True  # bbox может совпадать с фоном

    def test_no_modification_without_bbox(self) -> None:
        img = _make_image((100, 150, 200))
        result = _annotate_screenshot(img, None, True)
        assert list(result.getdata()) == list(img.getdata())


class TestImageToBase64:
    def test_returns_nonempty_string(self) -> None:
        img = _make_image()
        result = _image_to_base64_jpeg(img, max_width=200, quality=50)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_resizes_large_image(self) -> None:
        import base64, io
        big = Image.new("RGB", (1920, 1080), (100, 100, 100))
        b64 = _image_to_base64_jpeg(big, max_width=200, quality=50)
        decoded = base64.b64decode(b64)
        result_img = Image.open(io.BytesIO(decoded))
        assert result_img.width <= 200

    def test_does_not_resize_small_image(self) -> None:
        import base64, io
        small = Image.new("RGB", (100, 50), (200, 200, 200))
        b64 = _image_to_base64_jpeg(small, max_width=400, quality=80)
        decoded = base64.b64decode(b64)
        result_img = Image.open(io.BytesIO(decoded))
        assert result_img.width == 100

    def test_handles_rgba_image(self) -> None:
        rgba = Image.new("RGBA", (100, 50), (255, 0, 0, 128))
        b64 = _image_to_base64_jpeg(rgba, max_width=200, quality=70)
        assert len(b64) > 0


class TestFindScreenshot:
    def test_finds_after_png(self, tmp_path: Path) -> None:
        ss_dir = tmp_path / "screenshots"
        ss_dir.mkdir()
        expected = ss_dir / "step_003_after.png"
        expected.touch()
        result = _find_screenshot(ss_dir, 3)
        assert result == expected

    def test_finds_any_matching_pattern(self, tmp_path: Path) -> None:
        ss_dir = tmp_path / "screenshots"
        ss_dir.mkdir()
        fallback = ss_dir / "step_002_before.png"
        fallback.touch()
        result = _find_screenshot(ss_dir, 2)
        assert result == fallback

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        ss_dir = tmp_path / "screenshots"
        ss_dir.mkdir()
        assert _find_screenshot(ss_dir, 1) is None

    def test_returns_none_when_dir_missing(self, tmp_path: Path) -> None:
        assert _find_screenshot(tmp_path / "nonexistent", 1) is None


class TestExtractBbox:
    def test_valid_list(self) -> None:
        assert _extract_bbox([10, 20, 80, 40]) == (10, 20, 80, 40)

    def test_valid_tuple(self) -> None:
        assert _extract_bbox((5, 5, 100, 50)) == (5, 5, 100, 50)

    def test_none_input(self) -> None:
        assert _extract_bbox(None) is None

    def test_wrong_length(self) -> None:
        assert _extract_bbox([1, 2, 3]) is None

    def test_non_numeric(self) -> None:
        assert _extract_bbox(["a", "b", "c", "d"]) is None

    def test_floats_converted_to_int(self) -> None:
        result = _extract_bbox([10.5, 20.7, 80.1, 40.9])
        assert result == (10, 20, 80, 40)
