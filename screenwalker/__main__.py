"""CLI entry point for ScreenWalker.

Usage:
    python -m screenwalker run scenario.yaml
    python -m screenwalker run scenario.yaml --config config/default.yaml
    python -m screenwalker run scenario.yaml --dry-run
    python -m screenwalker validate scenario.yaml
    python -m screenwalker capture-template --name button_ok --region 100,200,50,30
    python -m screenwalker --version
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import structlog

from screenwalker import __version__
from screenwalker.core.engine import ScenarioEngine
from screenwalker.utils.config import load_config

logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="screenwalker")
def main() -> None:
    """ScreenWalker — vision-based RPA framework."""


@main.command()
@click.argument("scenario", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a YAML config file (merged on top of defaults).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate scenario and print planned steps without executing actions.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    show_default=True,
    help="Logging verbosity level.",
)
@click.option(
    "--var",
    "-v",
    multiple=True,
    metavar="KEY=VALUE",
    help="Override scenario variable (repeatable). Example: --var username=admin",
)
def run(
    scenario: Path,
    config: Path | None,
    dry_run: bool,
    log_level: str,
    var: tuple[str, ...],
) -> None:
    """Execute a YAML scenario file.

    SCENARIO is the path to the scenario YAML file to run.
    """
    _configure_logging(log_level)

    log = structlog.get_logger(__name__)
    log.info("Starting ScreenWalker", scenario=str(scenario), dry_run=dry_run)

    overrides = _parse_var_overrides(var)

    cfg = load_config(config)

    try:
        engine = ScenarioEngine.from_yaml(scenario, config=cfg, variable_overrides=overrides)
        if dry_run:
            engine.dry_run()
        else:
            engine.run()
    except Exception as exc:
        log.error("Scenario failed", error=str(exc), exc_info=True)
        sys.exit(1)

    log.info("Scenario completed successfully")


@main.command()
@click.argument("scenario", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a YAML config file.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="warning",
    show_default=True,
)
def validate(
    scenario: Path,
    config: Path | None,
    log_level: str,
) -> None:
    """Parse and validate a scenario YAML without executing any actions.

    Exits with code 0 on success, 1 on validation error.
    """
    _configure_logging(log_level)
    log = structlog.get_logger(__name__)

    cfg = load_config(config)
    try:
        engine = ScenarioEngine.from_yaml(scenario, config=cfg)
    except Exception as exc:
        click.echo(f"Validation FAILED: {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"OK  {scenario.name} — {len(engine.steps)} step(s), "
        f"{len(engine.teardown_steps)} teardown step(s)"
    )
    log.info(
        "Validation passed",
        scenario=str(scenario),
        steps=len(engine.steps),
        variables=list(engine.context.variables.keys()),
    )


@main.command("capture-template")
@click.option(
    "--name",
    "-n",
    required=True,
    help="Template name (saved as templates/<name>.png).",
)
@click.option(
    "--region",
    "-r",
    required=True,
    metavar="X,Y,W,H",
    help="Screen region to capture: left,top,width,height (pixels).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("templates"),
    show_default=True,
    help="Directory to save the template image.",
)
@click.option(
    "--delay",
    "-d",
    type=float,
    default=0.3,
    show_default=True,
    help="Seconds to wait before capturing (gives time to position the UI).",
)
def capture_template(
    name: str,
    region: str,
    output_dir: Path,
    delay: float,
) -> None:
    """Capture a screen region and save it as a named template image.

    Example:
        python -m screenwalker capture-template --name ok_button --region 100,200,80,30
    """
    # Parse region
    try:
        parts = [int(p.strip()) for p in region.split(",")]
        if len(parts) != 4:
            raise ValueError("Expected exactly 4 comma-separated integers")
        x, y, w, h = parts
    except ValueError as exc:
        click.echo(f"Error: --region must be X,Y,W,H integers: {exc}", err=True)
        sys.exit(1)

    if delay > 0:
        click.echo(f"Capturing in {delay:.1f}s — switch to the target window...")
        import time
        time.sleep(delay)

    try:
        from screenwalker.vision.capture import ScreenCapture
        capture = ScreenCapture(screenshot_delay=0.0)
        image = capture.capture_region(x, y, w, h)
    except Exception as exc:
        click.echo(f"Capture failed: {exc}", err=True)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.png"
    image.save(out_path, "PNG")
    click.echo(f"Template saved: {out_path}  ({w}x{h} px at {x},{y})")


@main.command("analyze")
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="warning",
    show_default=True,
)
def analyze(log_dir: Path, log_level: str) -> None:
    """Analyse execution logs in LOG_DIR and print a learning report.

    Reads all steps.jsonl files found under LOG_DIR, aggregates step
    statistics, detects OCR mismatches, and suggests timeout optimisations.

    Example:

        python -m screenwalker analyze logs/
    """
    _configure_logging(log_level)

    from screenwalker.learning.patterns import PatternLearner
    from screenwalker.matching.synonyms import SynonymRegistry

    registry = SynonymRegistry(load_defaults=True)
    learner = PatternLearner(registry=registry)
    report = learner.analyze_logs(log_dir)
    click.echo(report.format_text())


@main.command("suggest-synonyms")
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--min-count",
    "-n",
    type=int,
    default=1,
    show_default=True,
    help="Minimum observation count to include a candidate.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="warning",
    show_default=True,
)
def suggest_synonyms(log_dir: Path, min_count: int, log_level: str) -> None:
    """Suggest new synonym entries based on OCR mismatches in LOG_DIR.

    Analyses steps.jsonl files and prints pairs where the OCR reading
    differed from the expected text — these are candidates for the
    config/synonyms.yaml dictionary.

    Example:

        python -m screenwalker suggest-synonyms logs/
    """
    _configure_logging(log_level)

    from screenwalker.learning.patterns import PatternLearner
    from screenwalker.matching.synonyms import SynonymRegistry

    registry = SynonymRegistry(load_defaults=True)
    learner = PatternLearner(registry=registry, min_observations=min_count)
    report = learner.analyze_logs(log_dir)

    suggestions = learner.suggest_synonyms()
    if not suggestions:
        click.echo("No synonym suggestions found.")
        return

    click.echo("=== Synonym Suggestions ===")
    click.echo(
        "Add these to config/synonyms.yaml under the relevant group:\n"
    )
    for label, candidates in sorted(suggestions.items()):
        mismatches = [
            m for m in report.ocr_mismatches if m.find_target.lower() == label
        ]
        for cand in candidates:
            count = next((m.count for m in mismatches if m.found_text == cand), 1)
            if count >= min_count:
                click.echo(f"  [{label}]  +  '{cand}'  ({count} observation(s))")


def _parse_var_overrides(var: tuple[str, ...]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` pairs from --var options.

    Args:
        var: Tuple of raw ``KEY=VALUE`` strings.

    Returns:
        Dict of variable overrides.
    """
    overrides: dict[str, str] = {}
    for pair in var:
        if "=" not in pair:
            click.echo(f"Error: --var must be KEY=VALUE, got: {pair!r}", err=True)
            sys.exit(1)
        key, _, value = pair.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def _configure_logging(level: str) -> None:
    """Configure structlog with the requested verbosity.

    Args:
        level: One of debug | info | warning | error.
    """
    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


if __name__ == "__main__":
    main()
