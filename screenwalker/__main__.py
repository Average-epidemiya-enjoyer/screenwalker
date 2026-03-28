"""CLI entry point for ScreenWalker.

Usage:
    python -m screenwalker run scenario.yaml
    python -m screenwalker run scenario.yaml --config config/default.yaml
    python -m screenwalker run scenario.yaml --dry-run
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

    # Parse --var KEY=VALUE pairs into a dict
    overrides: dict[str, str] = {}
    for pair in var:
        if "=" not in pair:
            click.echo(f"Error: --var value must be KEY=VALUE, got: {pair!r}", err=True)
            sys.exit(1)
        key, _, value = pair.partition("=")
        overrides[key.strip()] = value.strip()

    # TODO: load config and merge with defaults
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
