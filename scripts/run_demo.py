"""ScreenWalker demo runner.

Runs the calculator and/or notepad demo scenarios with a rich CLI interface
showing step-by-step progress, timing, and a pass/fail summary.

Usage::

    python scripts/run_demo.py                    # both demos
    python scripts/run_demo.py --scenario calc    # calculator only
    python scripts/run_demo.py --scenario notepad # notepad only
    python scripts/run_demo.py --dry-run          # validate without executing
    python scripts/run_demo.py --verbose          # show debug logs
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path so this script works when run directly
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Rich output (falls back gracefully if rich is not installed)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich import print as rprint
    from rich.text import Text
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None  # type: ignore[assignment]


def _echo(msg: str, style: str = "") -> None:
    if _RICH:
        console.print(msg)  # type: ignore[union-attr]
    else:
        # Strip basic rich markup for plain output
        import re
        clean = re.sub(r"\[/?[^\]]*\]", "", msg)
        print(clean)


def _banner() -> None:
    if _RICH:
        console.print(Panel.fit(  # type: ignore[union-attr]
            "[bold cyan]ScreenWalker[/bold cyan]  [dim]vision-based RPA demo[/dim]\n"
            "[dim]Automates any UI via OCR + computer vision - no DOM/API needed[/dim]",
            border_style="cyan",
        ))
    else:
        print("=" * 60)
        print("  ScreenWalker - vision-based RPA demo")
        print("  Automates any UI via OCR + computer vision")
        print("=" * 60)


def _section(title: str) -> None:
    if _RICH:
        console.print(f"\n[bold yellow]>> {title}[/bold yellow]")  # type: ignore[union-attr]
    else:
        print(f"\n-- {title} --")


def _ok(msg: str) -> None:
    _echo(f"  [bold green]OK[/bold green]  {msg}" if _RICH else f"  [OK]  {msg}")


def _fail(msg: str) -> None:
    _echo(f"  [bold red]FAIL[/bold red]  {msg}" if _RICH else f"  [FAIL]  {msg}")


def _info(msg: str) -> None:
    _echo(f"  [dim]{msg}[/dim]" if _RICH else f"       {msg}")


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def validate_scenario(scenario_path: Path, config_path: Path) -> bool:
    """Validate a scenario YAML, printing the result.

    Returns True if valid, False otherwise.
    """
    from screenwalker.core.engine import ScenarioEngine
    from screenwalker.utils.config import load_config

    try:
        cfg = load_config(config_path)
        engine = ScenarioEngine.from_yaml(scenario_path, config=cfg)
        _ok(
            f"[bold]{scenario_path.name}[/bold]  -  "
            f"{len(engine.steps)} step(s), "
            f"{len(engine.teardown_steps)} teardown step(s)"
            if _RICH else
            f"{scenario_path.name}  -  {len(engine.steps)} step(s)"
        )
        _info(f"Variables: {list(engine.context.variables.keys())}")
        return True
    except Exception as exc:
        _fail(f"[bold]{scenario_path.name}[/bold]: {exc}" if _RICH else f"{scenario_path.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------


class StepTracker:
    """Hooks into engine log output to track step progress."""

    def __init__(self) -> None:
        self.completed: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []


def run_scenario(
    scenario_path: Path,
    config_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[bool, float, dict[str, Any]]:
    """Run a single scenario and return (success, elapsed_seconds, context_vars).

    Args:
        scenario_path: Path to the YAML scenario file.
        config_path: Path to the config YAML.
        dry_run: If True, validate and print steps without executing.
        verbose: If True, emit debug-level logs.

    Returns:
        (success, elapsed_seconds, context_variables)
    """
    import logging
    import structlog

    log_level = "debug" if verbose else "warning"
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    from screenwalker.core.engine import ScenarioEngine
    from screenwalker.utils.config import load_config

    cfg = load_config(config_path)
    engine = ScenarioEngine.from_yaml(scenario_path, config=cfg)

    t0 = time.monotonic()
    success = False
    context_vars: dict[str, Any] = {}

    try:
        if dry_run:
            engine.dry_run()
        else:
            engine.run()
        success = True
        context_vars = dict(engine.context.variables)
    except Exception as exc:
        _fail(str(exc))

    elapsed = time.monotonic() - t0
    return success, elapsed, context_vars


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict[str, Any]]) -> None:
    if _RICH:
        table = Table(title="Demo Results", show_header=True, header_style="bold cyan")
        table.add_column("Scenario", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Time", justify="right")
        table.add_column("Notes")
        for r in results:
            status = "[bold green]PASS[/bold green]" if r["success"] else "[bold red]FAIL[/bold red]"
            notes = r.get("notes", "")
            table.add_row(r["name"], status, f"{r['elapsed']:.1f}s", notes)
        console.print("\n")  # type: ignore[union-attr]
        console.print(table)  # type: ignore[union-attr]
    else:
        print("\n── Demo Results ──")
        print(f"{'Scenario':<30}  {'Status':<6}  {'Time':>6}  Notes")
        print("-" * 70)
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            print(f"{r['name']:<30}  {status:<6}  {r['elapsed']:>5.1f}s  {r.get('notes', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ScreenWalker demo scenarios with rich CLI output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_demo.py                    # run both demos
  python scripts/run_demo.py --scenario calc    # calculator only
  python scripts/run_demo.py --scenario notepad # notepad only
  python scripts/run_demo.py --dry-run          # validate + print plan
  python scripts/run_demo.py --verbose          # full debug logs
        """,
    )
    parser.add_argument(
        "--scenario",
        choices=["calc", "notepad", "both"],
        default="both",
        help="Which demo to run (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned steps without executing any actions",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show debug-level engine logs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config" / "demo.yaml",
        help="Config file (default: config/demo.yaml)",
    )
    args = parser.parse_args()

    config_path: Path = args.config
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    scenarios_to_run: list[tuple[str, Path]] = []
    if args.scenario in ("calc", "both"):
        scenarios_to_run.append(("Calculator  (7 + 3 = 10)", _ROOT / "scenarios" / "demo_calculator.yaml"))
    if args.scenario in ("notepad", "both"):
        scenarios_to_run.append(("Notepad  (clipboard round-trip)", _ROOT / "scenarios" / "demo_notepad.yaml"))

    _banner()

    # ── Validation pass ──────────────────────────────────────────────────────
    _section("Validating scenarios")
    all_valid = True
    for name, path in scenarios_to_run:
        if not validate_scenario(path, config_path):
            all_valid = False

    if not all_valid:
        _fail("One or more scenarios failed validation — aborting.")
        sys.exit(1)

    if args.dry_run:
        _section("Dry-run mode — printing planned steps")
        for name, path in scenarios_to_run:
            _echo(f"\n[bold]-- {name} --[/bold]" if _RICH else f"\n-- {name} --")
            run_scenario(path, config_path, dry_run=True, verbose=args.verbose)
        _ok("Dry-run complete - no actions were executed.")
        return

    # ── Prerequisites reminder ───────────────────────────────────────────────
    _section("Prerequisites check")
    _info("Ensure Tesseract OCR is installed and on PATH:  tesseract --version")
    _info("Close any existing Calculator / Notepad windows for a clean start.")
    if _RICH:
        console.print()  # type: ignore[union-attr]

    input_fn = input
    try:
        answer = input_fn("  Press [Enter] to start, or Ctrl+C to abort... ")
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    # ── Execute scenarios ────────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    overall_start = time.monotonic()

    for name, path in scenarios_to_run:
        _section(f"Running: {name}")
        _info(f"Scenario file : {path.relative_to(_ROOT)}")
        _info(f"Config file   : {config_path.relative_to(_ROOT)}")
        _info(f"Logs          : {config_path.parent.parent / 'logs' / 'demo'}")
        _echo("")

        if _RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"  Executing {path.name} ...", total=None)
                success, elapsed, ctx_vars = run_scenario(
                    path, config_path, dry_run=False, verbose=args.verbose
                )
        else:
            print(f"  Executing {path.name} ...")
            success, elapsed, ctx_vars = run_scenario(
                path, config_path, dry_run=False, verbose=args.verbose
            )

        # Show result
        notes = ""
        if "clipboard_text" in ctx_vars:
            notes = f"clipboard={ctx_vars['clipboard_text']!r}"
        if success:
            _ok(f"[bold green]PASSED[/bold green]  in {elapsed:.1f}s" if _RICH else f"PASSED in {elapsed:.1f}s")
        else:
            _fail(f"[bold red]FAILED[/bold red]  after {elapsed:.1f}s" if _RICH else f"FAILED after {elapsed:.1f}s")
        if notes:
            _info(notes)

        results.append({"name": name, "success": success, "elapsed": elapsed, "notes": notes})

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - overall_start
    _print_summary(results)

    passed = sum(1 for r in results if r["success"])
    total = len(results)
    _echo("")
    if passed == total:
        _ok(
            f"[bold green]All {total} demo(s) passed[/bold green]  "
            f"({total_elapsed:.1f}s total)"
            if _RICH else f"All {total} demo(s) passed ({total_elapsed:.1f}s total)"
        )
    else:
        _fail(
            f"[bold red]{total - passed} of {total} demo(s) failed[/bold red]"
            if _RICH else f"{total - passed} of {total} demo(s) failed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
