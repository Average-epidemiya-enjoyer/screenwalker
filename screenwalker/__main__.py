"""Точка входа CLI для ScreenWalker.

Использование:
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
    """ScreenWalker — RPA-фреймворк на основе компьютерного зрения."""


@main.command()
@click.argument("scenario", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Путь к YAML-файлу конфигурации (накладывается поверх значений по умолчанию).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Валидировать сценарий и вывести запланированные шаги без выполнения действий.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    show_default=True,
    help="Уровень детализации логирования.",
)
@click.option(
    "--var",
    "-v",
    multiple=True,
    metavar="KEY=VALUE",
    help="Переопределить переменную сценария (можно указывать несколько раз). Пример: --var username=admin",
)
def run(
    scenario: Path,
    config: Path | None,
    dry_run: bool,
    log_level: str,
    var: tuple[str, ...],
) -> None:
    """Выполнить YAML-файл сценария.

    SCENARIO — путь к YAML-файлу сценария для запуска.
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
    help="Путь к YAML-файлу конфигурации.",
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
    """Разобрать и валидировать YAML сценария без выполнения каких-либо действий.

    Завершается с кодом 0 при успехе, 1 при ошибке валидации.
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
    help="Имя шаблона (сохраняется как templates/<name>.png).",
)
@click.option(
    "--region",
    "-r",
    required=True,
    metavar="X,Y,W,H",
    help="Регион экрана для захвата: left,top,width,height (пиксели).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("templates"),
    show_default=True,
    help="Директория для сохранения изображения шаблона.",
)
@click.option(
    "--delay",
    "-d",
    type=float,
    default=0.3,
    show_default=True,
    help="Секунды ожидания перед захватом (время для позиционирования UI).",
)
def capture_template(
    name: str,
    region: str,
    output_dir: Path,
    delay: float,
) -> None:
    """Захватить регион экрана и сохранить его как именованное изображение-шаблон.

    Пример:
        python -m screenwalker capture-template --name ok_button --region 100,200,80,30
    """
    # Парсим регион
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


@main.command("report")
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Путь к выходному HTML-файлу. По умолчанию: <log_dir>/report.html",
)
@click.option(
    "--thumbnail-width",
    type=int,
    default=400,
    show_default=True,
    help="Максимальная ширина миниатюр скриншотов (пиксели).",
)
@click.option(
    "--jpeg-quality",
    type=int,
    default=72,
    show_default=True,
    help="Качество JPEG для встроенных скриншотов (10–100).",
)
@click.option(
    "--no-screenshots",
    is_flag=True,
    default=False,
    help="Не включать скриншоты в отчёт (уменьшает размер файла).",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="warning",
    show_default=True,
)
def report(
    log_dir: Path,
    output: Path | None,
    thumbnail_width: int,
    jpeg_quality: int,
    no_screenshots: bool,
    log_level: str,
) -> None:
    """Сгенерировать HTML-отчёт из директории с логами.

    LOG_DIR — директория с файлом steps.jsonl и папкой screenshots/.

    Пример:

        python -m screenwalker report logs/2024-01-15_run/ --output report.html
    """
    _configure_logging(log_level)

    from screenwalker.reporting.html_report import RunReportGenerator, RunResult

    output_path = output or (log_dir / "report.html")

    try:
        run_result = RunResult.from_log_dir(log_dir)
    except FileNotFoundError as exc:
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(1)

    generator = RunReportGenerator(
        thumbnail_max_width=thumbnail_width,
        jpeg_quality=jpeg_quality,
        include_screenshots=not no_screenshots,
    )

    try:
        report_path = generator.generate(run_result, output_path)
    except Exception as exc:
        click.echo(f"Ошибка генерации отчёта: {exc}", err=True)
        sys.exit(1)

    status = "PASS" if run_result.success else "FAIL"
    click.echo(
        f"{status}  {run_result.scenario_name} — "
        f"{len(run_result.steps)} шагов, "
        f"{run_result.duration_s:.1f} с"
    )
    click.echo(f"Отчёт сохранён: {report_path}")


@main.command("analyze")
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="warning",
    show_default=True,
)
def analyze(log_dir: Path, log_level: str) -> None:
    """Проанализировать логи выполнения в LOG_DIR и вывести отчёт об обучении.

    Читает все файлы steps.jsonl в LOG_DIR, агрегирует статистику шагов,
    обнаруживает OCR-расхождения и предлагает оптимизации таймаутов.

    Пример:

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
    help="Минимальное количество наблюдений для включения кандидата.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="warning",
    show_default=True,
)
def suggest_synonyms(log_dir: Path, min_count: int, log_level: str) -> None:
    """Предложить новые записи синонимов на основе OCR-расхождений в LOG_DIR.

    Анализирует файлы steps.jsonl и выводит пары, где OCR-чтение
    отличалось от ожидаемого текста — это кандидаты для словаря
    config/synonyms.yaml.

    Пример:

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
    """Разбирает пары ``KEY=VALUE`` из опций --var.

    Args:
        var: Кортеж «сырых» строк в формате ``KEY=VALUE``.

    Returns:
        Словарь переопределений переменных.
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
    """Настраивает structlog с запрошенным уровнем детализации.

    Args:
        level: Одно из значений: debug | info | warning | error.
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
