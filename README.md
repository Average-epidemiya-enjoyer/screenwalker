# ScreenWalker

Vision-based RPA framework for automating any UI via screenshots — no DOM access, no API hooks required.
Uses OpenCV template matching, OCR, and optional YOLO detection to locate elements and drive interactions.

---

## Architecture

```
screenwalker/
├── core/          # ScenarioEngine (state machine), Step, RunContext, errors
├── vision/        # Screen capture, OCR, template matching, YOLO detector, screen-state identification
├── actions/       # Mouse, keyboard, clipboard primitives
├── matching/      # Fuzzy text matching, UI synonym dictionary
├── learning/      # Step logger, action cache, pattern updater
└── utils/         # Config loader (Pydantic), retry decorator
```

### Key Concepts

| Concept | Description |
|---|---|
| **Scenario** | YAML file describing a sequence of steps |
| **Step** | Typed action unit: find element → act → assert |
| **FindResult** | Unified result from any vision method: `(element, confidence, bbox, method)` |
| **RunContext** | Per-run state bag: variables, history, screenshots |
| **ScenarioEngine** | State machine that executes steps, handles retries and branching |
| **ScreenState** | Identifies which "screen" the UI is on using OCR + template anchors |

### Vision Pipeline

```
Screenshot → Template Match ──┐
           → OCR Text Match ──┼──► FindResult(bbox, confidence, method)
           → YOLO Detection ──┘
```

All three finders implement the same `Finder` Protocol and return a `FindResult`.
`ScreenState` combines them to identify the current UI context before each step.

### Learning System

After each successful step the engine writes to a local cache:

```
screen_id + element_label → { bbox, method, confidence, timestamp }
```

On subsequent runs the cache is checked first (fast path) before running expensive vision.
`patterns.py` exposes hooks to expand synonym dictionaries from observed OCR text.

---

## Installation

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on `PATH`
- (Optional) CUDA-capable GPU for YOLO

### Install

```bash
# Clone
git clone https://github.com/your-org/screenwalker.git
cd screenwalker

# Install with pip (editable)
pip install -e .

# Or with optional YOLO support
pip install -e ".[yolo]"

# Dev dependencies
pip install -e ".[dev]"
```

### Verify

```bash
python -m screenwalker --version
```

---

## Quick Start

```bash
# Run the bundled example scenario
python -m screenwalker run scenarios/example_scenario.yaml

# Run with a custom config
python -m screenwalker run scenarios/example_scenario.yaml --config config/default.yaml

# Dry-run (validate only, no actions executed)
python -m screenwalker run scenarios/example_scenario.yaml --dry-run

# Verbose structured logs
python -m screenwalker run scenarios/example_scenario.yaml --log-level debug
```

---

## Writing Scenarios

```yaml
# scenarios/my_scenario.yaml
name: Login Flow
description: Log into the application

variables:
  username: "admin"
  password: "secret"

steps:
  - id: open_app
    action: launch
    target: "C:/Apps/MyApp.exe"

  - id: find_username_field
    action: click
    find:
      method: ocr          # ocr | template | yolo
      query: "Username"
      threshold: 0.7

  - id: type_username
    action: type
    text: "{{ username }}"

  - id: submit
    action: hotkey
    keys: ["enter"]

  - id: assert_logged_in
    action: assert_visible
    find:
      method: ocr
      query: "Dashboard"
    timeout: 10
```

### Supported Actions

| Action | Description |
|---|---|
| `click` | Left-click found element |
| `double_click` | Double-click found element |
| `right_click` | Right-click found element |
| `type` | Type text (supports `{{ variable }}`) |
| `hotkey` | Send key combination |
| `scroll` | Scroll at element position |
| `drag` | Drag from element to target |
| `copy` | Copy selection to clipboard variable |
| `paste` | Paste clipboard text |
| `assert_visible` | Assert element is visible |
| `assert_text` | Assert OCR text equals value |
| `wait` | Wait N seconds |
| `launch` | Launch application |
| `screenshot` | Save screenshot to logs |

---

## Configuration

Edit `config/default.yaml` to change global defaults:

```yaml
timeouts:
  step_default: 15        # seconds to wait for element
  screenshot_delay: 0.3   # seconds before capturing

vision:
  template_threshold: 0.8
  ocr_engine: tesseract   # tesseract | paddleocr
  ocr_lang: eng
  multi_scale: true

logging:
  level: info
  output_dir: logs/
  save_screenshots: true
```

---

## Running Tests

```bash
pytest                          # all tests with coverage
pytest tests/test_ocr.py -v     # single module
pytest --cov-report=html        # HTML coverage report
```

---

## Project Status

This is an alpha skeleton. Core interfaces are defined; implementations are marked with `TODO`.

Roadmap:
- [ ] Full Tesseract OCR integration
- [ ] Multi-scale template matching
- [ ] YOLO UI element detection model
- [ ] Action cache persistence
- [ ] Web UI for scenario authoring
- [ ] CI integration (headless mode)
