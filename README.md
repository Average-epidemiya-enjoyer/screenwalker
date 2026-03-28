# ScreenWalker

Vision-based RPA framework. Automates any UI by reading the screen —
OCR, template matching, and YOLO detection. No DOM access, no APIs,
no application-specific plugins required.

---

## 5-Minute Demo Quickstart

### Prerequisites

| Requirement | Check |
|-------------|-------|
| Python 3.10+ | `python --version` |
| Tesseract OCR | `tesseract --version` |
| ScreenWalker dependencies | `pip install -e .` |

**Install Tesseract (Windows):**
Download the installer from https://github.com/UB-Mannheim/tesseract/wiki and add it to `PATH`.

**Install Tesseract (macOS):**
```bash
brew install tesseract
```

**Install Tesseract (Linux):**
```bash
sudo apt install tesseract-ocr
```

**Install Python dependencies:**
```bash
pip install -e ".[ocr]"
# or manually:
pip install pytesseract pillow pyautogui pyperclip rapidfuzz pydantic structlog click pyyaml numpy
```

---

### Run the demos

```bash
# Both demos (Calculator + Notepad)
python scripts/run_demo.py

# Calculator only  (7 + 3 = 10)
python scripts/run_demo.py --scenario calc

# Notepad only  (type → copy → verify)
python scripts/run_demo.py --scenario notepad

# Validate without executing any actions
python scripts/run_demo.py --dry-run

# Show debug engine logs
python scripts/run_demo.py --verbose
```

Or use the standard CLI directly:

```bash
python -m screenwalker run scenarios/demo_calculator.yaml --config config/demo.yaml
python -m screenwalker run scenarios/demo_notepad.yaml   --config config/demo.yaml
```

---

### What the demos do

#### Calculator demo  (`scenarios/demo_calculator.yaml`)

1. Launches `calc.exe` (Windows Calculator in Standard mode)
2. Waits for the UI to be visible (OCR finds "Calculator")
3. Clicks button **7** via OCR text search
4. Clicks button **+** via OCR text search
5. Clicks button **3** via OCR text search
6. Clicks button **=** via OCR text search
7. Reads the result display with OCR — asserts it equals **"10"**
8. Saves a screenshot to `logs/demo/`

#### Notepad demo  (`scenarios/demo_notepad.yaml`)

1. Launches `notepad.exe`
2. Waits for the editor window (OCR finds "Notepad")
3. Types **"Hello from Screenwalker!"** into the text area
4. OCR asserts the text is visible on screen
5. Presses `Ctrl+A` to select all
6. Copies to clipboard via `Ctrl+C` — stores in context variable `clipboard_text`
7. OCR-asserts the text still matches the expected string
8. Saves a screenshot
9. Closes with `Alt+F4` and clicks **"Don't Save"**

---

### Expected output

```
╭──────────────────────────────────────────────────────╮
│  ScreenWalker  vision-based RPA demo                 │
│  Automates any UI via OCR + computer vision          │
╰──────────────────────────────────────────────────────╯

▶ Validating scenarios
  ✓  demo_calculator.yaml  —  8 step(s), 1 teardown step(s)
  ✓  demo_notepad.yaml     —  10 step(s), 1 teardown step(s)

▶ Running: Calculator  (7 + 3 = 10)
  ✓  PASSED  in 12.4s

▶ Running: Notepad  (clipboard round-trip)
  ✓  PASSED  in 9.8s

  Demo Results
  ┌──────────────────────────────────┬────────┬───────┬────────────────────────────┐
  │ Scenario                         │ Status │  Time │ Notes                      │
  ├──────────────────────────────────┼────────┼───────┼────────────────────────────┤
  │ Calculator  (7 + 3 = 10)         │  PASS  │ 12.4s │                            │
  │ Notepad  (clipboard round-trip)  │  PASS  │  9.8s │ clipboard='Hello from ...' │
  └──────────────────────────────────┴────────┴───────┴────────────────────────────┘

  ✓  All 2 demo(s) passed (22.2s total)
```

Logs and screenshots are saved to `logs/demo/`.

---

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `tesseract: command not found` | Install Tesseract and add to PATH |
| `ModuleNotFoundError: pytesseract` | `pip install pytesseract` |
| `ElementNotFound: "Calculator"` | Make sure Calculator opened in Standard mode; try `--verbose` |
| OCR finds wrong "7" (e.g. in title bar) | Open Calculator first so the screen is clean; or add a `region` to the find spec |
| Calculator opens in History/Scientific mode | Switch to Standard mode manually, or pass `--var` override |
| Notepad "Don't Save" not found | Windows 10: try `query: "Don't Save"` — Windows 11 text may vary |
| `pyautogui.FailSafeException` | Move mouse away from top-left corner; or set `pyautogui.FAILSAFE = False` |
| `import pyautogui` fails on Linux | `sudo apt install python3-tk python3-dev` |

---

### Adapting the demos

#### macOS

Change `target` in both scenarios:

```yaml
# Calculator
target: "open -a Calculator"

# Text editor
target: "open -a TextEdit"
```

#### Linux

```yaml
# Calculator
target: "gnome-calculator"

# Text editor
target: "gedit"
```

---

## Writing your own scenarios

A scenario is a YAML file. Minimal example:

```yaml
name: My Automation
steps:
  - id: open_app
    action: launch
    target: "notepad.exe"
    wait_after: 2.0

  - id: type_text
    action: type
    text: "Hello!"

  - id: save
    action: hotkey
    keys: [ctrl, s]
    wait_after: 1.0
```

### Available actions

| Action | What it does | Required fields |
|--------|-------------|-----------------|
| `launch` | Start an application | `target` |
| `click` | Left-click a located element | `find` |
| `double_click` | Double-click | `find` |
| `right_click` | Right-click | `find` |
| `type` | Type text | `text` |
| `hotkey` | Key combination | `keys` |
| `scroll` | Scroll at element | `find` (optional), `extra.direction`, `extra.clicks` |
| `drag` | Drag to position | `find`, `extra.to` |
| `copy` | Ctrl+C → clipboard | — |
| `paste` | Ctrl+V | — |
| `assert_visible` | Assert element is on screen | `find` |
| `assert_text` | Assert OCR text matches | `find`, `text` |
| `wait` | Sleep | `wait` (seconds) |
| `screenshot` | Save screenshot | `label` (optional) |

### Find methods

```yaml
find:
  method: ocr          # or template, yolo
  query: "Submit"      # text to find (OCR) or template name
  threshold: 0.75      # minimum confidence
  fuzzy: true          # enable fuzzy text matching
  fuzzy_threshold: 80  # 0-100 (rapidfuzz score)
  region: [x, y, w, h] # restrict search area (pixels)
  offset: [dx, dy]     # pixel offset from found centre
```

### Variables

Variables are defined in the `variables:` section and interpolated
with `{{ var_name }}` syntax:

```yaml
variables:
  username: admin

steps:
  - id: type_user
    action: type
    text: "{{ username }}"
```

Override from command line:
```bash
python -m screenwalker run scenario.yaml --var username=myuser
```

### Screen fingerprints

Verify the UI is in the right state before acting:

```yaml
screens:
  login_page:
    required_texts: ["Sign In", "Password"]
    forbidden_texts: ["Dashboard"]
    match_threshold: 0.70

steps:
  - id: type_password
    action: type
    text: "{{ password }}"
    expect_screen: login_page
```

---

## CLI reference

```bash
# Run a scenario
python -m screenwalker run SCENARIO [--config CONFIG] [--dry-run] [--var KEY=VALUE ...]

# Validate without running
python -m screenwalker validate SCENARIO

# Capture a template image
python -m screenwalker capture-template --name ok_button --region 100,200,80,30 --delay 3

# Analyse execution logs
python -m screenwalker analyze logs/

# Suggest synonym entries for OCR mismatches
python -m screenwalker suggest-synonyms logs/

# Version
python -m screenwalker --version
```

---

## Project structure

```
screenwalker/
├── config/
│   ├── default.yaml          ← default configuration
│   ├── demo.yaml             ← demo-optimised config
│   └── synonyms.yaml         ← OCR synonym groups (33 groups, en+ru)
├── scenarios/
│   ├── example_scenario.yaml ← Notepad automation example
│   ├── demo_calculator.yaml  ← Calculator demo (7+3=10)
│   └── demo_notepad.yaml     ← Notepad clipboard demo
├── scripts/
│   ├── run_demo.py           ← pretty demo runner
│   ├── download_model.py     ← download OmniParser V2 YOLO weights
│   └── train_detector.py     ← fine-tune YOLO on custom screenshots
├── templates/                ← PNG templates for template matching
├── screenwalker/
│   ├── core/                 ← ScenarioEngine, Step, RunContext, errors
│   ├── vision/               ← OCR, template matching, YOLO detector, screen state
│   ├── actions/              ← mouse, keyboard, clipboard controllers
│   ├── learning/             ← action cache, step logger, pattern learner
│   ├── matching/             ← fuzzy matcher, synonym registry
│   └── utils/                ← config, retry
└── tests/                    ← 464 tests, 81% coverage
```

---

## Architecture overview

```
YAML scenario
      │
      ▼
ScenarioEngine.from_yaml()
      │
      ├── VisionConfig → TesseractEngine / TemplateMatcher / UIElementDetector
      ├── ActionsConfig → MouseController / KeyboardController / ClipboardManager
      └── LearningConfig → ActionCache / StepLogger
      │
      ▼
engine.run()
      │
      for each step:
        1. interpolate {{ variables }}
        2. capture screenshot
        3. verify expect_screen (optional)
        4. locate element:
              cache hit?    → use cached BBox
              step.find     → OCR / template / YOLO
              step.fallback → alternative finder
              YOLO fallback → auto-detect if yolo_enabled
        5. dispatch action (click, type, hotkey, …)
        6. wait_after
        7. log to JSONL + screenshot
      │
      ▼
teardown steps (always run)
```

---

## Optional: YOLO detection

For richer UI element discovery (icons, checkboxes, sliders without visible text):

```bash
# Download OmniParser V2 weights (~25 MB)
python scripts/download_model.py

# Enable in config/demo.yaml:
# vision:
#   yolo_enabled: true
#   yolo_model_path: models/icon_detect/best.pt
#   yolo_confidence: 0.50
```

Then use `method: yolo` in find specs, or benefit from the automatic YOLO
fallback that kicks in when OCR and template matching both fail.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest                        # all 464 tests
pytest tests/test_detector.py # YOLO detector tests only
pytest --tb=short -q          # compact output with coverage report
```
