# ScreenWalker

Фреймворк RPA на основе компьютерного зрения. Автоматизирует любой интерфейс, читая экран —
OCR, сопоставление шаблонов и YOLO-детекция. Не требует доступа к DOM, API
и специфических плагинов для приложений.

---

## Быстрый старт: демо за 5 минут

### Предварительные требования

| Требование | Проверка |
|-------------|-------|
| Python 3.10+ | `python --version` |
| Tesseract OCR | `tesseract --version` |
| Зависимости ScreenWalker | `pip install -e .` |

**Установка Tesseract (Windows):**
Скачайте установщик с https://github.com/UB-Mannheim/tesseract/wiki и добавьте его в `PATH`.

**Установка Tesseract (macOS):**
```bash
brew install tesseract
```

**Установка Tesseract (Linux):**
```bash
sudo apt install tesseract-ocr
```

**Установка зависимостей Python:**
```bash
pip install -e ".[ocr]"
# или вручную:
pip install pytesseract pillow pyautogui pyperclip rapidfuzz pydantic structlog click pyyaml numpy
```

---

### Запуск демо

```bash
# Оба демо (Calculator + Notepad)
python scripts/run_demo.py

# Только Calculator  (7 + 3 = 10)
python scripts/run_demo.py --scenario calc

# Только Notepad  (ввод → копирование → проверка)
python scripts/run_demo.py --scenario notepad

# Проверка без выполнения действий
python scripts/run_demo.py --dry-run

# Показать отладочные логи движка
python scripts/run_demo.py --verbose
```

Или воспользуйтесь стандартным CLI напрямую:

```bash
python -m screenwalker run scenarios/demo_calculator.yaml --config config/demo.yaml
python -m screenwalker run scenarios/demo_notepad.yaml   --config config/demo.yaml
```

---

### Что делают демо

#### Демо Calculator  (`scenarios/demo_calculator.yaml`)

1. Запускает `calc.exe` (Калькулятор Windows в стандартном режиме)
2. Ожидает появления интерфейса (OCR находит "Calculator")
3. Нажимает кнопку **7** через поиск текста по OCR
4. Нажимает кнопку **+** через поиск текста по OCR
5. Нажимает кнопку **3** через поиск текста по OCR
6. Нажимает кнопку **=** через поиск текста по OCR
7. Считывает результат с помощью OCR — проверяет, что он равен **"10"**
8. Сохраняет снимок экрана в `logs/demo/`

#### Демо Notepad  (`scenarios/demo_notepad.yaml`)

1. Запускает `notepad.exe`
2. Ожидает появления окна редактора (OCR находит "Notepad")
3. Вводит **"Hello from Screenwalker!"** в текстовую область
4. OCR проверяет, что текст виден на экране
5. Нажимает `Ctrl+A` для выделения всего текста
6. Копирует в буфер обмена через `Ctrl+C` — сохраняет в переменную контекста `clipboard_text`
7. OCR проверяет, что текст по-прежнему совпадает с ожидаемой строкой
8. Сохраняет снимок экрана
9. Закрывает приложение через `Alt+F4` и нажимает **"Don't Save"**

---

### Ожидаемый вывод

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

Логи и снимки экрана сохраняются в `logs/demo/`.

---

### Устранение неполадок

| Симптом | Решение |
|---------|-----|
| `tesseract: command not found` | Установите Tesseract и добавьте его в PATH |
| `ModuleNotFoundError: pytesseract` | `pip install pytesseract` |
| `ElementNotFound: "Calculator"` | Убедитесь, что Калькулятор открыт в стандартном режиме; попробуйте `--verbose` |
| OCR находит не ту "7" (например, в строке заголовка) | Сначала откройте Калькулятор, чтобы экран был чистым; или добавьте `region` в спецификацию поиска |
| Калькулятор открывается в режиме Журнала/Инженерном | Вручную переключитесь в стандартный режим или передайте переопределение через `--var` |
| Кнопка "Don't Save" в Notepad не найдена | Windows 10: попробуйте `query: "Don't Save"` — в Windows 11 текст кнопки может отличаться |
| `pyautogui.FailSafeException` | Уберите мышь из верхнего левого угла; или установите `pyautogui.FAILSAFE = False` |
| `import pyautogui` завершается ошибкой на Linux | `sudo apt install python3-tk python3-dev` |

---

### Адаптация демо

#### macOS

Измените `target` в обоих сценариях:

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

## Создание собственных сценариев

Сценарий — это YAML-файл. Минимальный пример:

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

### Доступные действия

| Действие | Что делает | Обязательные поля |
|--------|-------------|-----------------|
| `launch` | Запускает приложение | `target` |
| `click` | Левый клик по найденному элементу | `find` |
| `double_click` | Двойной клик | `find` |
| `right_click` | Правый клик | `find` |
| `type` | Вводит текст | `text` |
| `hotkey` | Комбинация клавиш | `keys` |
| `scroll` | Прокрутка у элемента | `find` (необязательно), `extra.direction`, `extra.clicks` |
| `drag` | Перетаскивание в позицию | `find`, `extra.to` |
| `copy` | Ctrl+C → буфер обмена | — |
| `paste` | Ctrl+V | — |
| `assert_visible` | Проверяет наличие элемента на экране | `find` |
| `assert_text` | Проверяет совпадение текста по OCR | `find`, `text` |
| `wait` | Пауза | `wait` (секунды) |
| `screenshot` | Сохраняет снимок экрана | `label` (необязательно) |

### Методы поиска

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

### Переменные

Переменные определяются в секции `variables:` и подставляются
с помощью синтаксиса `{{ var_name }}`:

```yaml
variables:
  username: admin

steps:
  - id: type_user
    action: type
    text: "{{ username }}"
```

Переопределение через командную строку:
```bash
python -m screenwalker run scenario.yaml --var username=myuser
```

### Отпечатки экрана

Проверяйте, что интерфейс находится в нужном состоянии, перед выполнением действий:

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

## Справочник CLI

```bash
# Запуск сценария
python -m screenwalker run SCENARIO [--config CONFIG] [--dry-run] [--var KEY=VALUE ...]

# Проверка без запуска
python -m screenwalker validate SCENARIO

# Захват шаблонного изображения
python -m screenwalker capture-template --name ok_button --region 100,200,80,30 --delay 3

# Анализ логов выполнения
python -m screenwalker analyze logs/

# Предложить синонимы для ошибок OCR
python -m screenwalker suggest-synonyms logs/

# Версия
python -m screenwalker --version
```

---

## Структура проекта

```
screenwalker/
├── config/
│   ├── default.yaml          ← конфигурация по умолчанию
│   ├── demo.yaml             ← конфигурация, оптимизированная для демо
│   └── synonyms.yaml         ← группы синонимов OCR (33 группы, en+ru)
├── scenarios/
│   ├── example_scenario.yaml ← пример автоматизации Notepad
│   ├── demo_calculator.yaml  ← демо Calculator (7+3=10)
│   └── demo_notepad.yaml     ← демо Notepad с буфером обмена
├── scripts/
│   ├── run_demo.py           ← красивый запускатель демо
│   ├── download_model.py     ← загрузка весов YOLO OmniParser V2
│   └── train_detector.py     ← дообучение YOLO на пользовательских скриншотах
├── templates/                ← PNG-шаблоны для сопоставления шаблонов
├── screenwalker/
│   ├── core/                 ← ScenarioEngine, Step, RunContext, ошибки
│   ├── vision/               ← OCR, сопоставление шаблонов, YOLO-детектор, состояние экрана
│   ├── actions/              ← контроллеры мыши, клавиатуры и буфера обмена
│   ├── learning/             ← кэш действий, логгер шагов, анализатор паттернов
│   ├── matching/             ← нечёткое сопоставление, реестр синонимов
│   └── utils/                ← конфигурация, повторные попытки
└── tests/                    ← 464 теста, покрытие 81%
```

---

## Обзор архитектуры

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

## Дополнительно: YOLO-детекция

Для более широкого обнаружения элементов интерфейса (иконки, чекбоксы, слайдеры без видимого текста):

```bash
# Загрузка весов OmniParser V2 (~25 МБ)
python scripts/download_model.py

# Включение в config/demo.yaml:
# vision:
#   yolo_enabled: true
#   yolo_model_path: models/icon_detect/best.pt
#   yolo_confidence: 0.50
```

Затем используйте `method: yolo` в спецификациях поиска или воспользуйтесь автоматическим
резервным переходом на YOLO, который срабатывает, когда OCR и сопоставление шаблонов не дают результата.

---

## Запуск тестов

```bash
pip install -e ".[dev]"
pytest                        # все 464 теста
pytest tests/test_detector.py # только тесты YOLO-детектора
pytest --tb=short -q          # компактный вывод с отчётом о покрытии
```
