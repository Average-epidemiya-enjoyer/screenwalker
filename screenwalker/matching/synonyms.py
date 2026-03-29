"""Словарь синонимов / псевдонимов для меток элементов UI.

Предоставляются два взаимодополняющих класса:

* :class:`SynonymRegistry` — оригинальный реестр с разрешением canonical → alias.
  Используется повсеместно в пайплайне vision.

* :class:`SynonymDictionary` — высокоуровневый класс, ориентированный на
  YAML-конфигурацию. Оборачивает ``SynonymRegistry`` и добавляет нечёткое
  разрешение групп, добавление синонимов в режиме онлайн и round-trip YAML-персистентность.

Example:
    >>> reg = SynonymRegistry()
    >>> reg.add_group("ok", ["ok", "okay", "yes", "confirm", "apply"])
    >>> reg.resolve("Confirm")
    'ok'
    >>> reg.expand("ok")
    {'ok', 'okay', 'yes', 'confirm', 'apply'}

    >>> d = SynonymDictionary()
    >>> d.resolve("Confirm")
    ['accept', 'apply', 'confirm', 'ok', 'okay', 'sure', 'yes']
    >>> d.find_group("закрыть")
    'cancel'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Встроенные группы синонимов для распространённых паттернов UI
# ---------------------------------------------------------------------------

_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "ok": ["ok", "okay", "yes", "confirm", "apply", "accept", "sure"],
    "cancel": ["cancel", "no", "dismiss", "close", "abort", "decline"],
    "save": ["save", "save file", "save as", "write", "export"],
    "open": ["open", "load", "import", "browse", "choose file"],
    "next": ["next", "continue", "proceed", "forward", ">", ">>"],
    "back": ["back", "previous", "prev", "<", "<<"],
    "finish": ["finish", "done", "complete", "submit"],
    "search": ["search", "find", "lookup", "query", "filter"],
    "delete": ["delete", "remove", "erase", "clear", "discard"],
    "edit": ["edit", "modify", "change", "update", "rename"],
    "new": ["new", "create", "add", "insert", "+"],
    "settings": ["settings", "preferences", "options", "configuration", "config"],
    "help": ["help", "?", "support", "documentation", "about"],
    "home": ["home", "start", "main", "dashboard", "overview"],
    "logout": ["logout", "log out", "sign out", "signout", "exit"],
    "login": ["login", "log in", "sign in", "signin", "authenticate"],
}


# ---------------------------------------------------------------------------
# SynonymRegistry — оригинальный низкоуровневый реестр
# ---------------------------------------------------------------------------


class SynonymRegistry:
    """Реестр, отображающий канонические метки UI на наборы синонимов.

    Attributes:
        _canonical_to_aliases: Отображает каноническую метку → множество всех псевдонимов.
        _alias_to_canonical: Обратное отображение для быстрого разрешения.
    """

    def __init__(self, load_defaults: bool = True) -> None:
        """Инициализировать SynonymRegistry.

        Args:
            load_defaults: Предварительно загрузить встроенные :data:`_DEFAULT_SYNONYMS`.
        """
        self._canonical_to_aliases: dict[str, set[str]] = {}
        self._alias_to_canonical: dict[str, str] = {}

        if load_defaults:
            for canonical, aliases in _DEFAULT_SYNONYMS.items():
                self.add_group(canonical, aliases)

    def add_group(self, canonical: str, aliases: list[str]) -> None:
        """Зарегистрировать группу синонимов под каноническим именем.

        Псевдонимы нормализуются до нижнего регистра и очищаются от пробелов перед сохранением.

        Args:
            canonical: Основная метка, используемая внутри системы.
            aliases: Все формы, которые должны разрешаться в *canonical*.
        """
        norm_canonical = canonical.lower().strip()
        self._canonical_to_aliases.setdefault(norm_canonical, set())
        for alias in aliases:
            norm = alias.lower().strip()
            self._canonical_to_aliases[norm_canonical].add(norm)
            self._alias_to_canonical[norm] = norm_canonical

    def resolve(self, text: str) -> str | None:
        """Разрешить *text* в его каноническую метку.

        Args:
            text: Необработанная строка (например, из OCR-вывода или YAML).

        Returns:
            Каноническая метка, если *text* (нормализованный) является известным
            псевдонимом, иначе None.
        """
        return self._alias_to_canonical.get(text.lower().strip())

    def expand(self, canonical: str) -> set[str]:
        """Вернуть все известные псевдонимы для канонической метки.

        Args:
            canonical: Каноническая метка для расширения.

        Returns:
            Множество всех строк-псевдонимов (включая саму каноническую),
            или пустое множество, если *canonical* не зарегистрирован.
        """
        return self._canonical_to_aliases.get(canonical.lower().strip(), set())

    def all_aliases(self, text: str) -> set[str]:
        """Вернуть все синонимы канонической метки, на которую отображается *text*.

        Удобная обёртка: resolve, затем expand.

        Args:
            text: Любой известный псевдоним или каноническая метка.

        Returns:
            Полное множество синонимов, или множество, содержащее только *text*, если он неизвестен.
        """
        canonical = self.resolve(text)
        if canonical:
            return self.expand(canonical)
        return {text.lower().strip()}

    def load_from_yaml(self, path: Path | str) -> None:
        """Добавить дополнительные синонимы из YAML-файла.

        Формат файла::

            ok:
              - ok
              - okay
              - confirm
            my_custom_button:
              - Submit Order
              - Place Order

        Args:
            path: Путь к YAML-файлу синонимов.

        Raises:
            FileNotFoundError: Если *path* не существует.
            ValueError: Если структура YAML невалидна.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Synonyms file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError("Synonyms YAML must be a top-level mapping")
        for canonical, aliases in data.items():
            if not isinstance(aliases, list):
                raise ValueError(f"Aliases for {canonical!r} must be a list, got {type(aliases)}")
            self.add_group(str(canonical), [str(a) for a in aliases])

    def save_to_yaml(self, path: Path | str) -> None:
        """Сохранить текущий реестр в YAML-файл.

        Args:
            path: Путь к файлу назначения.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: sorted(v) for k, v in self._canonical_to_aliases.items()}
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, default_flow_style=False)


# ---------------------------------------------------------------------------
# SynonymDictionary — высокоуровневый YAML-ориентированный API
# ---------------------------------------------------------------------------


class SynonymDictionary:
    """YAML-ориентированный словарь синонимов с нечётким разрешением групп.

    Оборачивает :class:`SynonymRegistry` и добавляет:

    * :meth:`resolve` — возвращает полный список синонимов для *text*.
    * :meth:`find_group` — нечётко сопоставляет *text* с ближайшей группой синонимов.
    * :meth:`add_synonym` — добавляет новый псевдоним во время выполнения (полезно для обучения).
    * :meth:`save` — сохраняет изменения обратно в исходный YAML-файл.

    Формат YAML (ключ верхнего уровня ``synonyms:``)::

        synonyms:
          close:
            - закрыть
            - close
            - cancel
            - ×
          save:
            - сохранить
            - save
            - apply

    Ключ-обёртка ``synonyms:`` является необязательным — принимается и простое
    отображение (совместимо с :meth:`SynonymRegistry.load_from_yaml`).

    Args:
        dict_path: Путь к YAML-файлу синонимов. При значении ``None`` словарь
            начинает пустым.
    """

    def __init__(self, dict_path: Path | str | None = None) -> None:
        """Инициализировать SynonymDictionary.

        Args:
            dict_path: Опциональный путь к YAML-файлу синонимов для загрузки.
        """
        self._registry = SynonymRegistry(load_defaults=False)
        self._dict_path: Path | None = Path(dict_path) if dict_path else None
        if self._dict_path is not None and self._dict_path.exists():
            self._load(self._dict_path)

    # ------------------------------------------------------------------
    # Загрузка / сохранение
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Загрузить синонимы из *path*, поддерживая оба формата YAML.

        Args:
            path: YAML-файл для чтения.
        """
        with path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ValueError(f"Synonyms YAML must be a top-level mapping: {path}")

        # Поддержка ключа-обёртки `synonyms:`
        if "synonyms" in raw and isinstance(raw["synonyms"], dict):
            groups = raw["synonyms"]
        else:
            groups = raw

        for group_name, aliases in groups.items():
            if isinstance(aliases, list):
                self._registry.add_group(str(group_name), [str(a) for a in aliases])

    def save(self) -> None:
        """Сохранить текущий словарь в :attr:`dict_path`.

        Raises:
            RuntimeError: Если *dict_path* не был указан при создании экземпляра.
        """
        if self._dict_path is None:
            raise RuntimeError("No dict_path set — provide a path to save to")
        self._dict_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "synonyms": {
                k: sorted(v)
                for k, v in self._registry._canonical_to_aliases.items()
            }
        }
        with self._dict_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, default_flow_style=False)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def resolve(self, text: str) -> list[str]:
        """Вернуть все известные синонимы для *text*, отсортированные по алфавиту.

        Если *text* принадлежит группе синонимов, возвращается вся группа.
        Иначе возвращается ``[text.lower().strip()]`` (одноэлементный список).

        Args:
            text: Любой известный псевдоним или необработанный OCR-текст.

        Returns:
            Отсортированный список строк-синонимов.
        """
        return sorted(self._registry.all_aliases(text))

    def find_group(self, text: str, fuzzy_threshold: int = 80) -> str | None:
        """Найти, к какой группе синонимов принадлежит *text*, используя нечёткое сопоставление.

        Выполняет нечёткий поиск по *всем* псевдонимам всех групп и возвращает
        каноническое имя группы с наилучшим совпадением выше *fuzzy_threshold*.

        Args:
            text: Текст для классификации (например, результат OCR).
            fuzzy_threshold: Минимальная оценка ``token_sort_ratio`` 0–100.

        Returns:
            Каноническое имя группы, или ``None``, если ни одна группа не совпала достаточно хорошо.
        """
        # Быстрый путь — точный поиск (без учёта регистра)
        exact = self._registry.resolve(text)
        if exact is not None:
            return exact

        # Нечёткий путь — сравнение со всеми псевдонимами всех групп
        from rapidfuzz import fuzz as _rfuzz

        norm_text = text.lower().strip()
        best_score = 0
        best_canonical: str | None = None

        for canonical, aliases in self._registry._canonical_to_aliases.items():
            for alias in aliases:
                score = int(_rfuzz.token_sort_ratio(norm_text, alias))
                if score > best_score:
                    best_score = score
                    best_canonical = canonical

        if best_score >= fuzzy_threshold:
            return best_canonical
        return None

    def add_synonym(self, group: str, new_synonym: str) -> None:
        """Добавить новый псевдоним в *group*, создав группу при необходимости.

        Args:
            group: Каноническое имя группы (например, ``"close"``).
            new_synonym: Новая строка псевдонима для регистрации (например, ``"schließen"``).
        """
        self._registry.add_group(group, [new_synonym])

    def groups(self) -> list[str]:
        """Вернуть все зарегистрированные канонические имена групп.

        Returns:
            Отсортированный список имён групп.
        """
        return sorted(self._registry._canonical_to_aliases.keys())

    def all_aliases_flat(self) -> dict[str, list[str]]:
        """Вернуть все группы и их псевдонимы в виде простого словаря.

        Returns:
            Отображение canonical → отсортированный список псевдонимов.
        """
        return {k: sorted(v) for k, v in self._registry._canonical_to_aliases.items()}
