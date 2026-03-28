"""Synonym / alias dictionary for UI element labels.

Two complementary classes are provided:

* :class:`SynonymRegistry` — the original registry with canonical → alias
  resolution.  Used throughout the vision pipeline.

* :class:`SynonymDictionary` — higher-level class designed for YAML-driven
  configuration.  Wraps a ``SynonymRegistry`` and adds fuzzy group resolution,
  online synonym addition, and round-trip YAML persistence.

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
# Built-in synonym groups covering common UI patterns
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
# SynonymRegistry — original low-level registry
# ---------------------------------------------------------------------------


class SynonymRegistry:
    """Registry mapping canonical UI labels to synonym sets.

    Attributes:
        _canonical_to_aliases: Maps canonical label → set of all aliases.
        _alias_to_canonical: Reverse map for fast resolution.
    """

    def __init__(self, load_defaults: bool = True) -> None:
        """Initialize SynonymRegistry.

        Args:
            load_defaults: Pre-load the built-in :data:`_DEFAULT_SYNONYMS`.
        """
        self._canonical_to_aliases: dict[str, set[str]] = {}
        self._alias_to_canonical: dict[str, str] = {}

        if load_defaults:
            for canonical, aliases in _DEFAULT_SYNONYMS.items():
                self.add_group(canonical, aliases)

    def add_group(self, canonical: str, aliases: list[str]) -> None:
        """Register a group of synonyms under a canonical name.

        Aliases are normalised to lowercase and stripped before storage.

        Args:
            canonical: The primary label used internally.
            aliases: All forms that should resolve to *canonical*.
        """
        norm_canonical = canonical.lower().strip()
        self._canonical_to_aliases.setdefault(norm_canonical, set())
        for alias in aliases:
            norm = alias.lower().strip()
            self._canonical_to_aliases[norm_canonical].add(norm)
            self._alias_to_canonical[norm] = norm_canonical

    def resolve(self, text: str) -> str | None:
        """Resolve *text* to its canonical label.

        Args:
            text: Raw string (e.g. from OCR output or YAML).

        Returns:
            Canonical label if *text* (normalised) is a known alias, else None.
        """
        return self._alias_to_canonical.get(text.lower().strip())

    def expand(self, canonical: str) -> set[str]:
        """Return all known aliases for a canonical label.

        Args:
            canonical: Canonical label to expand.

        Returns:
            Set of all alias strings (including the canonical itself),
            or an empty set if *canonical* is not registered.
        """
        return self._canonical_to_aliases.get(canonical.lower().strip(), set())

    def all_aliases(self, text: str) -> set[str]:
        """Return all synonyms for the canonical label that *text* maps to.

        Convenience wrapper: resolve then expand.

        Args:
            text: Any known alias or canonical label.

        Returns:
            Full synonym set, or a set containing just *text* if unknown.
        """
        canonical = self.resolve(text)
        if canonical:
            return self.expand(canonical)
        return {text.lower().strip()}

    def load_from_yaml(self, path: Path | str) -> None:
        """Merge additional synonyms from a YAML file.

        File format::

            ok:
              - ok
              - okay
              - confirm
            my_custom_button:
              - Submit Order
              - Place Order

        Args:
            path: Path to the YAML synonyms file.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the YAML structure is invalid.
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
        """Persist the current registry to a YAML file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: sorted(v) for k, v in self._canonical_to_aliases.items()}
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, default_flow_style=False)


# ---------------------------------------------------------------------------
# SynonymDictionary — higher-level YAML-first API
# ---------------------------------------------------------------------------


class SynonymDictionary:
    """YAML-driven synonym dictionary with fuzzy group resolution.

    Wraps a :class:`SynonymRegistry` and adds:

    * :meth:`resolve` — returns the full list of synonyms for *text*.
    * :meth:`find_group` — fuzzy-matches *text* to the closest synonym group.
    * :meth:`add_synonym` — adds a new alias at runtime (useful for learning).
    * :meth:`save` — persists changes back to the source YAML file.

    YAML format (``synonyms:`` top-level key)::

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

    The ``synonyms:`` wrapper key is optional — a bare mapping is also accepted
    (compatible with :meth:`SynonymRegistry.load_from_yaml`).

    Args:
        dict_path: Path to a YAML synonyms file.  When ``None`` the dictionary
            starts empty.
    """

    def __init__(self, dict_path: Path | str | None = None) -> None:
        """Initialize SynonymDictionary.

        Args:
            dict_path: Optional path to a YAML synonyms file to load.
        """
        self._registry = SynonymRegistry(load_defaults=False)
        self._dict_path: Path | None = Path(dict_path) if dict_path else None
        if self._dict_path is not None and self._dict_path.exists():
            self._load(self._dict_path)

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Load synonyms from *path*, supporting both YAML formats.

        Args:
            path: YAML file to read.
        """
        with path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ValueError(f"Synonyms YAML must be a top-level mapping: {path}")

        # Support `synonyms:` wrapper key
        if "synonyms" in raw and isinstance(raw["synonyms"], dict):
            groups = raw["synonyms"]
        else:
            groups = raw

        for group_name, aliases in groups.items():
            if isinstance(aliases, list):
                self._registry.add_group(str(group_name), [str(a) for a in aliases])

    def save(self) -> None:
        """Persist the current dictionary to :attr:`dict_path`.

        Raises:
            RuntimeError: If no *dict_path* was provided at construction time.
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
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, text: str) -> list[str]:
        """Return all known synonyms for *text*, sorted alphabetically.

        If *text* belongs to a synonym group, the entire group is returned.
        Otherwise ``[text.lower().strip()]`` is returned (singleton).

        Args:
            text: Any known alias or raw OCR text.

        Returns:
            Sorted list of synonym strings.
        """
        return sorted(self._registry.all_aliases(text))

    def find_group(self, text: str, fuzzy_threshold: int = 80) -> str | None:
        """Find which synonym group *text* belongs to, using fuzzy matching.

        Performs a fuzzy search over *all* aliases across all groups and
        returns the canonical group name of the best match above
        *fuzzy_threshold*.

        Args:
            text: Text to classify (e.g. an OCR result).
            fuzzy_threshold: Minimum ``token_sort_ratio`` score 0–100.

        Returns:
            Canonical group name, or ``None`` if no group matches well enough.
        """
        # Fast path — exact (case-insensitive) lookup
        exact = self._registry.resolve(text)
        if exact is not None:
            return exact

        # Fuzzy path — compare against all aliases across all groups
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
        """Add a new alias to *group*, creating the group if necessary.

        Args:
            group: Canonical group name (e.g. ``"close"``).
            new_synonym: New alias string to register (e.g. ``"schließen"``).
        """
        self._registry.add_group(group, [new_synonym])

    def groups(self) -> list[str]:
        """Return all registered canonical group names.

        Returns:
            Sorted list of group names.
        """
        return sorted(self._registry._canonical_to_aliases.keys())

    def all_aliases_flat(self) -> dict[str, list[str]]:
        """Return all groups and their aliases as a plain dict.

        Returns:
            Mapping of canonical → sorted alias list.
        """
        return {k: sorted(v) for k, v in self._registry._canonical_to_aliases.items()}
