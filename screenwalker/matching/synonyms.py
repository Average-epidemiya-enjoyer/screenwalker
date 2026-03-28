"""Synonym / alias dictionary for UI element labels.

Many UI frameworks label the same concept differently across locales, themes,
or application versions.  The :class:`SynonymRegistry` maps canonical names
to sets of known aliases so the OCR finder can match any of them.

Example:
    >>> reg = SynonymRegistry()
    >>> reg.add_group("ok", ["ok", "okay", "yes", "confirm", "apply"])
    >>> reg.resolve("Confirm")
    'ok'
    >>> reg.expand("ok")
    {'ok', 'okay', 'yes', 'confirm', 'apply'}
"""

from __future__ import annotations

from pathlib import Path

import yaml


# Built-in synonym groups covering common UI patterns
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
