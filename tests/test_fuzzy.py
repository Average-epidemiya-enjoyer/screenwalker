"""Tests for the fuzzy matching and synonym registry modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from screenwalker.matching.fuzzy import FuzzyMatcher, normalise
from screenwalker.matching.synonyms import SynonymRegistry


# ---------------------------------------------------------------------------
# normalise()
# ---------------------------------------------------------------------------


class TestNormalise:
    """Unit tests for the string normalisation helper."""

    def test_lowercases_text(self) -> None:
        """Text is converted to lowercase by default."""
        assert normalise("HELLO WORLD") == "hello world"

    def test_collapses_whitespace(self) -> None:
        """Multiple whitespace characters are collapsed to single spaces."""
        assert normalise("hello   \t world") == "hello world"

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert normalise("  hello  ") == "hello"

    def test_preserves_case_when_disabled(self) -> None:
        """lowercase=False preserves original case."""
        result = normalise("Hello World", lowercase=False)
        assert result == "Hello World"

    def test_preserves_whitespace_when_disabled(self) -> None:
        """strip_whitespace=False preserves whitespace."""
        result = normalise("  hello  ", strip_whitespace=False)
        assert result == "  hello  "

    def test_empty_string(self) -> None:
        """Empty string normalises to empty string."""
        assert normalise("") == ""

    def test_unicode_nfc_normalisation(self) -> None:
        """Unicode NFC normalisation is applied."""
        # Decomposed 'é' (e + combining accent) should become composed 'é'
        decomposed = "e\u0301"  # e + combining acute accent
        result = normalise(decomposed)
        assert result == "\xe9"  # é (precomposed)


# ---------------------------------------------------------------------------
# FuzzyMatcher stubs
# ---------------------------------------------------------------------------


class TestFuzzyMatcher:
    """Tests for FuzzyMatcher — stubs raise NotImplementedError."""

    def test_score_raises_not_implemented(self) -> None:
        """score() raises NotImplementedError (stub)."""
        matcher = FuzzyMatcher()
        with pytest.raises(NotImplementedError):
            matcher.score("ok", "okay")

    def test_best_match_raises_not_implemented(self) -> None:
        """best_match() raises NotImplementedError (stub)."""
        matcher = FuzzyMatcher()
        with pytest.raises(NotImplementedError):
            matcher.best_match("Submit", ["Submit Order", "Cancel", "Apply"])

    # -----------------------------------------------------------------------
    # Contract tests
    # -----------------------------------------------------------------------

    @pytest.mark.skip(reason="Requires rapidfuzz implementation")
    def test_score_identical_strings_returns_one(self) -> None:
        """Identical strings score 1.0."""
        matcher = FuzzyMatcher()
        assert matcher.score("submit", "submit") == pytest.approx(1.0, abs=0.01)

    @pytest.mark.skip(reason="Requires rapidfuzz implementation")
    def test_score_completely_different_strings_below_threshold(self) -> None:
        """Completely different strings score below 0.5."""
        matcher = FuzzyMatcher()
        score = matcher.score("submit", "xyz123abc")
        assert score < 0.5

    @pytest.mark.skip(reason="Requires rapidfuzz implementation")
    def test_best_match_returns_highest_scoring_candidate(self) -> None:
        """best_match returns the closest candidate."""
        matcher = FuzzyMatcher(threshold=0.5)
        result = matcher.best_match("Submit", ["Submit Order", "Cancel", "Close"])
        assert result is not None
        candidate, score = result
        assert "Submit" in candidate

    @pytest.mark.skip(reason="Requires rapidfuzz implementation")
    def test_best_match_returns_none_below_threshold(self) -> None:
        """best_match returns None when no candidate exceeds threshold."""
        matcher = FuzzyMatcher(threshold=0.99)
        result = matcher.best_match("Submit", ["xyz", "abc", "123"])
        assert result is None

    @pytest.mark.skip(reason="Requires rapidfuzz implementation")
    def test_all_matches_returns_sorted_results(self) -> None:
        """all_matches returns list sorted by score descending."""
        matcher = FuzzyMatcher(threshold=0.3)
        results = matcher.all_matches("ok", ["okay", "ok!", "cancel", "o"])
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# SynonymRegistry
# ---------------------------------------------------------------------------


class TestSynonymRegistry:
    """Unit tests for SynonymRegistry."""

    def test_default_synonyms_loaded(self) -> None:
        """Registry is pre-loaded with built-in synonyms."""
        reg = SynonymRegistry(load_defaults=True)
        assert "ok" in reg._canonical_to_aliases
        assert "cancel" in reg._canonical_to_aliases

    def test_resolve_known_alias(self) -> None:
        """resolve returns canonical name for a known alias."""
        reg = SynonymRegistry()
        assert reg.resolve("confirm") == "ok"
        assert reg.resolve("dismiss") == "cancel"

    def test_resolve_unknown_returns_none(self) -> None:
        """resolve returns None for unregistered text."""
        reg = SynonymRegistry()
        assert reg.resolve("xyzzy_unknown") is None

    def test_resolve_case_insensitive(self) -> None:
        """resolve is case-insensitive."""
        reg = SynonymRegistry()
        assert reg.resolve("CONFIRM") == "ok"

    def test_expand_returns_all_aliases(self) -> None:
        """expand returns the full alias set for a canonical label."""
        reg = SynonymRegistry()
        aliases = reg.expand("ok")
        assert "confirm" in aliases
        assert "okay" in aliases

    def test_expand_unknown_returns_empty_set(self) -> None:
        """expand returns an empty set for unknown canonical."""
        reg = SynonymRegistry()
        assert reg.expand("nonexistent_canonical") == set()

    def test_add_group_custom_synonyms(self) -> None:
        """add_group registers custom synonyms correctly."""
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("submit_order", ["Submit Order", "Place Order", "Buy Now"])
        assert reg.resolve("buy now") == "submit_order"
        assert "place order" in reg.expand("submit_order")

    def test_all_aliases_resolves_and_expands(self) -> None:
        """all_aliases returns the full synonym set for any member."""
        reg = SynonymRegistry()
        # 'confirm' is an alias for 'ok', so all_aliases should return the full ok group
        result = reg.all_aliases("confirm")
        assert "ok" in result
        assert "okay" in result

    def test_all_aliases_unknown_returns_singleton(self) -> None:
        """all_aliases returns {text} for unknown input."""
        reg = SynonymRegistry()
        result = reg.all_aliases("xyz_unknown_label")
        assert result == {"xyz_unknown_label"}

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """load_from_yaml merges synonyms from a YAML file."""
        yaml_content = "checkout:\n  - Checkout\n  - Proceed to Payment\n  - Buy\n"
        yaml_path = tmp_path / "synonyms.yaml"
        yaml_path.write_text(yaml_content)

        reg = SynonymRegistry(load_defaults=False)
        reg.load_from_yaml(yaml_path)

        assert reg.resolve("buy") == "checkout"
        assert "checkout" in reg.expand("checkout")

    def test_load_from_yaml_missing_file_raises(self) -> None:
        """load_from_yaml raises FileNotFoundError for missing file."""
        reg = SynonymRegistry(load_defaults=False)
        with pytest.raises(FileNotFoundError):
            reg.load_from_yaml("/nonexistent/synonyms.yaml")

    def test_save_and_reload_yaml(self, tmp_path: Path) -> None:
        """save_to_yaml produces a file that can be loaded back."""
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("greet", ["hello", "hi", "hey"])

        path = tmp_path / "out.yaml"
        reg.save_to_yaml(path)
        assert path.exists()

        reg2 = SynonymRegistry(load_defaults=False)
        reg2.load_from_yaml(path)
        assert reg2.resolve("hi") == "greet"

    def test_no_defaults_flag(self) -> None:
        """load_defaults=False creates an empty registry."""
        reg = SynonymRegistry(load_defaults=False)
        assert len(reg._canonical_to_aliases) == 0
