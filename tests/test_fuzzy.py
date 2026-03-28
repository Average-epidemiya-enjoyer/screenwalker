"""Tests for the fuzzy matching and synonym modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from screenwalker.matching.fuzzy import FuzzyMatcher, FuzzyResult, fuzzy_match, fuzzy_score, normalise
from screenwalker.matching.synonyms import SynonymDictionary, SynonymRegistry


# ---------------------------------------------------------------------------
# normalise()
# ---------------------------------------------------------------------------


class TestNormalise:
    """Unit tests for the string normalisation helper."""

    def test_lowercases_text(self) -> None:
        assert normalise("HELLO WORLD") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert normalise("hello   \t world") == "hello world"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert normalise("  hello  ") == "hello"

    def test_preserves_case_when_disabled(self) -> None:
        result = normalise("Hello World", lowercase=False)
        assert result == "Hello World"

    def test_preserves_whitespace_when_disabled(self) -> None:
        result = normalise("  hello  ", strip_whitespace=False)
        assert result == "  hello  "

    def test_empty_string(self) -> None:
        assert normalise("") == ""

    def test_unicode_nfc_normalisation(self) -> None:
        decomposed = "e\u0301"  # e + combining acute accent
        assert normalise(decomposed) == "\xe9"


# ---------------------------------------------------------------------------
# Module-level helpers: fuzzy_score / fuzzy_match
# ---------------------------------------------------------------------------


class TestFuzzyScore:
    def test_identical_returns_one(self) -> None:
        assert fuzzy_score("submit", "submit") == pytest.approx(1.0, abs=0.01)

    def test_completely_different_below_half(self) -> None:
        assert fuzzy_score("submit", "xyz123abc") < 0.5

    def test_returns_float_in_range(self) -> None:
        s = fuzzy_score("hello", "world")
        assert 0.0 <= s <= 1.0

    def test_no_normalise(self) -> None:
        # Identical strings score 1.0 even without normalisation
        s = fuzzy_score("hello world", "hello world", normalise_inputs=False)
        assert s == pytest.approx(1.0, abs=0.01)


class TestFuzzyMatch:
    def test_returns_matching_candidates(self) -> None:
        results = fuzzy_match("submit", ["Submit", "Cancel", "OK"], threshold=0.7)
        texts = [t for t, _ in results]
        assert "Submit" in texts

    def test_excludes_below_threshold(self) -> None:
        results = fuzzy_match("submit", ["xyz", "abc", "123"], threshold=0.9)
        assert results == []

    def test_sorted_descending(self) -> None:
        results = fuzzy_match("ok", ["ok", "okay", "cancel", "ок"], threshold=0.3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self) -> None:
        assert fuzzy_match("query", []) == []

    def test_returns_float_scores(self) -> None:
        results = fuzzy_match("ok", ["ok", "okay"])
        for _, score in results:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# FuzzyMatcher — new-style methods (int scores)
# ---------------------------------------------------------------------------


class TestFuzzyMatcherMatch:
    def test_returns_fuzzy_result_list(self) -> None:
        matcher = FuzzyMatcher(threshold=60)
        results = matcher.match("submit", ["Submit", "Cancel", "OK"])
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, FuzzyResult)

    def test_match_finds_close_candidate(self) -> None:
        matcher = FuzzyMatcher(threshold=50)
        results = matcher.match("Submit", ["Submit", "Cancel", "Close"])
        texts = [r.text for r in results]
        assert "Submit" in texts

    def test_match_excludes_below_threshold(self) -> None:
        matcher = FuzzyMatcher(threshold=90)
        results = matcher.match("submit", ["xyz", "abc", "123"])
        assert results == []

    def test_match_sorted_descending(self) -> None:
        matcher = FuzzyMatcher(threshold=30)
        results = matcher.match("ok", ["ok", "okay", "cancel"])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_match_original_index_correct(self) -> None:
        matcher = FuzzyMatcher(threshold=50)
        candidates = ["Cancel", "Submit", "OK"]
        results = matcher.match("Submit", candidates)
        for r in results:
            assert candidates[r.original_index] == r.text

    def test_match_empty_candidates(self) -> None:
        matcher = FuzzyMatcher()
        assert matcher.match("query", []) == []

    def test_match_override_threshold(self) -> None:
        matcher = FuzzyMatcher(threshold=99)
        # Using per-call override
        results = matcher.match("ok", ["okay"], threshold=50)
        assert len(results) >= 1


class TestFuzzyMatcherBestMatch:
    def test_returns_best_fuzzy_result(self) -> None:
        matcher = FuzzyMatcher(threshold=50)
        result = matcher.best_match("Submit", ["Submit Order", "Cancel", "Close"])
        assert result is not None
        assert isinstance(result, FuzzyResult)
        assert "Submit" in result.text

    def test_returns_none_below_threshold(self) -> None:
        matcher = FuzzyMatcher(threshold=99)
        result = matcher.best_match("Submit", ["xyz", "abc", "123"])
        assert result is None

    def test_empty_candidates_returns_none(self) -> None:
        matcher = FuzzyMatcher()
        assert matcher.best_match("query", []) is None

    def test_score_in_valid_range(self) -> None:
        matcher = FuzzyMatcher(threshold=50)
        result = matcher.best_match("ok", ["ok", "okay", "cancel"])
        assert result is not None
        assert 0 <= result.score <= 100


class TestFuzzyMatcherContainsFuzzy:
    def test_exact_substring_found(self) -> None:
        matcher = FuzzyMatcher()
        assert matcher.contains_fuzzy("Save File As", "Save", threshold=80)

    def test_fuzzy_substring_found(self) -> None:
        matcher = FuzzyMatcher()
        assert matcher.contains_fuzzy("Saave File", "Save", threshold=70)

    def test_unrelated_string_not_found(self) -> None:
        matcher = FuzzyMatcher()
        assert not matcher.contains_fuzzy("Open Document", "xyz123", threshold=80)

    def test_empty_text_returns_false(self) -> None:
        matcher = FuzzyMatcher()
        assert not matcher.contains_fuzzy("", "save", threshold=50)


class TestFuzzyMatcherLegacy:
    """Legacy float-score methods (backward-compat)."""

    def test_score_identical_strings_returns_one(self) -> None:
        matcher = FuzzyMatcher()
        assert matcher.score("submit", "submit") == pytest.approx(1.0, abs=0.01)

    def test_score_different_strings_below_half(self) -> None:
        matcher = FuzzyMatcher()
        assert matcher.score("submit", "xyz123abc") < 0.5

    def test_score_returns_float(self) -> None:
        s = matcher = FuzzyMatcher()
        result = matcher.score("hello", "world")
        assert isinstance(result, float)

    def test_all_matches_sorted_descending(self) -> None:
        matcher = FuzzyMatcher(threshold=30)
        results = matcher.all_matches("ok", ["okay", "ok!", "cancel", "o"])
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_matches_threshold_override(self) -> None:
        matcher = FuzzyMatcher(threshold=99)
        results = matcher.all_matches("ok", ["ok", "okay"], threshold=0.5)
        assert len(results) >= 1

    def test_all_matches_empty_returns_empty(self) -> None:
        matcher = FuzzyMatcher()
        assert matcher.all_matches("ok", []) == []

    def test_normalize_static(self) -> None:
        assert FuzzyMatcher.normalize("  HELLO  ") == "hello"


# ---------------------------------------------------------------------------
# SynonymRegistry
# ---------------------------------------------------------------------------


class TestSynonymRegistry:
    def test_default_synonyms_loaded(self) -> None:
        reg = SynonymRegistry(load_defaults=True)
        assert "ok" in reg._canonical_to_aliases
        assert "cancel" in reg._canonical_to_aliases

    def test_resolve_known_alias(self) -> None:
        reg = SynonymRegistry()
        assert reg.resolve("confirm") == "ok"
        assert reg.resolve("dismiss") == "cancel"

    def test_resolve_unknown_returns_none(self) -> None:
        reg = SynonymRegistry()
        assert reg.resolve("xyzzy_unknown") is None

    def test_resolve_case_insensitive(self) -> None:
        reg = SynonymRegistry()
        assert reg.resolve("CONFIRM") == "ok"

    def test_expand_returns_all_aliases(self) -> None:
        reg = SynonymRegistry()
        aliases = reg.expand("ok")
        assert "confirm" in aliases
        assert "okay" in aliases

    def test_expand_unknown_returns_empty_set(self) -> None:
        reg = SynonymRegistry()
        assert reg.expand("nonexistent_canonical") == set()

    def test_add_group_custom_synonyms(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("submit_order", ["Submit Order", "Place Order", "Buy Now"])
        assert reg.resolve("buy now") == "submit_order"
        assert "place order" in reg.expand("submit_order")

    def test_all_aliases_resolves_and_expands(self) -> None:
        reg = SynonymRegistry()
        result = reg.all_aliases("confirm")
        assert "ok" in result
        assert "okay" in result

    def test_all_aliases_unknown_returns_singleton(self) -> None:
        reg = SynonymRegistry()
        result = reg.all_aliases("xyz_unknown_label")
        assert result == {"xyz_unknown_label"}

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = "checkout:\n  - Checkout\n  - Proceed to Payment\n  - Buy\n"
        yaml_path = tmp_path / "synonyms.yaml"
        yaml_path.write_text(yaml_content)

        reg = SynonymRegistry(load_defaults=False)
        reg.load_from_yaml(yaml_path)

        assert reg.resolve("buy") == "checkout"
        assert "checkout" in reg.expand("checkout")

    def test_load_from_yaml_missing_file_raises(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        with pytest.raises(FileNotFoundError):
            reg.load_from_yaml("/nonexistent/synonyms.yaml")

    def test_save_and_reload_yaml(self, tmp_path: Path) -> None:
        reg = SynonymRegistry(load_defaults=False)
        reg.add_group("greet", ["hello", "hi", "hey"])

        path = tmp_path / "out.yaml"
        reg.save_to_yaml(path)
        assert path.exists()

        reg2 = SynonymRegistry(load_defaults=False)
        reg2.load_from_yaml(path)
        assert reg2.resolve("hi") == "greet"

    def test_no_defaults_flag(self) -> None:
        reg = SynonymRegistry(load_defaults=False)
        assert len(reg._canonical_to_aliases) == 0


# ---------------------------------------------------------------------------
# SynonymDictionary
# ---------------------------------------------------------------------------


class TestSynonymDictionary:
    def test_empty_when_no_path(self) -> None:
        d = SynonymDictionary()
        assert d.groups() == []

    def test_load_yaml_with_synonyms_wrapper(self, tmp_path: Path) -> None:
        content = "synonyms:\n  ok:\n    - ok\n    - confirm\n    - apply\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        assert "ok" in d.groups()
        assert "confirm" in d.resolve("ok")

    def test_load_yaml_bare_mapping(self, tmp_path: Path) -> None:
        content = "close:\n  - close\n  - cancel\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        assert "close" in d.groups()

    def test_resolve_returns_sorted_list(self, tmp_path: Path) -> None:
        content = "synonyms:\n  ok:\n    - ok\n    - confirm\n    - apply\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        result = d.resolve("confirm")
        assert result == sorted(result)
        assert "ok" in result
        assert "apply" in result

    def test_resolve_unknown_returns_singleton(self) -> None:
        d = SynonymDictionary()
        assert d.resolve("xyz_unknown") == ["xyz_unknown"]

    def test_find_group_exact_match(self, tmp_path: Path) -> None:
        content = "synonyms:\n  cancel:\n    - cancel\n    - dismiss\n    - abort\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        assert d.find_group("cancel") == "cancel"
        assert d.find_group("dismiss") == "cancel"

    def test_find_group_case_insensitive(self, tmp_path: Path) -> None:
        content = "synonyms:\n  ok:\n    - ok\n    - confirm\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        assert d.find_group("CONFIRM") == "ok"

    def test_find_group_fuzzy_match(self, tmp_path: Path) -> None:
        content = "synonyms:\n  save:\n    - save\n    - store\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        # "savee" should fuzzy-match "save"
        result = d.find_group("savee", fuzzy_threshold=70)
        assert result == "save"

    def test_find_group_returns_none_when_no_match(self, tmp_path: Path) -> None:
        content = "synonyms:\n  ok:\n    - ok\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        assert d.find_group("xyzzy_completely_unknown", fuzzy_threshold=90) is None

    def test_add_synonym_creates_group(self) -> None:
        d = SynonymDictionary()
        d.add_synonym("greet", "hello")
        assert "greet" in d.groups()
        assert "hello" in d.resolve("hello")

    def test_add_synonym_extends_existing_group(self, tmp_path: Path) -> None:
        content = "synonyms:\n  ok:\n    - ok\n    - confirm\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content)

        d = SynonymDictionary(path)
        d.add_synonym("ok", "jawohl")
        assert "jawohl" in d.resolve("ok")

    def test_save_round_trip(self, tmp_path: Path) -> None:
        d = SynonymDictionary()
        d.add_synonym("greet", "hello")
        d.add_synonym("greet", "hi")
        d._dict_path = tmp_path / "out.yaml"
        d.save()

        d2 = SynonymDictionary(tmp_path / "out.yaml")
        assert "greet" in d2.groups()
        assert "hello" in d2.resolve("hello")

    def test_save_without_path_raises(self) -> None:
        d = SynonymDictionary()
        with pytest.raises(RuntimeError):
            d.save()

    def test_groups_sorted(self, tmp_path: Path) -> None:
        content = "synonyms:\n  ok:\n    - ok\n  cancel:\n    - cancel\n  save:\n    - save\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content)

        d = SynonymDictionary(path)
        g = d.groups()
        assert g == sorted(g)

    def test_all_aliases_flat(self, tmp_path: Path) -> None:
        content = "synonyms:\n  ok:\n    - ok\n    - apply\n"
        path = tmp_path / "synonyms.yaml"
        path.write_text(content, encoding="utf-8")

        d = SynonymDictionary(path)
        flat = d.all_aliases_flat()
        assert "ok" in flat
        assert flat["ok"] == sorted(flat["ok"])

    def test_loads_config_synonyms_yaml(self) -> None:
        """The shipped config/synonyms.yaml loads without errors."""
        config_path = Path(__file__).parent.parent / "config" / "synonyms.yaml"
        if not config_path.exists():
            pytest.skip("config/synonyms.yaml not found")

        d = SynonymDictionary(config_path)
        assert len(d.groups()) >= 10
        assert "ok" in d.groups()
        assert "cancel" in d.groups()

    def test_config_synonyms_russian_aliases(self) -> None:
        """Russian aliases from config/synonyms.yaml resolve correctly."""
        config_path = Path(__file__).parent.parent / "config" / "synonyms.yaml"
        if not config_path.exists():
            pytest.skip("config/synonyms.yaml not found")

        d = SynonymDictionary(config_path)
        assert d.find_group("подтвердить") == "ok"
        assert d.find_group("отмена") == "cancel"
        assert d.find_group("сохранить") == "save"
