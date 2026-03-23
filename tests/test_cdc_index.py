"""Tests for CDC ICD-9-CM and ICD-10-CM index parsers.

Validates that parsed lookup dictionaries are free of known bad patterns:
  - No stray punctuation artifacts in keys
  - No standalone generic/anatomical keys that cause false fuzzy matches
  - No generic see-ref pollution (anatomical terms -> unrelated codes)
  - Stored names are hierarchical for directly registered codes
  - Specific known-good lookups return correct codes
  - Specific known-bad lookups from previous bugs do NOT occur
"""

from collections import Counter

import pytest

from tctk.cdc.icd10cm_index import CDCIndex
from tctk.cdc.icd9cm_index import CDCIndex9


# ---------------------------------------------------------------------------
# Fixtures — parse once per session (expensive: ~2s each)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def icd10():
    return CDCIndex()


@pytest.fixture(scope="session")
def icd9():
    return CDCIndex9()


# ===================================================================
# ICD-10-CM tests
# ===================================================================

class TestICD10KeyQuality:
    """Keys should be clean normalized terms without artifacts."""

    def test_no_stray_closing_parens(self, icd10):
        """Stray ')' from nested nemod parentheses should be stripped."""
        bad = [k for k in icd10.keys if ")" in k]
        assert bad == [], (
            f"{len(bad)} keys contain stray ')'. "
            f"First 10: {bad[:10]}"
        )

    def test_no_stray_opening_parens(self, icd10):
        """Opening '(' should also be stripped by normalize."""
        bad = [k for k in icd10.keys if "(" in k]
        assert bad == [], (
            f"{len(bad)} keys contain '('. First 10: {bad[:10]}"
        )

    def test_no_curly_braces(self, icd10):
        """RTF/XML artifacts: no braces in keys."""
        bad = [k for k in icd10.keys if "{" in k or "}" in k]
        assert bad == [], f"Keys with braces: {bad[:10]}"

    def test_no_backslash(self, icd10):
        """No RTF escape sequences leaking through."""
        bad = [k for k in icd10.keys if "\\" in k]
        assert bad == [], f"Keys with backslash: {bad[:10]}"

    def test_no_empty_keys(self, icd10):
        """No empty-string keys."""
        assert "" not in icd10._code_lookup

    def test_no_whitespace_only_keys(self, icd10):
        bad = [k for k in icd10.keys if not k.strip()]
        assert bad == [], f"Whitespace-only keys found: {len(bad)}"

    def test_reasonable_key_count(self, icd10):
        """Sanity: should have 50K-100K keys."""
        n = len(icd10.keys)
        assert 50_000 < n < 100_000, f"Unexpected key count: {n}"


class TestICD10NoGenericKeys:
    """Standalone generic/anatomical terms must not appear as keys.

    These were previously registered via overly broad see-ref resolution
    (e.g. "Hip — see condition" polluting "hip" with U09.9).
    """

    # Generic terms that should NOT be standalone lookup keys.
    # They are anatomical sites or qualifiers, not conditions.
    FORBIDDEN_STANDALONE = [
        "hip", "chronic", "acute", "with", "simple",
        "upper", "lower", "left", "right", "bilateral",
        "old", "new", "other", "unspecified",
    ]

    @pytest.mark.parametrize("term", FORBIDDEN_STANDALONE)
    def test_no_standalone_generic_key(self, icd10, term):
        assert term not in icd10._code_lookup, (
            f"Standalone key '{term}' should not exist. "
            f"Entries: {icd10._code_lookup[term][:3]}"
        )


class TestICD10NoSeeRefPollution:
    """Generic navigational see-refs should not pollute the lookup."""

    # Anatomical terms that say "see condition/disease/disorder".
    # None of them should map to codes like U09.9, I27.848, etc.
    ANATOMICAL_REDIRECTS = [
        "alveolus, alveolar",
        "bronchi, bronchial",
        "femur, femoral",
        "larynx, laryngeal",
        "pylorus, pyloric",
        "rectum, rectal",
        "thymus, thymic",
        "vagina, vaginal",
    ]

    @pytest.mark.parametrize("term", ANATOMICAL_REDIRECTS)
    def test_anatomical_term_not_in_lookup(self, icd10, term):
        assert term not in icd10._code_lookup, (
            f"Anatomical redirect '{term}' should not have codes. "
            f"Entries: {icd10._code_lookup[term][:3]}"
        )

    def test_u09_9_not_over_registered(self, icd10):
        """U09.9 (post COVID-19) should appear under <20 keys, not 100+."""
        count = sum(
            1 for entries in icd10._code_lookup.values()
            for code, _ in entries if code == "U09.9"
        )
        assert count < 20, (
            f"U09.9 registered under {count} keys — likely see-ref pollution"
        )

    def test_no_single_code_with_excessive_keys(self, icd10):
        """No single ICD code should appear under >500 distinct keys.

        Some codes legitimately appear under many terms (e.g. F45.8
        under ~350 psychosomatic terms). But >500 would indicate
        unresolved generic-target pollution.
        """
        code_counts: Counter[str] = Counter()
        for entries in icd10._code_lookup.values():
            for code, _ in entries:
                code_counts[code] += 1
        worst_code, worst_count = code_counts.most_common(1)[0]
        assert worst_count <= 500, (
            f"Code {worst_code} appears under {worst_count} keys — "
            f"likely generic see-ref pollution"
        )


class TestICD10HierarchicalNames:
    """Directly registered names should reflect parent hierarchy."""

    def test_m05_30_names_contain_rheumatoid(self, icd10):
        """M05.30 = Rheumatoid carditis. All names must mention 'rheumatoid'."""
        for key, entries in icd10._code_lookup.items():
            for code, name in entries:
                if code == "M05.30":
                    assert "rheumatoid" in name.lower(), (
                        f"M05.30 stored with short name '{name}' "
                        f"under key '{key}' — missing parent hierarchy"
                    )

    def test_see_ref_resolved_names_from_target(self, icd10):
        """See-ref resolved entries store the target's name, not the
        source's hierarchy. Verify a known see-ref: the source key
        'arthritis, arthritic rheumatoid with carditis' resolves via
        'Rheumatoid, carditis' and should store 'Rheumatoid carditis'
        (from the target), not just 'carditis' (leaf only).
        """
        key = "arthritis, arthritic rheumatoid with carditis"
        entries = icd10._code_lookup.get(key, [])
        assert entries, f"Expected key '{key}' to exist"
        for code, name in entries:
            # The resolved name should contain the target's main term
            assert len(name.split()) >= 2, (
                f"See-ref resolved name '{name}' for code {code} "
                f"under key '{key}' is too short — "
                f"_collect_codes may not be propagating parent titles"
            )


class TestICD10KnownGoodLookups:
    """Specific conditions should resolve to correct codes."""

    # ICD-10 index uses "Diabetes, diabetic" not "diabetes mellitus"
    EXACT_CASES = [
        ("diabetes, diabetic", "E11"),     # type 2 diabetes
        ("rheumatoid carditis", "M05.30"),
        ("long covid", "U09.9"),
        ("polyarteritis nodosa", "M30.0"),
    ]

    @pytest.mark.parametrize("term,expected_prefix", EXACT_CASES)
    def test_exact_lookup_returns_expected_code(self, icd10, term, expected_prefix):
        results = icd10.lookup(term)
        codes = [r["code"] for r in results]
        assert any(c.startswith(expected_prefix) for c in codes), (
            f"lookup('{term}') expected code starting with {expected_prefix}, "
            f"got {codes}"
        )

    # Fuzzy cases: terms close to actual index phrasing
    FUZZY_CASES = [
        ("rheumatoid arthritis", "M0"),
        ("osteoarthritis knee", "M17"),
        ("pulmonary embolism", "I26"),
    ]

    @pytest.mark.parametrize("term,expected_prefix", FUZZY_CASES)
    def test_fuzzy_lookup_finds_condition(self, icd10, term, expected_prefix):
        """Fuzzy lookup for real conditions should find matches."""
        results = icd10.fuzzy_lookup(term, threshold=70)
        codes = [r["code"] for r in results]
        assert any(c.startswith(expected_prefix) for c in codes), (
            f"fuzzy_lookup('{term}') expected code starting with "
            f"{expected_prefix}, got {codes}"
        )


class TestICD10Regressions:
    """Regression tests for specific bugs found during development."""

    def test_no_daily_chronic_key(self, icd10):
        """'daily chronic' was a phantom key from unhierarchical sub-terms."""
        assert "daily chronic" not in icd10._code_lookup

    def test_no_simple_chronic_key(self, icd10):
        assert "simple chronic" not in icd10._code_lookup

    def test_carditis_maps_to_I51_not_M05(self, icd10):
        """Standalone 'carditis' = I51.89 (non-rheumatic), NOT M05.30."""
        entries = icd10._code_lookup.get("carditis", [])
        if entries:
            codes = [c for c, _ in entries]
            assert "M05.30" not in codes, (
                "Standalone 'carditis' should not map to M05.30 (rheumatoid)"
            )

    def test_hip_not_in_lookup(self, icd10):
        """'hip' was polluted via 'see condition' resolve."""
        assert "hip" not in icd10._code_lookup


# ===================================================================
# ICD-9-CM tests
# ===================================================================

class TestICD9KeyQuality:
    """ICD-9 keys should be clean normalized terms."""

    def test_no_stray_closing_parens(self, icd9):
        bad = [k for k in icd9.keys if ")" in k]
        assert bad == [], (
            f"{len(bad)} keys contain stray ')'. First 10: {bad[:10]}"
        )

    def test_few_stray_opening_parens(self, icd9):
        """A small number of unclosed '(' from RTF parsing is tolerable.

        The RTF Disease Index has some unclosed parentheticals (e.g.
        multi-line "see also" references). Accept < 20 as an RTF
        parsing limitation.
        """
        bad = [k for k in icd9.keys if "(" in k]
        assert len(bad) < 20, (
            f"{len(bad)} keys contain '(' (expected < 20). "
            f"First 10: {bad[:10]}"
        )

    def test_no_curly_braces(self, icd9):
        bad = [k for k in icd9.keys if "{" in k or "}" in k]
        assert bad == [], f"Keys with braces: {bad[:10]}"

    def test_no_backslash(self, icd9):
        bad = [k for k in icd9.keys if "\\" in k]
        assert bad == [], f"Keys with backslash: {bad[:10]}"

    def test_no_empty_keys(self, icd9):
        assert "" not in icd9._code_lookup

    def test_reasonable_key_count(self, icd9):
        n = len(icd9.keys)
        assert 40_000 < n < 80_000, f"Unexpected key count: {n}"


class TestICD9NoGenericKeys:
    FORBIDDEN_STANDALONE = [
        "chronic", "acute", "with", "hip",
        "upper", "lower", "left", "right",
    ]

    @pytest.mark.parametrize("term", FORBIDDEN_STANDALONE)
    def test_no_standalone_generic_key(self, icd9, term):
        assert term not in icd9._code_lookup, (
            f"Standalone key '{term}' should not exist. "
            f"Entries: {icd9._code_lookup[term][:3]}"
        )


class TestICD9NoSeeRefPollution:

    def test_959_9_not_over_registered(self, icd9):
        """959.9 (Injury NOS) should appear under <30 keys, not 160+."""
        count = sum(
            1 for entries in icd9._code_lookup.values()
            for code, _ in entries if code == "959.9"
        )
        assert count < 30, (
            f"959.9 registered under {count} keys — likely see-ref pollution"
        )

    def test_829_0_not_over_registered(self, icd9):
        """829.0 (Fracture NOS) should appear under <30 keys, not 190+."""
        count = sum(
            1 for entries in icd9._code_lookup.values()
            for code, _ in entries if code == "829.0"
        )
        assert count < 30, (
            f"829.0 registered under {count} keys — likely see-ref pollution"
        )

    def test_no_single_code_with_excessive_keys(self, icd9):
        code_counts: Counter[str] = Counter()
        for entries in icd9._code_lookup.values():
            for code, _ in entries:
                code_counts[code] += 1
        worst_code, worst_count = code_counts.most_common(1)[0]
        assert worst_count <= 200, (
            f"Code {worst_code} appears under {worst_count} keys"
        )


class TestICD9KnownGoodLookups:
    """Known conditions should be findable via exact or fuzzy lookup."""

    # ICD-9 index uses "Diabetes, diabetic" not "diabetes mellitus"
    EXACT_CASES = [
        ("diabetes, diabetic", "250"),
    ]

    @pytest.mark.parametrize("term,expected_prefix", EXACT_CASES)
    def test_exact_lookup_returns_expected_code(self, icd9, term, expected_prefix):
        results = icd9.lookup(term)
        codes = [r["code"] for r in results]
        assert any(c.startswith(expected_prefix) for c in codes), (
            f"lookup('{term}') expected code starting with {expected_prefix}, "
            f"got {codes}"
        )

    def test_fuzzy_lookup_returns_results(self, icd9):
        """Fuzzy lookup for a real condition should find matches."""
        results = icd9.fuzzy_lookup("rheumatoid arthritis", threshold=70)
        assert len(results) > 0, "fuzzy_lookup should return results"


# ===================================================================
# Cross-parser consistency
# ===================================================================

class TestNormalizerConsistency:
    """Both parsers should use the same normalization logic."""

    CASES = [
        "Arthritis, arthritic (acute) (chronic)",
        "Diabetes-related condition",
        "Cogan's syndrome",
        "Hip (joint)",
        "Absence (of) (organ or part) breast(s) (and nipple(s))",
    ]

    @pytest.mark.parametrize("text", CASES)
    def test_normalizers_agree(self, text):
        n9 = CDCIndex9._default_normalize(text)
        n10 = CDCIndex._default_normalize(text)
        assert n9 == n10, (
            f"Normalizers disagree on '{text}': "
            f"ICD-9='{n9}', ICD-10='{n10}'"
        )

    def test_nested_parens_fully_stripped(self):
        """Nested parens like 'breast(s) (and nipple(s))' should be clean."""
        text = "Absence (of) breast(s) (and nipple(s)) (acquired)"
        result = CDCIndex._default_normalize(text)
        assert ")" not in result
        assert "(" not in result

    def test_possessives_stripped(self):
        assert CDCIndex._default_normalize("Cogan's syndrome") == "cogan syndrome"

    def test_hyphens_to_spaces(self):
        assert CDCIndex._default_normalize("post-COVID-19") == "post covid 19"
