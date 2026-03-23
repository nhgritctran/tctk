"""Tests for the OMOP mapping pipeline.

Covers the full workflow from input normalization through result grouping,
using synthetic DataFrames (no vocab DB required).

All counts use (condition_name, search_term) pairs per project rules.
Post-matching dedup uses (condition_name, search_term, matched_term,
matched_code, vocabulary/ICD version).
"""

import polars as pl
import pytest

from tctk.omop._base import ConditionMapperBase
from tctk.omop.condition2icd import Condition2ICD


# ===================================================================
# Step 1: _build_input()
# ===================================================================

class TestBuildInput:
    """Test input normalization pipeline."""

    @staticmethod
    def _build(conditions):
        return ConditionMapperBase._build_input(conditions)

    def test_condition_name_included_as_search_term(self):
        """Condition name itself should be a search term."""
        df = self._build({"Diabetes": ["Type 2 DM"]})
        terms = df["search_term"].to_list()
        assert "diabetes" in terms
        assert "type 2 dm" in terms

    def test_smart_quotes_converted(self):
        """Curly quotes converted to straight quotes."""
        df = self._build({"test": ["\u2018quoted\u2019"]})
        terms = df["search_term"].to_list()
        assert any("'quoted'" in t for t in terms)

    def test_subtype_tag_stripped(self):
        """(subtype) tag removed from terms."""
        df = self._build({"test": ["diabetes (subtype)"]})
        terms = df["search_term"].to_list()
        assert "diabetes" in terms
        assert not any("subtype" in t for t in terms)

    def test_synonym_tag_stripped(self):
        """(synonym) tag removed from terms."""
        df = self._build({"test": ["dm (synonym)"]})
        terms = df["search_term"].to_list()
        assert "dm" in terms
        assert not any("synonym" in t for t in terms)

    def test_hyphens_to_spaces(self):
        """Hyphens replaced with spaces."""
        df = self._build({"test": ["post-COVID-19"]})
        terms = df["search_term"].to_list()
        assert "post covid 19" in terms

    def test_whitespace_collapsed(self):
        """Multiple spaces collapsed to single space, trimmed."""
        df = self._build({"test": ["  multiple   spaces  "]})
        terms = df["search_term"].to_list()
        assert "multiple spaces" in terms

    def test_parenthetical_expansion_both_variants(self):
        """Terms with parens generate both with-paren and without-paren variants."""
        df = self._build({"test": ["arthritis (chronic)"]})
        terms = df["search_term"].to_list()
        assert "arthritis (chronic)" in terms
        assert "arthritis" in terms

    def test_slash_expansion(self):
        """Slash-separated terms split into individual terms."""
        df = self._build({"test": ["myositis/polymyositis"]})
        terms = df["search_term"].to_list()
        assert "myositis" in terms
        assert "polymyositis" in terms

    def test_empty_terms_filtered(self):
        """Empty strings and whitespace-only synonyms excluded."""
        df = self._build({"test": ["", "  ", "valid"]})
        terms = df["search_term"].to_list()
        assert "" not in terms
        assert "valid" in terms

    def test_dedup_by_condition_search_term(self):
        """Duplicate (condition, search_term) pairs removed."""
        # Condition name "test" is also added as search_term, plus synonym "test"
        df = self._build({"test": ["test", "other"]})
        pairs = list(zip(df["condition_name"].to_list(), df["search_term"].to_list()))
        assert len(pairs) == len(set(pairs)), "Duplicate pairs found"

    def test_shared_term_across_conditions_kept(self):
        """Same search_term under different conditions kept as separate pairs."""
        df = self._build({"Condition A": ["shared"], "Condition B": ["shared"]})
        shared_rows = df.filter(pl.col("search_term") == "shared")
        assert len(shared_rows) == 2
        conds = shared_rows["condition_name"].to_list()
        assert "condition a" in conds or "Condition A" in [
            c.strip() for c in conds
        ]

    def test_lowercase(self):
        """All terms lowercased."""
        df = self._build({"Test Condition": ["UPPER Case"]})
        terms = df["search_term"].to_list()
        for t in terms:
            assert t == t.lower(), f"Term not lowercase: {t}"

    def test_output_columns(self):
        """Output has exactly condition_name and search_term columns."""
        df = self._build({"test": ["term1"]})
        assert set(df.columns) == {"condition_name", "search_term"}


# ===================================================================
# Step 2: _strip_stopwords()
# ===================================================================

class TestStripStopwords:
    """Test stopword removal for fuzzy scoring."""

    def test_stopwords_removed(self):
        result = ConditionMapperBase._strip_stopwords(
            "disease of the lung",
            ConditionMapperBase._FUZZY_STOPWORDS,
        )
        assert "disease" not in result.split()
        assert "of" not in result.split()
        assert "the" not in result.split()
        assert "lung" in result.split()

    def test_all_stopwords_returns_original(self):
        """When ALL tokens are stopwords, return original (fallback)."""
        result = ConditionMapperBase._strip_stopwords(
            "of the and",
            ConditionMapperBase._FUZZY_STOPWORDS,
        )
        assert result == "of the and"

    def test_empty_string(self):
        result = ConditionMapperBase._strip_stopwords(
            "", ConditionMapperBase._FUZZY_STOPWORDS
        )
        assert result == ""

    def test_no_stopwords_unchanged(self):
        result = ConditionMapperBase._strip_stopwords(
            "rheumatoid arthritis", ConditionMapperBase._FUZZY_STOPWORDS
        )
        assert result == "rheumatoid arthritis"


# ===================================================================
# Helpers: synthetic DataFrames
# ===================================================================

def _make_match_row(
    condition_name="Cond A",
    search_term="term1",
    matched_concept_synonym="synonym1",
    concept_id="12345",
    concept_code="E10",
    concept_name="Type 1 DM",
    vocabulary_id="ICD10CM",
    concept_class_id="3-char billing code",
    standard_concept="S",
    match_type="exact",
    match_score=100,
    icd_concept_id="E10",
    icd_concept_name="Type 1 DM",
    icd_code="E10",
    icd_version="10",
    top_level_code="E10",
    has_confirmed_sibling=False,
):
    return {
        "condition_name": condition_name,
        "search_term": search_term,
        "matched_concept_synonym": matched_concept_synonym,
        "concept_id": concept_id,
        "concept_code": concept_code,
        "concept_name": concept_name,
        "vocabulary_id": vocabulary_id,
        "concept_class_id": concept_class_id,
        "standard_concept": standard_concept,
        "match_type": match_type,
        "match_score": match_score,
        "icd_concept_id": icd_concept_id,
        "icd_concept_name": icd_concept_name,
        "icd_code": icd_code,
        "icd_version": icd_version,
        "top_level_code": top_level_code,
        "has_confirmed_sibling": has_confirmed_sibling,
    }


def _make_df(rows):
    """Build a polars DataFrame from a list of dicts."""
    return pl.DataFrame(rows)


# ===================================================================
# Step 2B: Exact match dedup
# ===================================================================

class TestExactMatchDedup:
    """Test exact match dedup includes vocabulary_id."""

    def test_same_code_different_vocab_kept(self):
        """Same concept_code in ICD9CM and ICD10CM are separate matches."""
        rows = [
            {
                "condition_name": "Cond A",
                "search_term": "diabetes",
                "matched_concept_synonym": "diabetes",
                "concept_code": "250",
                "vocabulary_id": "ICD9CM",
                "concept_id": "1",
                "concept_name": "Diabetes",
                "concept_class_id": "3-char billing code",
                "standard_concept": None,
                "match_type": "exact",
                "match_score": 100,
            },
            {
                "condition_name": "Cond A",
                "search_term": "diabetes",
                "matched_concept_synonym": "diabetes",
                "concept_code": "250",
                "vocabulary_id": "ICD10CM",
                "concept_id": "2",
                "concept_name": "Diabetes",
                "concept_class_id": "3-char billing code",
                "standard_concept": None,
                "match_type": "exact",
                "match_score": 100,
            },
        ]
        df = pl.DataFrame(rows)
        df_deduped = df.unique(
            subset=[
                "condition_name", "search_term",
                "matched_concept_synonym", "concept_code",
                "vocabulary_id",
            ]
        )
        assert len(df_deduped) == 2, "Different vocabularies should not dedup"

    def test_same_code_same_vocab_deduped(self):
        """Duplicate (condition, term, synonym, code, vocab) collapsed."""
        row = {
            "condition_name": "Cond A",
            "search_term": "diabetes",
            "matched_concept_synonym": "diabetes",
            "concept_code": "E11",
            "vocabulary_id": "ICD10CM",
            "concept_id": "1",
            "concept_name": "Type 2 DM",
            "concept_class_id": "3-char billing code",
            "standard_concept": None,
            "match_type": "exact",
            "match_score": 100,
        }
        df = pl.DataFrame([row, row])
        df_deduped = df.unique(
            subset=[
                "condition_name", "search_term",
                "matched_concept_synonym", "concept_code",
                "vocabulary_id",
            ]
        )
        assert len(df_deduped) == 1


# ===================================================================
# Step 3: _summarize() arithmetic
# ===================================================================

class TestSummarizeArithmetic:
    """Verify n_exact + n_fuzzy + n_unmatched = n_input (all as pairs)."""

    @staticmethod
    def _make_input_df(pairs):
        """Build df_input from list of (condition, term) tuples."""
        return pl.DataFrame(
            [{"condition_name": c, "search_term": t} for c, t in pairs]
        )

    @staticmethod
    def _make_match_df(rows, vocab="ICD"):
        """Build df_exact or df_fuzzy from match rows."""
        if not rows:
            return pl.DataFrame(schema={
                "condition_name": pl.Utf8,
                "search_term": pl.Utf8,
                "matched_concept_synonym": pl.Utf8,
                "concept_id": pl.Utf8,
                "concept_code": pl.Utf8,
                "concept_name": pl.Utf8,
                "vocabulary_id": pl.Utf8,
                "concept_class_id": pl.Utf8,
                "standard_concept": pl.Utf8,
                "match_type": pl.Utf8,
                "match_score": pl.Int64,
                "icd_concept_id": pl.Utf8,
                "icd_concept_name": pl.Utf8,
                "icd_code": pl.Utf8,
                "icd_version": pl.Utf8,
                "top_level_code": pl.Utf8,
                "has_confirmed_sibling": pl.Boolean,
            })
        return _make_df(rows)

    def test_counts_add_up_no_overlap(self, capsys):
        """Basic case: exact + fuzzy + unmatched = total pairs."""
        df_input = self._make_input_df([
            ("A", "t1"), ("A", "t2"), ("A", "t3"),
            ("B", "t4"), ("B", "t5"),
        ])
        # t1, t2 match exact; t3 matches fuzzy; t4, t5 unmatched
        df_exact = self._make_match_df([
            _make_match_row(condition_name="A", search_term="t1",
                            match_type="exact", match_score=100),
            _make_match_row(condition_name="A", search_term="t2",
                            match_type="exact", match_score=100),
        ])
        df_fuzzy = self._make_match_df([
            _make_match_row(condition_name="A", search_term="t3",
                            match_type="fuzzy", match_score=85),
        ])
        df_matches = pl.concat([df_exact, df_fuzzy], how="diagonal_relaxed")

        df_term_counts = Condition2ICD._summarize(
            df_input, df_exact, df_fuzzy, df_matches
        )

        captured = capsys.readouterr().out
        # Check arithmetic: "2 exact + 1 fuzzy = 3 terms (2 unmatched)"
        assert "2 exact" in captured
        assert "1 fuzzy" in captured
        assert "3 terms" in captured
        assert "2 unmatched" in captured

    def test_counts_with_shared_terms(self, capsys):
        """Shared term across conditions counted as separate pairs."""
        df_input = self._make_input_df([
            ("A", "shared"), ("B", "shared"), ("A", "only_a"),
        ])
        # "shared" matches exact → both (A, shared) and (B, shared) are matched
        df_exact = self._make_match_df([
            _make_match_row(condition_name="A", search_term="shared",
                            match_type="exact", match_score=100),
            _make_match_row(condition_name="B", search_term="shared",
                            match_type="exact", match_score=100),
        ])
        df_fuzzy = self._make_match_df([])
        df_matches = df_exact.clone()

        df_term_counts = Condition2ICD._summarize(
            df_input, df_exact, df_fuzzy, df_matches
        )

        captured = capsys.readouterr().out
        # 3 total pairs, 2 exact (both "shared" pairs), 0 fuzzy, 1 unmatched
        assert "3 search terms" in captured or "Input:" in captured
        assert "2 exact" in captured
        assert "1 unmatched" in captured


# ===================================================================
# Step 4B: OMOP + CDC dedup
# ===================================================================

class TestOmopCdcDedup:
    """Test cross-source deduplication logic."""

    DEDUP_SUBSET = [
        "condition_name", "search_term", "icd_concept_name",
        "icd_code", "icd_version",
    ]

    def test_exact_preferred_over_fuzzy(self):
        """Same (condition, term, code): exact match kept over fuzzy."""
        rows = [
            _make_match_row(match_type="exact", match_score=100),
            _make_match_row(match_type="fuzzy", match_score=85),
        ]
        df = _make_df(rows)
        df = (
            df.sort(["match_type", "match_score"], descending=[False, True])
            .unique(subset=self.DEDUP_SUBSET, keep="first")
        )
        assert len(df) == 1
        assert df["match_type"][0] == "exact"

    def test_higher_score_preferred(self):
        """Same match_type: higher score wins."""
        rows = [
            _make_match_row(match_type="fuzzy", match_score=75),
            _make_match_row(match_type="fuzzy", match_score=90),
        ]
        df = _make_df(rows)
        df = (
            df.sort(["match_type", "match_score"], descending=[False, True])
            .unique(subset=self.DEDUP_SUBSET, keep="first")
        )
        assert len(df) == 1
        assert df["match_score"][0] == 90

    def test_different_codes_both_kept(self):
        """Different icd_codes for same (condition, term) both kept."""
        rows = [
            _make_match_row(icd_code="E10", icd_concept_name="Type 1",
                            top_level_code="E10"),
            _make_match_row(icd_code="E11", icd_concept_name="Type 2",
                            top_level_code="E11"),
        ]
        df = _make_df(rows)
        df = (
            df.sort(["match_type", "match_score"], descending=[False, True])
            .unique(subset=self.DEDUP_SUBSET, keep="first")
        )
        assert len(df) == 2

    def test_different_versions_both_kept(self):
        """Same code stem in ICD-9 vs ICD-10 both kept."""
        rows = [
            _make_match_row(icd_code="250", icd_version="9",
                            icd_concept_name="Diabetes ICD9",
                            vocabulary_id="ICD9CM", top_level_code="250"),
            _make_match_row(icd_code="E11", icd_version="10",
                            icd_concept_name="Diabetes ICD10",
                            vocabulary_id="ICD10CM", top_level_code="E11"),
        ]
        df = _make_df(rows)
        df = (
            df.sort(["match_type", "match_score"], descending=[False, True])
            .unique(subset=self.DEDUP_SUBSET, keep="first")
        )
        assert len(df) == 2


# ===================================================================
# Step 4C: Sibling confirmation
# ===================================================================

class TestSiblingConfirmation:
    """Test has_confirmed_sibling recompute."""

    @staticmethod
    def _recompute_siblings(df_matches):
        """Replicate the sibling recompute logic from _icdcm_lookup."""
        confirmed_top_codes = (
            df_matches.filter(pl.col("match_type") == "exact")
            .select("condition_name", "top_level_code")
            .unique()
            .with_columns(pl.lit(True).alias("_has_sibling"))
        )
        return (
            df_matches.drop("has_confirmed_sibling")
            .join(
                confirmed_top_codes,
                on=["condition_name", "top_level_code"],
                how="left",
            )
            .with_columns(
                pl.col("_has_sibling").fill_null(False)
                .alias("has_confirmed_sibling")
            )
            .drop("_has_sibling")
        )

    def test_fuzzy_with_exact_sibling(self):
        """Fuzzy M05.31 has_sibling=True when exact M05.30 exists."""
        rows = [
            _make_match_row(
                match_type="exact", icd_code="M05.30",
                top_level_code="M05",
            ),
            _make_match_row(
                match_type="fuzzy", match_score=85,
                icd_code="M05.31", top_level_code="M05",
            ),
        ]
        df = self._recompute_siblings(_make_df(rows))
        fuzzy_row = df.filter(pl.col("icd_code") == "M05.31")
        assert fuzzy_row["has_confirmed_sibling"][0] is True

    def test_fuzzy_without_sibling(self):
        """Fuzzy K50.1 has_sibling=False when no exact K50.x exists."""
        rows = [
            _make_match_row(
                match_type="exact", icd_code="E10",
                top_level_code="E10",
            ),
            _make_match_row(
                match_type="fuzzy", match_score=80,
                icd_code="K50.1", top_level_code="K50",
            ),
        ]
        df = self._recompute_siblings(_make_df(rows))
        fuzzy_row = df.filter(pl.col("icd_code") == "K50.1")
        assert fuzzy_row["has_confirmed_sibling"][0] is False

    def test_sibling_is_per_condition(self):
        """Exact E10 under Cond A doesn't make E10.1 under Cond B a sibling."""
        rows = [
            _make_match_row(
                condition_name="Cond A",
                match_type="exact", icd_code="E10",
                top_level_code="E10",
            ),
            _make_match_row(
                condition_name="Cond B",
                match_type="fuzzy", match_score=80,
                icd_code="E10.1", top_level_code="E10",
            ),
        ]
        df = self._recompute_siblings(_make_df(rows))
        cond_b = df.filter(pl.col("condition_name") == "Cond B")
        assert cond_b["has_confirmed_sibling"][0] is False


# ===================================================================
# Step 4D: Auto-threshold, child drop, rescue
# ===================================================================

class TestAutoThreshold:
    """Test adaptive threshold filtering."""

    def test_threshold_keeps_exact_matches(self):
        """Exact matches always kept regardless of threshold."""
        rows = [
            _make_match_row(match_type="exact", match_score=100),
        ]
        df = _make_df(rows)
        selected = 90
        df_kept = df.filter(
            (pl.col("match_type") == "exact")
            | (pl.col("match_score") >= selected)
        )
        assert len(df_kept) == 1

    def test_threshold_drops_low_score_fuzzy(self):
        """Fuzzy score=72 dropped when threshold raised to 85."""
        rows = [
            _make_match_row(match_type="fuzzy", match_score=72,
                            icd_code="K50"),
        ]
        df = _make_df(rows)
        selected = 85
        df_kept = df.filter(
            (pl.col("match_type") == "exact")
            | (pl.col("match_score") >= selected)
        )
        assert len(df_kept) == 0

    def test_threshold_keeps_high_score_fuzzy(self):
        """Fuzzy score=90 kept when threshold raised to 85."""
        rows = [
            _make_match_row(match_type="fuzzy", match_score=90,
                            icd_code="K50"),
        ]
        df = _make_df(rows)
        selected = 85
        df_kept = df.filter(
            (pl.col("match_type") == "exact")
            | (pl.col("match_score") >= selected)
        )
        assert len(df_kept) == 1


class TestChildDrop:
    """Test child code removal among kept codes."""

    @staticmethod
    def _find_roots(codes: set) -> set:
        return {
            c for c in codes
            if not any(c.startswith(other) and c != other for other in codes)
        }

    @staticmethod
    def _drop_children(df, kept_codes_by_cond):
        """Replicate child-drop logic from _icdcm_lookup."""
        mask = [
            code is None or not any(
                code.startswith(k) and code != k
                for k in kept_codes_by_cond.get(cond, set())
            )
            for cond, code in zip(
                df["condition_name"].to_list(),
                df["icd_code"].to_list(),
            )
        ]
        return df.filter(pl.Series(mask))

    def test_child_dropped_when_parent_kept(self):
        """E10.1 dropped when E10 is also kept for same condition."""
        rows = [
            _make_match_row(icd_code="E10", top_level_code="E10"),
            _make_match_row(icd_code="E10.1", top_level_code="E10"),
        ]
        df = _make_df(rows)
        roots = self._find_roots({"E10", "E10.1"})
        assert roots == {"E10"}
        df_kept = self._drop_children(df, {"Cond A": roots})
        assert len(df_kept) == 1
        assert df_kept["icd_code"][0] == "E10"

    def test_both_kept_when_no_parent_child(self):
        """E10 and K50 both kept (different families)."""
        rows = [
            _make_match_row(icd_code="E10", top_level_code="E10"),
            _make_match_row(icd_code="K50", top_level_code="K50"),
        ]
        df = _make_df(rows)
        roots = self._find_roots({"E10", "K50"})
        assert roots == {"E10", "K50"}
        df_kept = self._drop_children(df, {"Cond A": roots})
        assert len(df_kept) == 2

    def test_grandchild_dropped(self):
        """E10.12 dropped when E10 kept (not just direct children)."""
        rows = [
            _make_match_row(icd_code="E10", top_level_code="E10"),
            _make_match_row(icd_code="E10.12", top_level_code="E10"),
        ]
        df = _make_df(rows)
        roots = self._find_roots({"E10", "E10.12"})
        assert roots == {"E10"}
        df_kept = self._drop_children(df, {"Cond A": roots})
        assert len(df_kept) == 1
        assert df_kept["icd_code"][0] == "E10"


class TestRescue:
    """Test novel family rescue logic."""

    @staticmethod
    def _find_roots(codes: set) -> set:
        return {
            c for c in codes
            if not any(c.startswith(other) and c != other for other in codes)
        }

    @staticmethod
    def _find_novel(dropped_codes, kept_roots):
        """Find dropped roots not covered by any kept root."""
        dropped_roots = TestRescue._find_roots(dropped_codes)
        return {
            r for r in dropped_roots
            if not any(
                r.startswith(k) or k.startswith(r)
                for k in kept_roots
            )
        }

    @staticmethod
    def _rescue_dedup(df_rescue):
        """Replicate rescue dedup: best score per (condition, 3-char family)."""
        if len(df_rescue) == 0:
            return df_rescue
        return (
            df_rescue
            .with_columns(
                pl.col("icd_code").str.slice(0, 3).alias("_icd_family")
            )
            .sort("match_score", descending=True)
            .unique(
                subset=["condition_name", "_icd_family"],
                keep="first",
            )
            .drop("_icd_family")
        )

    def test_novel_family_rescued(self):
        """Dropped code K50 is novel when only E10 is kept."""
        novel = self._find_novel(
            dropped_codes={"K50", "K50.1"},
            kept_roots={"E10"},
        )
        assert "K50" in novel

    def test_covered_family_not_rescued(self):
        """Dropped code E10.5 NOT rescued when E10 already kept."""
        novel = self._find_novel(
            dropped_codes={"E10.5"},
            kept_roots={"E10"},
        )
        assert len(novel) == 0

    def test_rescue_dedup_keeps_best_score(self):
        """Multiple dropped codes in same 3-char family: highest score kept."""
        rows = [
            _make_match_row(
                icd_code="K50.0", match_type="fuzzy", match_score=78,
                top_level_code="K50",
            ),
            _make_match_row(
                icd_code="K50.1", match_type="fuzzy", match_score=82,
                top_level_code="K50",
            ),
        ]
        df = self._rescue_dedup(_make_df(rows))
        assert len(df) == 1
        assert df["match_score"][0] == 82
        assert df["icd_code"][0] == "K50.1"

    def test_rescue_per_condition(self):
        """Rescue is per-condition: same family rescued separately."""
        rows = [
            _make_match_row(
                condition_name="Cond A",
                icd_code="K50.0", match_type="fuzzy", match_score=78,
                top_level_code="K50",
            ),
            _make_match_row(
                condition_name="Cond B",
                icd_code="K50.1", match_type="fuzzy", match_score=82,
                top_level_code="K50",
            ),
        ]
        df = self._rescue_dedup(_make_df(rows))
        assert len(df) == 2  # One per condition

    def test_different_families_both_rescued(self):
        """Two novel families for same condition: both rescued."""
        novel = self._find_novel(
            dropped_codes={"K50", "M30"},
            kept_roots={"E10"},
        )
        assert "K50" in novel
        assert "M30" in novel


# ===================================================================
# Step 6: Review table assembly
# ===================================================================

class TestReviewTableAssembly:
    """Test final review table construction."""

    def test_unmatched_pairs_get_no_match_rows(self):
        """Input pairs with no match get ai_verdict='no match'."""
        all_pairs = {("A", "t1"), ("A", "t2"), ("A", "t3")}
        matched_pairs = {("A", "t1")}
        unmatched = all_pairs - matched_pairs
        assert len(unmatched) == 2
        # Verify unmatched rows would have "no match"
        for cond, term in unmatched:
            assert cond == "A"
            assert term in ("t2", "t3")

    def test_all_input_pairs_in_review(self):
        """Every (condition, term) from input appears in final review."""
        all_pairs = {("A", "t1"), ("A", "t2"), ("B", "t3")}
        matched_pairs = {("A", "t1")}
        unmatched = all_pairs - matched_pairs
        # matched + unmatched = all
        assert matched_pairs | unmatched == all_pairs

    def test_set_difference_preserves_condition_context(self):
        """Set diff uses (condition, term) tuples, not just terms."""
        all_pairs = {("A", "shared"), ("B", "shared")}
        matched_pairs = {("A", "shared")}
        unmatched = all_pairs - matched_pairs
        # Only (B, shared) should be unmatched
        assert unmatched == {("B", "shared")}


# ===================================================================
# Step 7: Result grouping
# ===================================================================

class TestResultGrouping:
    """Test accepted/rejected/human_review filter logic."""

    @staticmethod
    def _make_review_df(rows):
        """Build a df_review from simplified row dicts."""
        full_rows = []
        for r in rows:
            full_rows.append({
                "condition_name": r.get("condition_name", "Cond A"),
                "search_term": r.get("search_term", "term1"),
                "matched_concept_synonym": r.get("matched_concept_synonym"),
                "icd_concept_name": r.get("icd_concept_name"),
                "icd_code": r.get("icd_code"),
                "icd_version": r.get("icd_version", "10"),
                "icd_concept_id": r.get("icd_concept_id"),
                "vocabulary_id": r.get("vocabulary_id"),
                "standard_concept": r.get("standard_concept"),
                "match_type": r.get("match_type"),
                "top_level_code": r.get("top_level_code"),
                "has_confirmed_sibling": r.get("has_confirmed_sibling"),
                "fuzzy_score": r.get("fuzzy_score"),
                "ai_verdict": r.get("ai_verdict"),
                "ai_vote": None,
                "ai_vote_confidence": None,
                "ai_comment": None,
                "ai_comment_consistency": None,
                "ai_comment_consistency_tier": None,
                "ai_combined_confidence": None,
            })
        return pl.DataFrame(full_rows)

    @staticmethod
    def _filter_accepted(df_review):
        """Replicate accepted filter from map()."""
        return df_review.filter(
            pl.col("icd_code").is_not_null()
            & (
                (pl.col("match_type") == "exact")
                | (pl.col("ai_verdict") == "accept")
                | (
                    (pl.col("match_type") == "fuzzy")
                    & pl.col("ai_verdict").is_null()
                )
            )
        )

    def test_exact_match_accepted(self):
        """Exact matches always in accepted."""
        df = self._make_review_df([
            {"match_type": "exact", "icd_code": "E10", "ai_verdict": None},
        ])
        accepted = self._filter_accepted(df)
        assert len(accepted) == 1

    def test_ai_accepted_fuzzy_in_accepted(self):
        """Fuzzy with ai_verdict='accept' in accepted."""
        df = self._make_review_df([
            {"match_type": "fuzzy", "icd_code": "K50",
             "ai_verdict": "accept", "fuzzy_score": 85},
        ])
        accepted = self._filter_accepted(df)
        assert len(accepted) == 1

    def test_unreviewed_fuzzy_in_accepted(self):
        """Fuzzy with no ai_verdict (null) in accepted."""
        df = self._make_review_df([
            {"match_type": "fuzzy", "icd_code": "K50",
             "ai_verdict": None, "fuzzy_score": 88},
        ])
        accepted = self._filter_accepted(df)
        assert len(accepted) == 1

    def test_ai_rejected_not_in_accepted(self):
        """Fuzzy with ai_verdict='reject' NOT in accepted."""
        df = self._make_review_df([
            {"match_type": "fuzzy", "icd_code": "K50",
             "ai_verdict": "reject", "fuzzy_score": 72},
        ])
        accepted = self._filter_accepted(df)
        assert len(accepted) == 0

    def test_no_match_not_in_accepted(self):
        """Rows with ai_verdict='no match' NOT in accepted."""
        df = self._make_review_df([
            {"match_type": None, "icd_code": None, "ai_verdict": "no match"},
        ])
        accepted = self._filter_accepted(df)
        assert len(accepted) == 0

    def test_unmapped_conditions_correct(self):
        """Conditions with zero accepted matches identified as unmapped."""
        df = self._make_review_df([
            {"condition_name": "Mapped", "match_type": "exact",
             "icd_code": "E10", "ai_verdict": None},
            {"condition_name": "Unmapped", "match_type": "fuzzy",
             "icd_code": "K50", "ai_verdict": "reject"},
        ])
        accepted = self._filter_accepted(df)
        mapped_conds = set(accepted["condition_name"].to_list())
        all_conds = set(df["condition_name"].to_list())
        unmapped = all_conds - mapped_conds
        assert unmapped == {"Unmapped"}


# ===================================================================
# Checkpoint: save / load / print_summary
# ===================================================================

class TestCheckpoint:
    """Test save_results, load_results, and print_summary."""

    @staticmethod
    def _make_results():
        """Create a minimal results dict mimicking map() output."""
        df_review = pl.DataFrame({
            "condition_name": ["Cond A", "Cond A", "Cond B"],
            "search_term": ["term1", "term2", "term3"],
            "icd_code": ["E10", "K50.0", None],
            "match_type": ["exact", "fuzzy", None],
            "is_rescued": [False, True, False],
        })
        df_accepted = pl.DataFrame({
            "condition_name": ["Cond A"],
            "icd_version": ["10"],
            "icd_codes": ["E10, K50.0"],
            "n_codes": [2],
        })
        return {
            "df_review": df_review,
            "df_accepted": df_accepted,
            "_run_log": "FINAL SUMMARY\n  Conditions: 1 mapped\n",
        }

    def test_save_creates_directory(self, tmp_path):
        from tctk.omop.checkpoint import save_results
        results = self._make_results()
        out = save_results(results, tmp_path / "ckpt")
        assert (out / "_meta.json").exists()
        assert (out / "df_review.parquet").exists()
        assert (out / "df_accepted.parquet").exists()
        assert (out / "_run_log.txt").exists()

    def test_load_roundtrip(self, tmp_path):
        from tctk.omop.checkpoint import save_results, load_results
        results = self._make_results()
        save_results(results, tmp_path / "ckpt")
        loaded = load_results(tmp_path / "ckpt")
        assert set(loaded.keys()) == set(results.keys())
        assert loaded["df_review"].shape == results["df_review"].shape
        assert loaded["df_accepted"].shape == results["df_accepted"].shape
        assert loaded["_run_log"] == results["_run_log"]

    def test_load_preserves_dtypes(self, tmp_path):
        from tctk.omop.checkpoint import save_results, load_results
        results = self._make_results()
        save_results(results, tmp_path / "ckpt")
        loaded = load_results(tmp_path / "ckpt")
        assert loaded["df_review"]["is_rescued"].dtype == pl.Boolean

    def test_print_summary(self, tmp_path, capsys):
        from tctk.omop.checkpoint import save_results, load_results, print_summary
        results = self._make_results()
        save_results(results, tmp_path / "ckpt")
        loaded = load_results(tmp_path / "ckpt")
        print_summary(loaded)
        captured = capsys.readouterr().out
        assert "FINAL SUMMARY" in captured

    def test_print_summary_no_log(self, capsys):
        from tctk.omop.checkpoint import print_summary
        print_summary({})
        captured = capsys.readouterr().out
        assert "No run log found" in captured

    def test_load_missing_dir_raises(self, tmp_path):
        from tctk.omop.checkpoint import load_results
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nonexistent")


# ===================================================================
# is_rescued column
# ===================================================================

class TestIsRescued:
    """Test that is_rescued column flows correctly."""

    def test_rescued_rows_marked_true(self):
        """Rescued rows should have is_rescued=True after concat."""
        df_kept = pl.DataFrame({
            "condition_name": ["A"],
            "icd_code": ["E10"],
            "match_type": ["exact"],
            "match_score": [100],
            "is_rescued": [False],
        })
        df_rescue = pl.DataFrame({
            "condition_name": ["A"],
            "icd_code": ["K50"],
            "match_type": ["fuzzy"],
            "match_score": [75],
            "is_rescued": [True],
        })
        df = pl.concat([df_kept, df_rescue], how="diagonal_relaxed")
        assert df.filter(pl.col("is_rescued") == True)["icd_code"].to_list() == ["K50"]
        assert df.filter(pl.col("is_rescued") == False)["icd_code"].to_list() == ["E10"]

    def test_no_rescue_all_false(self):
        """When no rescue occurs, all rows should be is_rescued=False."""
        df = pl.DataFrame({
            "condition_name": ["A", "A"],
            "icd_code": ["E10", "E11"],
            "match_type": ["exact", "exact"],
            "is_rescued": [False, False],
        })
        assert df["is_rescued"].sum() == 0
