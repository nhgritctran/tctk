"""
Condition2ICD — map free-text condition names to OMOP ICD Concept IDs.

Pipeline:
    1. Normalize input terms
    2. OMOP lookup — exact + fuzzy match against ICD-9-CM / ICD-10-CM
       synonyms via local DuckDB
    3. Summarize OMOP matching stats
    4. (Optional) ICD-CM index lookup — search all input terms against
       CDC ICD-10-CM and ICD-9-CM indices, then deduplicate across
       OMOP + CDC results (exact > fuzzy, highest score wins)
    5. (Optional) AI review of all fuzzy matches via Gemini API

All steps can be run via a single ``map()`` call:

    mapper = Condition2ICD()

    # OMOP + ICD-CM index lookup (default)
    results = mapper.map(conditions, fuzzy_threshold=70)

    # OMOP + ICD-CM + AI review
    mapper.set_api_key(key_file="gemini_api_key.json")
    results = mapper.map(conditions, fuzzy_threshold=70, ai_review=True)

Setup:
    # Vocab database is auto-downloaded from Hugging Face on first use
"""

import io
import re
import sys
from contextlib import contextmanager
from typing import Optional

import polars as pl
from tqdm.auto import tqdm

from tctk._utils import (
    strip_accents,
    write_tsv_bom,
)
from tctk.omop._base import ConditionMapperBase

__all__ = ["Condition2ICD"]


# -------------------------------------------------------------------
# Print-capture utility
# -------------------------------------------------------------------

@contextmanager
def _capture_prints():
    """Capture stdout while still printing to terminal.

    Usage::

        with _capture_prints() as buf:
            print("hello")
        captured = buf.getvalue()
    """
    buf = io.StringIO()
    original = sys.stdout
    # Tee: write to both original stdout and buffer
    class _Tee:
        def write(self, s):
            original.write(s)
            buf.write(s)
        def flush(self):
            original.flush()
    sys.stdout = _Tee()
    try:
        yield buf
    finally:
        sys.stdout = original


class Condition2ICD(ConditionMapperBase):
    """Map condition names and synonyms to OMOP ICD Concept IDs.

    Uses a local DuckDB vocabulary database built from Athena CSV files.
    No network access required for mapping — only for optional AI review.

    Parameters
    ----------
    vocab_db : str, optional
        Path to the DuckDB vocabulary database.
        Default: auto-downloaded from Hugging Face
    force_download_db : bool
        Force re-download of the vocabulary database. Default False.
    """

    _TARGET_ID_COL = "icd_concept_id"
    _TARGET_NAME_COL = "icd_concept_name"
    _VOCAB_LABEL = "ICD"

    # -------------------------------------------------------------------
    # AI review hook overrides
    # -------------------------------------------------------------------

    def _ai_review_format_match_line(self, row: dict) -> str:
        parts = [
            f"term={row.get('search_term', '')}",
            f"fuzzy_synonym={row.get('matched_concept_synonym', '')}",
            f"target={row.get(self._TARGET_NAME_COL, '')}",
            f"icd_code={row.get('icd_code', '')}",
            f"icd_version={row.get('icd_version', '')}",
            f"sibling_confirmed={row.get('has_confirmed_sibling', False)}",
            f"id={row[self._TARGET_ID_COL]}",
            f"fuzzy={row['match_score']}",
        ]
        return " | ".join(parts)

    def _ai_review_system_prompt(self) -> str:
        return (
            f"You are a clinical terminologist with expertise in OMOP and {self._VOCAB_LABEL} vocabularies.\n"
            f"You are validating fuzzy matches between search terms and {self._VOCAB_LABEL} concepts.\n"
            "Each entry shows: term (search term), fuzzy_synonym (the vocabulary "
            f"synonym text that fuzzy-matched), target (the {self._VOCAB_LABEL} concept name), "
            "icd_code (the ICD billing code), icd_version (9 or 10), and "
            "sibling_confirmed (whether an exact match shares this top-level ICD code).\n"
            "sibling_confirmed indicates whether an exact match for the same "
            "condition shares the same top-level ICD code (first 3 characters). "
            "If True, this match is more likely to be a valid sub-code within "
            "the same clinical category.\n"
            "Accept if the concept captures the same clinical meaning "
            "as the search term in the context of the condition.\n"
            "Reject if it refers to a different condition, wrong body site, "
            "wrong specificity, or unrelated finding.\n"
            "Comment: 2-5 word clinical rationale "
            "(e.g. 'exact anatomical match', 'wrong body site', 'different etiology').\n"
        )

    def _ai_review_response_schema(self) -> dict:
        return {
            "type": "OBJECT",
            "properties": {
                "verdicts": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "condition": {"type": "STRING"},
                            "t": {"type": "STRING"},
                            "id": {"type": "STRING"},
                            "v": {"type": "STRING", "enum": ["accept", "reject"]},
                            "comment": {"type": "STRING"},
                        },
                        "required": ["condition", "t", "id", "v", "comment"],
                    },
                }
            },
            "required": ["verdicts"],
        }

    def _ai_review_parse_verdict(self, v: dict) -> dict:
        return {
            "condition_name": v.get("condition", ""),
            "search_term": v.get("t", ""),
            self._TARGET_ID_COL: str(v.get("id", "")),
            "ai_verdict": v.get("v", "human review"),
            "ai_comment": v.get("comment", ""),
        }

    # -------------------------------------------------------------------
    # Summarize
    # -------------------------------------------------------------------

    @staticmethod
    def _summarize(
        df_input: pl.DataFrame,
        df_exact: pl.DataFrame,
        df_fuzzy: pl.DataFrame,
        df_matches: pl.DataFrame,
    ) -> pl.DataFrame:
        """Print matching stats and return per-condition term counts.

        All counts are (condition, search_term) pairs, not unique terms.
        """
        # Per-condition pair counts
        df_term_counts = (
            df_input.group_by("condition_name").agg(
                pl.len().alias("total_terms")
            )
        )

        all_matched_terms = set(df_matches["search_term"].unique().to_list())

        df_matched_counts = (
            df_input.filter(pl.col("search_term").is_in(all_matched_terms))
            .group_by("condition_name")
            .agg(pl.len().alias("matched_terms"))
        )

        df_term_counts = (
            df_term_counts.join(df_matched_counts, on="condition_name", how="left")
            .with_columns(pl.col("matched_terms").fill_null(0))
            .with_columns(
                (pl.col("total_terms") - pl.col("matched_terms")).alias("unmatched_terms")
            )
        )

        all_conditions = set(df_input["condition_name"].unique().to_list())
        conditions_with_any_match = set(df_matches["condition_name"].unique().to_list())
        conditions_no_match = all_conditions - conditions_with_any_match

        still_unmatched = df_input.filter(~pl.col("search_term").is_in(all_matched_terms))

        # Count by (condition, search_term) pairs throughout
        n_conditions = len(all_conditions)
        n_search_terms = len(df_input)
        exact_term_set = set(df_exact["search_term"].unique().to_list())
        fuzzy_term_set = (
            set(df_fuzzy["search_term"].unique().to_list())
            if len(df_fuzzy) > 0 else set()
        )
        fuzzy_only_set = fuzzy_term_set - exact_term_set
        # Count input pairs whose term landed in each bucket
        n_exact_pairs = len(df_input.filter(
            pl.col("search_term").is_in(list(exact_term_set))
        ))
        n_fuzzy_pairs = len(df_input.filter(
            pl.col("search_term").is_in(list(fuzzy_only_set))
        ))
        n_matched_pairs = n_exact_pairs + n_fuzzy_pairs
        n_unmatched_pairs = len(still_unmatched)
        n_cond_matched = len(conditions_with_any_match)
        n_cond_no_match = len(conditions_no_match)

        n_exact_hits = len(df_exact)
        n_fuzzy_hits = len(df_fuzzy) if len(df_fuzzy) > 0 else 0
        n_concept_hits = n_exact_hits + n_fuzzy_hits

        n_icd = df_matches["icd_concept_id"].drop_nulls().n_unique()
        n_total_matches = len(df_matches.filter(pl.col("icd_concept_id").is_not_null()))

        print(f"\n{'=' * 40}")
        print(f"  MAPPING SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Input: {n_conditions} conditions -> {n_search_terms} search terms")
        print(f"  Matched: {n_exact_pairs} exact + {n_fuzzy_pairs} fuzzy "
              f"= {n_matched_pairs} terms ({n_unmatched_pairs} unmatched)")
        print(f"  Conditions with >=1 match: {n_cond_matched}")
        print(f"  Conditions with 0 matches: {n_cond_no_match}")
        print(f"")
        print(f"  Concept matches (each term can match multiple concepts):")
        print(f"    Exact: {n_exact_pairs} terms -> {n_exact_hits} matches")
        print(f"    Fuzzy: {n_fuzzy_pairs} terms -> {n_fuzzy_hits} matches")
        print(f"    Total: {n_matched_pairs} terms -> {n_concept_hits} matches")
        print(f"")
        print(f"  ICD concepts: {n_icd} unique "
              f"({n_total_matches} total matches)")
        print(f"{'=' * 40}")

        return df_term_counts

    # -------------------------------------------------------------------
    # Text normalization for CDC / fuzzy matching
    # -------------------------------------------------------------------

    @staticmethod
    def _normalize_for_fuzzy(text: str) -> str:
        """Normalize text for fuzzy matching, matching _build_input conventions.

        Applies: strip accents, lowercase, hyphen→space, strip possessives ('s),
        strip parentheticals, collapse whitespace.

        Note: Smart/curly quotes are normalized once at the input entry point
        (_build_input), so they are already straight quotes by the time text
        reaches this function. OMOP and CDC data use straight quotes natively.
        """
        t = strip_accents(text)
        t = t.lower()
        t = t.replace("-", " ")
        t = re.sub(r"'s\b", "", t)
        t = re.sub(r"\s*\([^)]*\)", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    # -------------------------------------------------------------------
    # Step 4: ICD-CM index lookup (ICD-10 + ICD-9 via CDC indices)
    # -------------------------------------------------------------------

    def _icdcm_lookup(
        self,
        results: dict,
        fuzzy_threshold: int = 85,
        icd9cm_index_path: Optional[str] = None,
        icd10cm_index_path: Optional[str] = None,
        auto_threshold: bool = True,
        auto_threshold_max: int = 90,
    ) -> dict:
        """Search all input terms against CDC ICD-CM indices.

        Searches every input term against both ICD-10-CM and ICD-9-CM CDC
        indices.  CDC exact hits (score 100) get ``match_type="exact"``,
        fuzzy hits get ``match_type="fuzzy"``.  Results are appended to
        ``df_matches``.

        Parameters
        ----------
        results : dict
            Pipeline results containing df_matches, df_input, df_exact, df_fuzzy.
        fuzzy_threshold : int
            Minimum fuzzy score (0-100) for CDC matching. Default 85.
        icd9cm_index_path : str, optional
            Path to pre-extracted ICD-9-CM text file (offline mode).
        icd10cm_index_path : str, optional
            Path to pre-extracted ICD-10-CM XML file (offline mode).

        Returns
        -------
        dict
            Updated results with CDC matches appended to df_matches.
        """
        from tctk.cdc import CDCIndex, CDCIndex9

        cdc10 = CDCIndex(
            xml_path=icd10cm_index_path,
            normalize_fn=self._normalize_for_fuzzy,
        )
        cdc9 = CDCIndex9(
            txt_path=icd9cm_index_path,
            normalize_fn=self._normalize_for_fuzzy,
        )

        df_matches = results["df_matches"]
        df_input = results["df_input"]

        # Search ALL input terms against both CDC indices
        all_pairs = list(
            zip(
                df_input["condition_name"].to_list(),
                df_input["search_term"].to_list(),
            )
        )
        unique_terms = sorted(df_input["search_term"].unique().to_list())
        print(f"  Searching {len(all_pairs)} search terms "
              f"({len(unique_terms)} unique) against CDC indices "
              f"(threshold={fuzzy_threshold})...")

        # Lookup all unique terms against both indices
        term_results_10: dict[str, list[dict]] = {}
        term_results_9: dict[str, list[dict]] = {}

        for term in tqdm(unique_terms, desc="ICD-CM lookup"):
            hits10 = cdc10.fuzzy_lookup(
                term,
                threshold=fuzzy_threshold,
                normalize_fn=self._normalize_for_fuzzy,
                stopwords=self._FUZZY_STOPWORDS,
            )
            if hits10:
                term_results_10[term] = hits10

            hits9 = cdc9.fuzzy_lookup(
                term,
                threshold=fuzzy_threshold,
                normalize_fn=self._normalize_for_fuzzy,
                stopwords=self._FUZZY_STOPWORDS,
            )
            if hits9:
                term_results_9[term] = hits9

        # Build new match rows
        new_match_rows = []
        for condition_name, search_term in sorted(all_pairs):
            # ICD-10 matches
            hits10 = term_results_10.get(search_term)
            if hits10:
                for hit in hits10:
                    code = hit["code"]
                    name = hit["name"]
                    score = hit["score"]
                    new_match_rows.append({
                        "condition_name": condition_name,
                        "search_term": search_term,
                        "matched_concept_synonym": name,
                        "concept_id": None,
                        "concept_code": code,
                        "concept_name": name,
                        "vocabulary_id": "ICD10CM",
                        "concept_class_id": None,
                        "standard_concept": None,
                        "match_type": "exact" if score == 100 else "fuzzy",
                        "match_score": score,
                        "icd_concept_id": code,
                        "icd_concept_name": name,
                        "icd_code": code,
                        "icd_version": "10",
                        "top_level_code": code[:3] if len(code) >= 3 else code,
                        "has_confirmed_sibling": False,
                    })

            # ICD-9 matches
            hits9 = term_results_9.get(search_term)
            if hits9:
                for hit in hits9:
                    code = hit["code"]
                    name = hit["name"]
                    score = hit["score"]
                    new_match_rows.append({
                        "condition_name": condition_name,
                        "search_term": search_term,
                        "matched_concept_synonym": name,
                        "concept_id": None,
                        "concept_code": code,
                        "concept_name": name,
                        "vocabulary_id": "ICD9CM",
                        "concept_class_id": None,
                        "standard_concept": None,
                        "match_type": "exact" if score == 100 else "fuzzy",
                        "match_score": score,
                        "icd_concept_id": code,
                        "icd_concept_name": name,
                        "icd_code": code,
                        "icd_version": "9",
                        "top_level_code": code[:3] if len(code) >= 3 else code,
                        "has_confirmed_sibling": False,
                    })

        n_new_icd10 = 0
        n_new_icd10_exact = 0
        n_new_icd9 = 0
        n_new_icd9_exact = 0

        if new_match_rows:
            df_new = pl.DataFrame(new_match_rows)
            df_new_10 = df_new.filter(pl.col("icd_version") == "10")
            df_new_9 = df_new.filter(pl.col("icd_version") == "9")
            n_new_icd10 = len(df_new_10)
            n_new_icd10_exact = len(df_new_10.filter(pl.col("match_score") == 100))
            n_new_icd9 = len(df_new_9)
            n_new_icd9_exact = len(df_new_9.filter(pl.col("match_score") == 100))

            df_matches = pl.concat([df_matches, df_new], how="diagonal_relaxed")

        # ---- Deduplicate across OMOP + CDC ----
        # Group by [condition_name, search_term, icd_concept_name, icd_code, icd_version]
        # Keep best: exact > fuzzy, then highest score
        # Sort: exact first (alphabetically "exact" < "fuzzy"), then score desc
        n_before_dedup = len(df_matches)
        df_matches = (
            df_matches
            .sort(["match_type", "match_score"], descending=[False, True])
            .unique(
                subset=["condition_name", "search_term", "icd_concept_name",
                        "icd_code", "icd_version"],
                keep="first",
            )
        )
        n_deduped = n_before_dedup - len(df_matches)

        # ---- Recompute has_confirmed_sibling from all exact matches ----
        confirmed_top_codes = (
            df_matches.filter(pl.col("match_type") == "exact")
            .select("condition_name", "top_level_code")
            .unique()
            .with_columns(pl.lit(True).alias("_has_sibling"))
        )
        df_matches = (
            df_matches.drop("has_confirmed_sibling")
            .join(confirmed_top_codes, on=["condition_name", "top_level_code"], how="left")
            .with_columns(
                pl.col("_has_sibling").fill_null(False).alias("has_confirmed_sibling")
            )
            .drop("_has_sibling")
        )

        # ---- Summary ----
        icd10_exact_note = f" ({n_new_icd10_exact} exact)" if n_new_icd10_exact > 0 else ""
        icd9_exact_note = f" ({n_new_icd9_exact} exact)" if n_new_icd9_exact > 0 else ""

        print(f"\n{'=' * 40}")
        print(f"  ICD-CM INDEX SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Terms searched:              {len(unique_terms)}")
        print(f"  ICD-10-CM matches:           {n_new_icd10}{icd10_exact_note}")
        print(f"  ICD-9-CM matches:            {n_new_icd9}{icd9_exact_note}")
        if n_deduped > 0:
            print(f"  Duplicates removed:          {n_deduped}")
        print(f"  Total after dedup:           {len(df_matches)}")
        print(f"{'=' * 40}")

        # ---- Adaptive threshold sweep ----
        if auto_threshold and fuzzy_threshold < 95:
            base_n_conds = df_matches["condition_name"].n_unique()

            sweep = []
            for thresh in range(fuzzy_threshold, 100, 5):
                df_t = df_matches.filter(
                    (pl.col("match_type") == "exact")
                    | (pl.col("match_score") >= thresh)
                )
                n_conds = df_t["condition_name"].n_unique()
                n_fuzzy = len(df_t.filter(
                    pl.col("match_type") == "fuzzy"
                ))
                sweep.append((thresh, n_conds, n_fuzzy))

            print("\n  Threshold sweep (pre-rescue counts):")
            print("  Thresh | Conds | Fuzzy | d Cond")
            rule = "─" * 8
            print(f"  {rule[:6]}─┼─{rule[:5]}─┼─{rule[:5]}─┼─{rule[:6]}")
            base_conds = sweep[0][1]
            for thresh, n_c, n_f in sweep:
                delta = n_c - base_conds
                d_str = f"{delta:+d}" if delta != 0 else "—"
                print(f"  {thresh:>6} | {n_c:>5} | {n_f:>5} | {d_str:>6}")

            # Auto-select: raise to cap, rescue conditions that
            # would lose all matches by keeping their root-level codes.
            max_auto = auto_threshold_max
            selected = fuzzy_threshold
            for thresh, n_c, n_f in sweep:
                if thresh <= max_auto:
                    selected = thresh

            if selected > fuzzy_threshold:
                n_total = len(df_matches)
                n_before = len(df_matches.filter(pl.col("match_type") == "fuzzy"))
                n_exact_total = n_total - n_before

                # Split into kept (above threshold / exact) and dropped
                df_kept = df_matches.filter(
                    (pl.col("match_type") == "exact")
                    | (pl.col("match_score") >= selected)
                )
                df_dropped = df_matches.filter(
                    (pl.col("match_type") != "exact")
                    & (pl.col("match_score") < selected)
                )

                # Build condition -> root-level kept ICD codes
                # Root = highest parent: remove any code whose parent is
                # also in the set.  E.g. {E10, E10.1, E10.12} -> {E10}
                # {E10.1, E10.12} -> {E10.1}
                # Descendant expansion later fills in children, so the
                # broadest code already in the data is what matters.
                raw_codes_by_cond: dict[str, set[str]] = {}
                for cond, code in (
                    df_kept.select("condition_name", "icd_code")
                    .unique().iter_rows()
                ):
                    if code is not None:
                        raw_codes_by_cond.setdefault(cond, set()).add(code)

                kept_codes_by_cond: dict[str, set[str]] = {}
                for cond, codes in raw_codes_by_cond.items():
                    roots = {
                        c for c in codes
                        if not any(
                            c.startswith(other) and c != other
                            for other in codes
                        )
                    }
                    kept_codes_by_cond[cond] = roots

                # Helper: find root codes (highest parents) in a set
                def _find_roots(codes: set[str]) -> set[str]:
                    return {
                        c for c in codes
                        if not any(
                            c.startswith(other) and c != other
                            for other in codes
                        )
                    }

                # Snapshot fuzzy count right after threshold filter (pre child-drop)
                n_thresh_fuzzy = len(df_kept.filter(
                    pl.col("match_type") == "fuzzy"
                ))

                # Drop children among kept codes — expansion covers them
                n_kept_before = len(df_kept)
                kept_mask = [
                    code is None or not any(
                        code.startswith(k) and code != k
                        for k in kept_codes_by_cond.get(cond, set())
                    )
                    for cond, code in zip(
                        df_kept["condition_name"].to_list(),
                        df_kept["icd_code"].to_list(),
                    )
                ]
                df_kept = df_kept.filter(pl.Series(kept_mask))
                n_children_dropped = n_kept_before - len(df_kept)
                # Track fuzzy children specifically for accurate reporting
                n_fuzzy_after_child = len(df_kept.filter(
                    pl.col("match_type") == "fuzzy"
                ))
                n_fuzzy_children = n_thresh_fuzzy - n_fuzzy_after_child

                # Rescue: for every condition, check dropped codes for
                # root-level codes NOT already covered by kept codes.
                # This brings back diverse ICD families that would
                # otherwise be lost at the higher threshold.
                dropped_codes_by_cond: dict[str, set[str]] = {}
                for cond, code in (
                    df_dropped.select("condition_name", "icd_code")
                    .unique().iter_rows()
                ):
                    if code is not None:
                        dropped_codes_by_cond.setdefault(
                            cond, set()
                        ).add(code)

                # For each condition, find dropped roots not covered
                # by any kept root (i.e. new ICD families)
                rescue_roots_by_cond: dict[str, set[str]] = {}
                for cond, dropped_codes in dropped_codes_by_cond.items():
                    dropped_roots = _find_roots(dropped_codes)
                    kept_roots = kept_codes_by_cond.get(cond, set())
                    # A dropped root is "new" if no kept root is its
                    # prefix (same family) and it's not a child of one
                    novel = {
                        r for r in dropped_roots
                        if not any(
                            r.startswith(k) or k.startswith(r)
                            for k in kept_roots
                        )
                    }
                    if novel:
                        rescue_roots_by_cond[cond] = novel

                df_rescue = pl.DataFrame()
                if rescue_roots_by_cond:
                    rescue_conds = list(rescue_roots_by_cond.keys())
                    df_rescue_pool = df_dropped.filter(
                        pl.col("condition_name").is_in(rescue_conds)
                    )
                    # Keep only rows whose code is a rescue root
                    rescue_mask = [
                        code is not None and code in rescue_roots_by_cond.get(
                            cond, set()
                        )
                        for cond, code in zip(
                            df_rescue_pool["condition_name"].to_list(),
                            df_rescue_pool["icd_code"].to_list(),
                        )
                    ]
                    df_rescue = df_rescue_pool.filter(pl.Series(rescue_mask))

                    # De-dup rescue: keep only the single best-scoring row
                    # per (condition, ICD family).  Family = 3-char prefix
                    # (e.g. E11, M30, K50) so we rescue at most one
                    # representative per chapter-level family per condition.
                    if len(df_rescue) > 0:
                        df_rescue = (
                            df_rescue
                            .with_columns(
                                pl.col("icd_code")
                                .str.slice(0, 3)
                                .alias("_icd_family")
                            )
                            .sort("match_score", descending=True)
                            .unique(
                                subset=["condition_name", "_icd_family"],
                                keep="first",
                            )
                            .drop("_icd_family")
                        )

                # Mark rescued rows before combining
                df_kept = df_kept.with_columns(
                    pl.lit(False).alias("is_rescued")
                )
                if len(df_rescue) > 0:
                    df_rescue = df_rescue.with_columns(
                        pl.lit(True).alias("is_rescued")
                    )

                # Combine
                df_matches = df_kept
                if len(df_rescue) > 0:
                    df_matches = pl.concat(
                        [df_kept, df_rescue], how="diagonal_relaxed"
                    )

                # Rebuild rescue_roots from actual rescued rows (post de-dup)
                actual_rescue_codes: dict[str, set[str]] = {}
                if len(df_rescue) > 0:
                    for cond, code in (
                        df_rescue.select("condition_name", "icd_code")
                        .unique().iter_rows()
                    ):
                        if code is not None:
                            actual_rescue_codes.setdefault(
                                cond, set()
                            ).add(code)

                # Add root_codes column for inspection
                merged_roots: dict[str, set[str]] = {}
                for cond in set(
                    list(kept_codes_by_cond) + list(actual_rescue_codes)
                ):
                    merged_roots[cond] = (
                        kept_codes_by_cond.get(cond, set())
                        | actual_rescue_codes.get(cond, set())
                    )
                root_labels = {
                    cond: ", ".join(sorted(roots))
                    for cond, roots in merged_roots.items()
                }
                df_matches = df_matches.with_columns(
                    pl.col("condition_name")
                    .replace(root_labels, default="")
                    .alias("root_codes")
                )

                n_thresh_removed = n_before - n_thresh_fuzzy
                n_rescue_rows = len(df_rescue)
                n_after = len(df_matches.filter(pl.col("match_type") == "fuzzy"))

                print(f"\n  Auto-threshold: {fuzzy_threshold} -> {selected}"
                      f"  ({n_exact_total} exact + {n_before} fuzzy"
                      f" = {n_total} total)")
                print(f"    Fuzzy removed by threshold:  {n_thresh_removed}"
                      f"  ({n_before} -> {n_thresh_fuzzy})")
                if n_children_dropped > 0:
                    n_exact_children = n_children_dropped - n_fuzzy_children
                    print(f"    Child codes dropped:         {n_children_dropped}"
                          f"  ({n_exact_children} exact + {n_fuzzy_children} fuzzy)")
                if n_rescue_rows > 0:
                    n_rescue_conds = df_rescue["condition_name"].n_unique()
                    n_rescue_codes = df_rescue["icd_code"].n_unique()
                    print(f"    Rescued back (novel ICD):   +{n_rescue_rows} matches"
                          f"  ({n_rescue_conds} conditions, "
                          f"{n_rescue_codes} codes)")
                print(f"    Net fuzzy for AI review:     {n_after}"
                      f"  (= {n_thresh_fuzzy} - {n_fuzzy_children} + {n_rescue_rows})")
            else:
                print(f"\n  Auto-threshold: already at cap. "
                      f"Keeping threshold={fuzzy_threshold}.")

        results["df_matches"] = df_matches
        return results

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def map(
        self,
        conditions: dict[str, list[str]],
        fuzzy_threshold: int = 85,
        icdcm_lookup: bool = True,
        auto_threshold: bool = True,
        auto_threshold_max: int = 90,
        icd9cm_index_path: Optional[str] = None,
        icd10cm_index_path: Optional[str] = None,
        ai_review: bool = False,
        gemini_api_key: Optional[str] = None,
        ai_tier: str = "flash",
        ai_min_version: float = 3.0,
        config_path: Optional[str] = None,
        ai_batch_size: Optional[int] = None,
        ai_passes: int = 2,
        export_tsv: bool = False,
        export_prefix: str = "mapping",
    ) -> dict:
        """Map condition names and their synonyms to OMOP ICD Concept IDs.

        Parameters
        ----------
        conditions : dict[str, list[str]]
            Keys are condition names; values are lists of condition synonyms.
        fuzzy_threshold : int
            Minimum score (0-100) for rapidfuzz token_sort_ratio.
            Shared across OMOP and CDC index matching.  Default 85.
        icdcm_lookup : bool
            If True (default), search all input terms against the CDC
            ICD-10-CM and ICD-9-CM indices after OMOP matching, then
            deduplicate across both sources.
        auto_threshold : bool
            When True (default) and ``icdcm_lookup`` is enabled, perform
            a post-hoc threshold sweep after combining OMOP + CDC
            results. Auto-raises the threshold up to ``auto_threshold_max``
            to reduce fuzzy matches sent to AI review.  Dropped codes
            that add diversity (parents or unrelated codes) are rescued;
            only children of already-kept codes are discarded.
        auto_threshold_max : int
            Ceiling for the auto-threshold sweep (default 90).  The sweep
            will raise the effective threshold up to this value.
        icd9cm_index_path : str, optional
            Path to pre-extracted ICD-9-CM text file for offline mode.
        icd10cm_index_path : str, optional
            Path to pre-extracted ICD-10-CM XML file for offline mode.
        ai_review : bool
            If True, run AI review of fuzzy matches via Gemini API.
            Default False.
        gemini_api_key : str, optional
            Gemini API key for AI review. Falls back to key set via
            :meth:`set_api_key`, then env var, then config file.
        ai_tier : str
            Preferred Gemini model tier: "pro", "flash", or "flash-lite".
            Default "flash".
        ai_min_version : float
            Minimum Gemini model version. Default 3.0 (prefer Gemini 3.x+).
            Set to 2.5 to allow older models (e.g. gemini-2.5-flash).
        config_path : str, optional
            Path to JSON config file for API key.
        ai_batch_size : int, optional
            Conditions per AI review API call. If None, auto-calculated.
        ai_passes : int
            Number of initial AI review passes. Default 2.
            Uses adaptive replication: 2 initial passes, then up to 5
            for disagreements. Set to 1 for single-pass mode.
        export_tsv : bool
            If True, write three TSV files:
            ``{export_prefix}_full.tsv`` — complete review table (1 row
            per match pair, the most comprehensive output).
            ``{export_prefix}_accepted.tsv`` — grouped accepted matches.
            ``{export_prefix}_rejected.tsv`` — rejected, human review,
            and unmatched rows.
            Default False.
        export_prefix : str
            Filename prefix for exported TSV files. Default "mapping".

        Returns
        -------
        dict with keys:
            df_review : pl.DataFrame
                One row per match pair with columns: condition_name,
                search_term, matched_concept_synonym, icd_concept_name,
                icd_code, icd_version, icd_concept_id, vocabulary_id,
                standard_concept (bool), match_type ("exact" or "fuzzy"),
                top_level_code (first 3 chars of icd_code),
                has_confirmed_sibling (bool — True when a fuzzy match's
                top-level code also appears in an exact match for the same
                condition), fuzzy_score, ai_verdict, ai_vote,
                ai_vote_confidence, ai_comment, ai_comment_consistency,
                ai_comment_consistency_tier, ai_combined_confidence.
                Unmatched conditions appear with null ICD columns and
                ai_verdict="no match".
            df_accepted : pl.DataFrame
                Grouped table with max 2 rows per condition (one ICD-9,
                one ICD-10). Columns: condition_name, icd_version,
                icd_codes, top_level_codes, icd_concept_names,
                icd_concept_ids, n_codes. Code columns are comma-separated
                aggregations of unique values.
            df_human_review : pl.DataFrame
                Matches needing manual verification: unreviewed fuzzy
                matches (no AI verdict) and AI verdicts flagged as
                "human review" (low confidence).
            df_rejected : pl.DataFrame
                Flat table (1 row per pair) where ai_verdict is "reject",
                "human review", or "no match". Includes top_level_code
                and has_confirmed_sibling columns.
            df_unmatched_terms : pl.DataFrame
                Search terms that found no match (exact or fuzzy).
                Null ICD columns, ai_verdict="no match". Includes
                terms from partially-matched conditions.
            df_unmapped_conditions : pl.DataFrame
                All rows for conditions with no usable ICD mapping —
                either zero matches or all matches rejected by AI.
                Includes both "no match" and "reject" rows for
                manual investigation.
            df_term_counts : pl.DataFrame
                Per-condition matching coverage (total/matched/unmatched terms).
            df_input, df_exact, df_fuzzy : pl.DataFrame
                Intermediate DataFrames from the matching pipeline.
        """
        do_ai_review = ai_review
        if do_ai_review:
            from tctk._utils import load_api_key
            resolved_key = gemini_api_key or self._api_key or load_api_key(config_path=config_path)
            if not resolved_key:
                print("  Warning: ai_review=True but no Gemini API key found. "
                      "Skipping AI review.\n"
                      "  Set a key via: mapper.set_api_key(key='...') or "
                      "mapper.set_api_key(key_file='path.json')\n"
                      "  Get a free key at: https://aistudio.google.com/apikey")
                do_ai_review = False
        n_steps = 2 + (1 if icdcm_lookup else 0) + (1 if do_ai_review else 0)
        step = 0

        # Capture all print output for later replay via print_summary()
        _log = _capture_prints()
        _buf = _log.__enter__()

        step += 1
        print(f"\033[1m[{step}/{n_steps}] Building search terms...\033[0m")
        df_input = self._build_input(conditions)
        print(
            f"  Conditions: {df_input['condition_name'].n_unique()}, "
            f"Search terms: {len(df_input)}"
        )

        step += 1
        print(f"\n\033[1m[{step}/{n_steps}] Matching against vocabulary...\033[0m")
        df_exact, df_fuzzy = self._omop_lookup(df_input, vocab="ICD", fuzzy_threshold=fuzzy_threshold)

        # Combine exact + fuzzy into single DataFrame
        df_matches = pl.concat([df_exact, df_fuzzy], how="diagonal_relaxed")
        df_matches = df_matches.with_columns(pl.col("concept_id").cast(pl.Utf8))
        # Alias columns for ICD
        df_matches = df_matches.with_columns(
            pl.col("concept_id").alias("icd_concept_id"),
            pl.col("concept_name").alias("icd_concept_name"),
            pl.col("concept_code").alias("icd_code"),
            pl.when(pl.col("vocabulary_id") == "ICD9CM")
            .then(pl.lit("9"))
            .when(pl.col("vocabulary_id") == "ICD10CM")
            .then(pl.lit("10"))
            .otherwise(pl.col("vocabulary_id"))
            .alias("icd_version"),
        )

        # Top-level ICD code (first 3 chars, e.g. "K51", "E10")
        df_matches = df_matches.with_columns(
            pl.col("icd_code").str.slice(0, 3).alias("top_level_code")
        )

        # has_confirmed_sibling is computed later (step 4 or before AI review)
        df_matches = df_matches.with_columns(
            pl.lit(False).alias("has_confirmed_sibling")
        )

        if not do_ai_review and fuzzy_threshold < 85 and len(df_fuzzy) > 0:
            print(f"\n  Warning: AI review is off and fuzzy_threshold={fuzzy_threshold}. "
                  f"Low-score fuzzy matches won't be vetted. "
                  f"Consider fuzzy_threshold >= 85 or enabling ai_review=True.")

        df_term_counts = self._summarize(df_input, df_exact, df_fuzzy, df_matches)

        results = {
            "df_input": df_input,
            "df_exact": df_exact,
            "df_fuzzy": df_fuzzy,
            "df_matches": df_matches,
            "df_term_counts": df_term_counts,
        }

        if icdcm_lookup:
            step += 1
            print(f"\n\033[1m[{step}/{n_steps}] ICD-CM index lookup...\033[0m")
            results = self._icdcm_lookup(
                results,
                fuzzy_threshold=fuzzy_threshold,
                icd9cm_index_path=icd9cm_index_path,
                icd10cm_index_path=icd10cm_index_path,
                auto_threshold=auto_threshold,
                auto_threshold_max=auto_threshold_max,
            )
            df_matches = results["df_matches"]
        else:
            # Compute has_confirmed_sibling from exact matches only
            exact_top_codes = (
                df_matches
                .filter(pl.col("match_type") == "exact")
                .select("condition_name", "top_level_code")
                .unique()
                .with_columns(pl.lit(True).alias("_has_sibling"))
            )
            df_matches = (
                df_matches.drop("has_confirmed_sibling")
                .join(exact_top_codes, on=["condition_name", "top_level_code"], how="left")
                .with_columns(
                    pl.col("_has_sibling").fill_null(False).alias("has_confirmed_sibling")
                )
                .drop("_has_sibling")
            )
            results["df_matches"] = df_matches

        if do_ai_review:
            step += 1
            print(f"\n\033[1m[{step}/{n_steps}] AI review...\033[0m")
            results = self.ai_review(
                results,
                batch_size=ai_batch_size,
                gemini_api_key=gemini_api_key,
                ai_tier=ai_tier,
                ai_min_version=ai_min_version,
                config_path=config_path,
                ai_passes=ai_passes,
            )

        # --- Build df_review: one row per match pair, flat columns ---
        df_final = results["df_matches"]
        has_ai_cols = "ai_verdict" in df_final.columns

        # Recompute has_confirmed_sibling using all confirmed matches
        # (exact + AI-accepted)
        if has_ai_cols:
            confirmed_filter = (
                (pl.col("match_type") == "exact")
                | (pl.col("ai_verdict") == "accept")
            )
            confirmed_top_codes = (
                df_final.filter(confirmed_filter)
                .select("condition_name", "top_level_code")
                .unique()
                .with_columns(pl.lit(True).alias("_has_sibling"))
            )
            df_final = (
                df_final.drop("has_confirmed_sibling")
                .join(confirmed_top_codes, on=["condition_name", "top_level_code"], how="left")
                .with_columns(
                    pl.col("_has_sibling").fill_null(False).alias("has_confirmed_sibling")
                )
                .drop("_has_sibling")
            )

        review_cols = {
            "condition_name": pl.col("condition_name"),
            "search_term": pl.col("search_term"),
            "matched_concept_synonym": pl.col("matched_concept_synonym"),
            "icd_concept_name": pl.col("icd_concept_name"),
            "icd_code": pl.col("icd_code"),
            "icd_version": pl.col("icd_version"),
            "icd_concept_id": pl.col("icd_concept_id"),
            "vocabulary_id": pl.col("vocabulary_id"),
            "standard_concept": (pl.col("standard_concept") == "S"),
            "match_type": pl.col("match_type"),
            "top_level_code": pl.col("top_level_code"),
            "has_confirmed_sibling": pl.col("has_confirmed_sibling"),
            "fuzzy_score": pl.col("match_score"),
        }

        df_review = df_final.select(**review_cols)

        # Add root_codes and is_rescued if present (from auto_threshold)
        if "root_codes" in df_final.columns:
            df_review = df_review.with_columns(df_final["root_codes"])
        if "is_rescued" in df_final.columns:
            df_review = df_review.with_columns(df_final["is_rescued"])
        else:
            df_review = df_review.with_columns(
                pl.lit(False).alias("is_rescued")
            )

        # Add AI columns if present, otherwise nulls
        if has_ai_cols:
            ai_col_names = [
                "ai_verdict", "ai_vote", "ai_vote_confidence",
                "ai_comment", "ai_comment_consistency",
                "ai_comment_consistency_tier", "ai_combined_confidence",
            ]
            ai_cols = [df_final[c] for c in ai_col_names if c in df_final.columns]
            if ai_cols:
                df_review = df_review.with_columns(*ai_cols)
        else:
            df_review = df_review.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("ai_verdict"),
                pl.lit(None).cast(pl.Utf8).alias("ai_vote"),
                pl.lit(None).cast(pl.Utf8).alias("ai_vote_confidence"),
                pl.lit(None).cast(pl.Utf8).alias("ai_comment"),
                pl.lit(None).cast(pl.Int64).alias("ai_comment_consistency"),
                pl.lit(None).cast(pl.Utf8).alias("ai_comment_consistency_tier"),
                pl.lit(None).cast(pl.Utf8).alias("ai_combined_confidence"),
            )

        # Append "no match" rows for search terms that didn't match anything,
        # including terms from conditions that have other (possibly rejected) matches.
        matched_pairs = set(
            zip(
                df_review["condition_name"].to_list(),
                df_review["search_term"].to_list(),
            )
        )
        all_pairs = set(
            zip(
                df_input["condition_name"].to_list(),
                df_input["search_term"].to_list(),
            )
        )
        unmatched_pairs = all_pairs - matched_pairs

        if unmatched_pairs:
            unmatched_rows = [
                {
                    "condition_name": cond,
                    "search_term": term,
                    "matched_concept_synonym": None,
                    "icd_concept_name": None,
                    "icd_code": None,
                    "icd_version": None,
                    "icd_concept_id": None,
                    "vocabulary_id": None,
                    "standard_concept": None,
                    "match_type": None,
                    "top_level_code": None,
                    "has_confirmed_sibling": None,
                    "fuzzy_score": None,
                    "ai_verdict": "no match",
                    "ai_vote": None,
                    "ai_vote_confidence": None,
                    "ai_comment": None,
                    "ai_comment_consistency": None,
                    "ai_comment_consistency_tier": None,
                    "ai_combined_confidence": None,
                    "is_rescued": False,
                }
                for cond, term in unmatched_pairs
            ]
            df_unmatched_terms = pl.DataFrame(unmatched_rows, schema=df_review.schema)
            df_review = pl.concat([df_review, df_unmatched_terms], how="diagonal_relaxed")
        else:
            df_unmatched_terms = pl.DataFrame(schema=df_review.schema)

        # Build grouped accepted table (max 2 rows per condition: ICD-9 + ICD-10)
        # Accepted = exact + ai-accepted + unreviewed fuzzy
        df_acc_flat = df_review.filter(
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
        df_accepted = (
            df_acc_flat
            .group_by(["condition_name", "icd_version"])
            .agg(
                pl.col("icd_code").unique().sort().str.join(", ").alias("icd_codes"),
                pl.col("top_level_code").unique().sort().str.join(", ").alias("top_level_codes"),
                pl.col("icd_concept_name").unique().sort().str.join(", ").alias("icd_concept_names"),
                pl.col("icd_concept_id").unique().sort().str.join(", ").alias("icd_concept_ids"),
                pl.col("icd_code").n_unique().alias("n_codes"),
            )
            .sort(["condition_name", "icd_version"])
        )

        # Rejected table stays flat (1 row per pair)
        # Excludes "human review" — those go in df_human_review only
        df_rejected = df_review.filter(
            pl.col("ai_verdict").is_in(["reject", "no match"])
        )

        # Human review table: matches that need manual verification
        # - Unreviewed fuzzy matches (no AI verdict)
        # - AI verdicts flagged as "human review" (low confidence)
        df_human_review = df_review.filter(
            (pl.col("ai_verdict") == "human review")
            | (
                pl.col("icd_code").is_not_null()
                & (pl.col("match_type") == "fuzzy")
                & pl.col("ai_verdict").is_null()
            )
        )

        # Conditions with no usable ICD mapping (no match or all rejected)
        mapped_conds = set(
            df_review.filter(
                pl.col("icd_code").is_not_null()
                & (
                    (pl.col("match_type") == "exact")
                    | (pl.col("ai_verdict") == "accept")
                    | (
                        (pl.col("match_type") == "fuzzy")
                        & pl.col("ai_verdict").is_null()
                    )
                )
            )["condition_name"].to_list()
        )
        all_conds = set(df_review["condition_name"].to_list())
        unmapped_conds = all_conds - mapped_conds
        df_unmapped_conditions = df_review.filter(pl.col("condition_name").is_in(list(unmapped_conds)))

        results["df_review"] = df_review
        results["df_accepted"] = df_accepted
        results["df_rejected"] = df_rejected
        results["df_human_review"] = df_human_review
        results["df_unmatched_terms"] = df_unmatched_terms
        results["df_unmapped_conditions"] = df_unmapped_conditions

        # Remove old keys no longer produced
        results.pop("df_matches", None)

        # --- Final summary ---
        n_conditions = len(conditions)
        n_mapped = df_accepted["condition_name"].n_unique() if len(df_accepted) > 0 else 0
        n_unmapped = n_conditions - n_mapped

        n_exact = len(df_review.filter(pl.col("match_type") == "exact"))
        n_fuzzy = len(df_review.filter(pl.col("match_type") == "fuzzy"))
        n_no_match = len(df_review.filter(pl.col("ai_verdict") == "no match"))
        n_human_review = len(df_human_review)

        # Rescue counts per category
        has_rescue = "is_rescued" in df_review.columns
        _rescue_filter = (pl.col("is_rescued") == True) if has_rescue else pl.lit(False)
        n_rescued_accepted = len(df_acc_flat.filter(_rescue_filter)) if has_rescue else 0
        n_rescued_human = len(df_human_review.filter(_rescue_filter)) if has_rescue else 0
        n_rescued_rejected = len(df_rejected.filter(
            _rescue_filter & (pl.col("ai_verdict") != "no match")
        )) if has_rescue else 0
        n_rescued_total = n_rescued_accepted + n_rescued_human + n_rescued_rejected

        def _rescue_note(n):
            return f" [{n} rescued]" if n > 0 else ""

        print(f"\n{'=' * 40}")
        print(f"  FINAL SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Conditions: {n_mapped} mapped, {n_unmapped} unmapped (of {n_conditions} total)")
        print(f"  Accepted:     {len(df_acc_flat)} matches{_rescue_note(n_rescued_accepted)}")
        print(f"  Human review: {n_human_review} matches{_rescue_note(n_rescued_human)}")
        print(f"  Rejected:     {len(df_rejected)} matches (incl. {n_no_match} no-match)"
              f"{_rescue_note(n_rescued_rejected)}")
        print(f"")
        print(f"  Match breakdown:")
        print(f"    Exact matches:             {n_exact}")
        print(f"    Fuzzy matches:             {n_fuzzy}")
        if n_rescued_total > 0:
            print(f"    Rescued matches:           {n_rescued_total} (included in fuzzy above)")

        # Hits per condition (only conditions with >=1 match)
        df_with_hits = df_review.filter(pl.col("icd_code").is_not_null())
        if len(df_with_hits) > 0:
            hits_per_cond = (
                df_with_hits.group_by("condition_name")
                .agg(pl.len().alias("n_hits"))
                ["n_hits"]
            )
            print(f"")
            print(f"  Hits per condition: "
                  f"min={hits_per_cond.min()}, "
                  f"max={hits_per_cond.max()}, "
                  f"median={hits_per_cond.median():.0f}, "
                  f"mean={hits_per_cond.mean():.1f}")

        if len(df_accepted) > 0:
            version_counts = df_accepted.group_by("icd_version").agg(
                pl.col("n_codes").sum().alias("total_codes"),
                pl.len().alias("n_condition_version_pairs"),
            ).sort("icd_version")
            print(f"")
            print(f"  Accepted ICD version breakdown:")
            for row in version_counts.iter_rows(named=True):
                label = f"ICD-{row['icd_version']}-CM"
                print(f"    {label}: {row['total_codes']} codes across "
                      f"{row['n_condition_version_pairs']} conditions")

        if has_ai_cols:
            reviewed = df_review.filter(
                pl.col("ai_verdict").is_not_null()
                & (pl.col("ai_verdict") != "no match")
            )
            if len(reviewed) > 0:
                verdicts = reviewed.group_by("ai_verdict").len().sort("ai_verdict")
                parts = [f"{r['ai_verdict']}={r['len']}" for r in verdicts.iter_rows(named=True)]
                print(f"  AI review:  {len(reviewed)} reviewed ({', '.join(parts)})")
                if "ai_combined_confidence" in reviewed.columns:
                    rel = reviewed.filter(pl.col("ai_combined_confidence").is_not_null()).group_by("ai_combined_confidence").len().sort("ai_combined_confidence")
                    if len(rel) > 0:
                        rel_parts = [f"{r['ai_combined_confidence']}={r['len']}" for r in rel.iter_rows(named=True)]
                        print(f"  AI reliability: {', '.join(rel_parts)}")
        print(f"{'=' * 40}")

        # --- Export ---
        if export_tsv:
            write_tsv_bom(df_review, f"{export_prefix}_full.tsv")
            write_tsv_bom(df_accepted, f"{export_prefix}_accepted.tsv")
            write_tsv_bom(df_human_review, f"{export_prefix}_human_review.tsv")
            write_tsv_bom(df_rejected, f"{export_prefix}_rejected.tsv")
            write_tsv_bom(df_unmatched_terms, f"{export_prefix}_unmatched_terms.tsv")
            write_tsv_bom(df_unmapped_conditions, f"{export_prefix}_unmapped_conditions.tsv")
            print(f"\nExported:")
            print(f"  {export_prefix}_full.tsv                  ({len(df_review)} matches)")
            print(f"  {export_prefix}_accepted.tsv              ({len(df_accepted)} matches)")
            print(f"  {export_prefix}_human_review.tsv          ({len(df_human_review)} matches)")
            print(f"  {export_prefix}_rejected.tsv              ({len(df_rejected)} matches)")
            print(f"  {export_prefix}_unmatched_terms.tsv       ({len(df_unmatched_terms)} terms)")
            print(f"  {export_prefix}_unmapped_conditions.tsv   ({len(df_unmapped_conditions)} conditions)")

        # Stop capturing and store the log
        _log.__exit__(None, None, None)
        results["_run_log"] = _buf.getvalue()

        return results
