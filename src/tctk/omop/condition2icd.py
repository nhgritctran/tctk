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
import json
import re
from typing import Optional

import polars as pl
from tqdm.auto import tqdm

from tctk._utils import (
    sql_escape,
    strip_accents,
    write_tsv_bom,
)
from tctk.omop._base import ConditionMapperBase

__all__ = ["Condition2ICD"]


# -------------------------------------------------------------------
# Print-capture utility
# -------------------------------------------------------------------

class _LogBuf:
    """Accumulate text alongside normal ``print()`` — no stdout hijack.

    Usage::

        log = _LogBuf()
        log.print("hello")       # prints AND records
        captured = log.getvalue()
    """

    def __init__(self):
        self._buf = io.StringIO()

    def print(self, *args, **kwargs):
        """Drop-in for ``print()`` that also records to internal buffer."""
        print(*args, **kwargs)
        kwargs.pop("file", None)
        kwargs.pop("flush", None)
        print(*args, file=self._buf, **kwargs)

    def getvalue(self):
        return self._buf.getvalue()


class Condition2ICD(ConditionMapperBase):
    """Map condition names and synonyms to OMOP ICD Concept IDs.

    Uses a local DuckDB vocabulary database built from Athena CSV files.
    No network access required for mapping — only for optional AI review.

    Args:
        vocab_db (str, optional): Path to the DuckDB vocabulary database.
            Default: auto-downloaded from Hugging Face
        force_download_db (bool): Force re-download of the vocabulary database. Default False.
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
        strip parentheticals/brackets, strip punctuation, collapse whitespace.

        Note: Smart/curly quotes are normalized once at the input entry point
        (_build_input), so they are already straight quotes by the time text
        reaches this function. OMOP and CDC data use straight quotes natively.
        """
        t = strip_accents(text)
        t = t.lower()
        t = t.replace("-", " ")
        t = re.sub(r"'s\b", "", t)
        t = re.sub(r"\s*\([^)]*\)", "", t)
        t = re.sub(r"\s*\[[^\]]*\]", "", t)      # strip []
        for ch in ',:;"/':
            t = t.replace(ch, " ")
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

        Args:
            results (dict): Pipeline results containing df_matches, df_input,
                df_exact, df_fuzzy.
            fuzzy_threshold (int): Minimum fuzzy score (0-100) for CDC matching.
                Default 85.
            icd9cm_index_path (str, optional): Path to pre-extracted ICD-9-CM text
                file (offline mode).
            icd10cm_index_path (str, optional): Path to pre-extracted ICD-10-CM XML
                file (offline mode).

        Returns:
            dict: Updated results with CDC matches appended to df_matches.
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
                    cdc_name = hit.get("cdc_name", hit["name"])
                    score = hit["score"]
                    new_match_rows.append({
                        "condition_name": condition_name,
                        "search_term": search_term,
                        "_cdc_matched_query": hit.get("matched_query", search_term),
                        "matched_concept_synonym": hit["name"],
                        "concept_id": None,
                        "concept_code": code,
                        "vocabulary_id": "ICD10CM",
                        "concept_class_id": None,
                        "standard_concept": None,
                        "match_type": "exact" if score == 100 else "fuzzy",
                        "match_score": score,
                        "icd_concept_id": code,
                        "icd_concept_name": cdc_name,
                        "icd_code": code,
                        "icd_version": "10",
                        "top_level_code": code[:3] if len(code) >= 3 else code,
                        "has_confirmed_sibling": False,
                        "match_source": "cdc",
                        "match_threshold": fuzzy_threshold,
                    })

            # ICD-9 matches
            hits9 = term_results_9.get(search_term)
            if hits9:
                for hit in hits9:
                    code = hit["code"]
                    cdc_name = hit.get("cdc_name", hit["name"])
                    score = hit["score"]
                    new_match_rows.append({
                        "condition_name": condition_name,
                        "search_term": search_term,
                        "_cdc_matched_query": hit.get("matched_query", search_term),
                        "matched_concept_synonym": hit["name"],
                        "concept_id": None,
                        "concept_code": code,
                        "vocabulary_id": "ICD9CM",
                        "concept_class_id": None,
                        "standard_concept": None,
                        "match_type": "exact" if score == 100 else "fuzzy",
                        "match_score": score,
                        "icd_concept_id": code,
                        "icd_concept_name": cdc_name,
                        "icd_code": code,
                        "icd_version": "9",
                        "top_level_code": code[:3] if len(code) >= 3 else code,
                        "has_confirmed_sibling": False,
                        "match_source": "cdc",
                        "match_threshold": fuzzy_threshold,
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
        dedup_key = ["condition_name", "search_term", "icd_concept_name",
                     "icd_code", "icd_version"]

        # Aggregate sources before dedup (e.g. "cdc+omop" when both matched)
        source_agg = (
            df_matches
            .group_by(dedup_key)
            .agg(pl.col("match_source").unique().sort().str.join("+").alias("_merged_source"))
        )

        n_before_dedup = len(df_matches)
        df_matches = (
            df_matches.drop("match_source")
            .sort(["match_type", "match_score"], descending=[False, True])
            .unique(subset=dedup_key, keep="first")
        )
        n_deduped = n_before_dedup - len(df_matches)

        # Join merged source back
        df_matches = df_matches.join(source_agg, on=dedup_key, how="left")
        df_matches = df_matches.rename({"_merged_source": "match_source"})

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

        # Replace search_term with processed query for CDC rows (display only).
        # Raw search_term was kept above for correct dedup grouping.
        if "_cdc_matched_query" in df_matches.columns:
            df_matches = df_matches.with_columns(
                pl.when(pl.col("_cdc_matched_query").is_not_null())
                .then(pl.col("_cdc_matched_query"))
                .otherwise(pl.col("search_term"))
                .alias("search_term")
            ).drop("_cdc_matched_query")

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
        fallback_step: int = 5,
        fallback_floor: int = 50,
        icd9cm_index_path: Optional[str] = None,
        icd10cm_index_path: Optional[str] = None,
        ai_review: bool = False,
        gemini_api_key: Optional[str] = None,
        ai_provider: str = "gemini",
        ai_tier: Optional[str] = None,
        ai_min_version: Optional[float] = None,
        config_path: Optional[str] = None,
        ai_batch_size: Optional[int] = None,
        ai_passes: int = 2,
        export_tsv: bool = False,
        export_prefix: str = "mapping",
    ) -> dict:
        """Map condition names and their synonyms to OMOP ICD Concept IDs.

        Args:
            conditions (dict[str, list[str]]): Keys are condition names; values are
                lists of condition synonyms.
            fuzzy_threshold (int): Minimum score (0-100) for rapidfuzz token_sort_ratio.
                Shared across OMOP and CDC index matching.  Default 85.
            icdcm_lookup (bool): If True (default), search all input terms against the
                CDC ICD-10-CM and ICD-9-CM indices after OMOP matching, then deduplicate
                across both sources.
            auto_threshold (bool): When True (default) and ``icdcm_lookup`` is enabled,
                perform a post-hoc threshold sweep after combining OMOP + CDC results.
                Auto-raises the threshold up to ``auto_threshold_max`` to reduce fuzzy
                matches sent to AI review.  Dropped codes that add diversity (parents or
                unrelated codes) are rescued; only children of already-kept codes are
                discarded.
            auto_threshold_max (int): Ceiling for the auto-threshold sweep (default 90).
                The sweep will raise the effective threshold up to this value.
            fallback_step (int): After the main pass, re-run OMOP + CDC lookup at
                progressively lower thresholds for conditions with zero hits.  Each
                iteration lowers the threshold by this amount (default 5).  Set to 0 to
                disable fallback.
            fallback_floor (int): Lowest threshold the fallback loop will try
                (default 50). Conditions still unmapped at the floor are reported as
                unmapped.
            icd9cm_index_path (str, optional): Path to pre-extracted ICD-9-CM text file
                for offline mode.
            icd10cm_index_path (str, optional): Path to pre-extracted ICD-10-CM XML file
                for offline mode.
            ai_review (bool): If True, run AI review of fuzzy matches. Default False.
            gemini_api_key (str, optional): Gemini API key for AI review. Falls back to
                key set via :meth:`set_api_key`, then env var, then config file.
            ai_provider (str): Primary AI provider: ``"gemini"`` or ``"claude"``.
                Default ``"gemini"``. If primary fails, auto-falls back to the other
                provider if its key is configured.
            ai_tier (str, optional): Preferred model tier. Auto-resolves per provider if
                None: Gemini → "pro", Claude → "sonnet".
                Gemini options: "pro"/"flash"/"flash-lite".
                Claude options: "opus"/"sonnet"/"haiku".
            ai_min_version (float, optional): Minimum model version. Gemini default 3.0,
                Claude default 4.6.
            config_path (str, optional): Path to JSON config file for API key.
            ai_batch_size (int, optional): Conditions per AI review API call. If None,
                auto-calculated.
            ai_passes (int): Number of initial AI review passes. Default 2.
                Uses adaptive replication: 2 initial passes, then up to 5 for
                disagreements. Set to 1 for single-pass mode.
            export_tsv (bool): If True, write three TSV files:
                ``{export_prefix}_full.tsv`` — complete review table (1 row per match
                pair, the most comprehensive output).
                ``{export_prefix}_accepted.tsv`` — grouped accepted matches.
                ``{export_prefix}_rejected.tsv`` — rejected, human review, and unmatched
                rows.
                Default False.
            export_prefix (str): Filename prefix for exported TSV files. Default "mapping".

        Returns:
            dict: Dictionary with keys:

                - ``df_review`` (pl.DataFrame) — One row per match pair with columns:
                  condition_name, search_term, matched_concept_synonym, icd_concept_name,
                  icd_code, icd_version, icd_concept_id, vocabulary_id,
                  standard_concept (bool), match_type ("exact" or "fuzzy"),
                  top_level_code (first 3 chars of icd_code),
                  has_confirmed_sibling (bool — True when a fuzzy match's top-level code
                  also appears in an exact match for the same condition), fuzzy_score,
                  ai_verdict, ai_vote, ai_vote_confidence, ai_comment,
                  ai_comment_consistency, ai_comment_consistency_tier,
                  ai_combined_confidence. Unmatched conditions appear with null ICD
                  columns and ai_verdict="no match".
                - ``df_accepted`` (pl.DataFrame) — Grouped table with max 2 rows per
                  condition (one ICD-9, one ICD-10). Columns: condition_name, icd_version,
                  icd_codes, top_level_codes, icd_concept_names, icd_concept_ids, n_codes.
                  Code columns are comma-separated aggregations of unique values.
                - ``df_human_review`` (pl.DataFrame) — Matches needing manual
                  verification: unreviewed fuzzy matches (no AI verdict) and AI verdicts
                  flagged as "human review" (low confidence).
                - ``df_rejected`` (pl.DataFrame) — Flat table (1 row per pair) where
                  ai_verdict is "reject", "human review", or "no match". Includes
                  top_level_code and has_confirmed_sibling columns.
                - ``df_unmatched_terms`` (pl.DataFrame) — Search terms that found no
                  match (exact or fuzzy). Null ICD columns, ai_verdict="no match".
                  Includes terms from partially-matched conditions.
                - ``df_unmapped_conditions`` (pl.DataFrame) — All rows for conditions
                  with no usable ICD mapping — either zero matches or all matches rejected
                  by AI. Includes both "no match" and "reject" rows for manual
                  investigation.
                - ``df_term_counts`` (pl.DataFrame) — Per-condition matching coverage
                  (total/matched/unmatched terms).
                - ``df_input``, ``df_exact``, ``df_fuzzy`` (pl.DataFrame) — Intermediate
                  DataFrames from the matching pipeline.
        """
        do_ai_review = ai_review
        if do_ai_review:
            if ai_provider == "claude":
                if not self._claude_api_key:
                    print("  Warning: ai_review=True with ai_provider='claude' "
                          "but no Claude API key found. Skipping AI review.\n"
                          "  Set a key via: mapper.set_api_key(claude_key='...') or "
                          "mapper.set_api_key(claude_key_file='path.json')")
                    do_ai_review = False
            else:
                from tctk._utils import load_api_key
                resolved_key = gemini_api_key or self._api_key or load_api_key(config_path=config_path)
                if not resolved_key:
                    print("  Warning: ai_review=True but no Gemini API key found. "
                          "Skipping AI review.\n"
                          "  Set a key via: mapper.set_api_key(key='...') or "
                          "mapper.set_api_key(key_file='path.json')\n"
                          "  Get a free key at: https://aistudio.google.com/apikey")
                    do_ai_review = False
        do_fallback = fallback_step > 0 and icdcm_lookup
        n_steps = 2 + (1 if icdcm_lookup else 0) + (1 if do_fallback else 0) + (1 if do_ai_review else 0)
        step = 0

        # Log all print output for later replay via print_summary()
        _log = _LogBuf()

        step += 1
        _log.print(f"\033[1m[{step}/{n_steps}] Building search terms...\033[0m")
        df_input = self._build_input(conditions)
        _log.print(
            f"  Conditions: {df_input['condition_name'].n_unique()}, "
            f"Search terms: {len(df_input)}"
        )

        step += 1
        _log.print(f"\n\033[1m[{step}/{n_steps}] Matching against vocabulary...\033[0m")
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

        # Tag OMOP rows
        df_matches = df_matches.with_columns(
            pl.lit("omop").alias("match_source"),
            pl.lit(fuzzy_threshold).alias("match_threshold"),
        )

        if not do_ai_review and fuzzy_threshold < 85 and len(df_fuzzy) > 0:
            _log.print(f"\n  Warning: AI review is off and fuzzy_threshold={fuzzy_threshold}. "
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
            _log.print(f"\n\033[1m[{step}/{n_steps}] ICD-CM index lookup...\033[0m")
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

        # ---- Fallback loop for unmapped conditions ----
        if do_fallback:
            step += 1
            all_conds = set(df_input["condition_name"].unique().to_list())
            matched_conds = set(df_matches["condition_name"].unique().to_list())
            unmapped_conds = all_conds - matched_conds

            fb_threshold = fuzzy_threshold - fallback_step
            fb_stats = []  # [(threshold, n_newly_matched)]

            if unmapped_conds and fb_threshold >= fallback_floor:
                _log.print(f"\n\033[1m[{step}/{n_steps}] Fallback for {len(unmapped_conds)} unmapped conditions...\033[0m")
            else:
                _log.print(f"\n\033[1m[{step}/{n_steps}] Fallback: no unmapped conditions, skipping.\033[0m")

            while unmapped_conds and fb_threshold >= fallback_floor:
                _log.print(f"\n  --- Fallback pass (threshold={fb_threshold}) ---")
                _log.print(f"  Unmapped conditions: {len(unmapped_conds)}")

                # Filter input to unmapped conditions only
                df_input_fb = df_input.filter(
                    pl.col("condition_name").is_in(list(unmapped_conds))
                )

                # OMOP lookup at lower threshold
                df_exact_fb, df_fuzzy_fb = self._omop_lookup(
                    df_input_fb, vocab="ICD", fuzzy_threshold=fb_threshold,
                )

                # Prepare OMOP fallback matches (same column setup as main pass)
                df_fb = pl.concat([df_exact_fb, df_fuzzy_fb], how="diagonal_relaxed")
                df_fb = df_fb.with_columns(pl.col("concept_id").cast(pl.Utf8))
                df_fb = df_fb.with_columns(
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
                df_fb = df_fb.with_columns(
                    pl.col("icd_code").str.slice(0, 3).alias("top_level_code"),
                    pl.lit(False).alias("has_confirmed_sibling"),
                    pl.lit("omop").alias("match_source"),
                    pl.lit(fb_threshold).alias("match_threshold"),
                )

                # CDC lookup at lower threshold (no auto-threshold)
                fb_results = {
                    "df_input": df_input_fb,
                    "df_exact": df_exact_fb,
                    "df_fuzzy": df_fuzzy_fb,
                    "df_matches": df_fb,
                }
                fb_results = self._icdcm_lookup(
                    fb_results,
                    fuzzy_threshold=fb_threshold,
                    icd9cm_index_path=icd9cm_index_path,
                    icd10cm_index_path=icd10cm_index_path,
                    auto_threshold=False,
                )
                df_fb = fb_results["df_matches"]

                # Count newly matched conditions
                if len(df_fb) > 0:
                    new_matched = set(df_fb["condition_name"].unique().to_list()) & unmapped_conds
                    if new_matched:
                        fb_stats.append((fb_threshold, len(new_matched)))
                        df_matches = pl.concat(
                            [df_matches, df_fb], how="diagonal_relaxed"
                        )
                        unmapped_conds -= new_matched
                        _log.print(f"  Matched {len(new_matched)} new conditions")
                    else:
                        _log.print(f"  No new conditions matched")
                else:
                    _log.print(f"  No new conditions matched")

                fb_threshold -= fallback_step

            # Summary
            if fb_stats:
                total_rescued = sum(n for _, n in fb_stats)
                parts = [f"{n} at {t}" for t, n in fb_stats]
                _log.print(f"\n  Fallback total: {total_rescued} conditions rescued "
                      f"({', '.join(parts)})")
                if unmapped_conds:
                    _log.print(f"  Still unmapped: {len(unmapped_conds)} conditions")
            elif unmapped_conds:
                _log.print(f"  Fallback: no new matches found. "
                      f"{len(unmapped_conds)} conditions remain unmapped.")

            results["df_matches"] = df_matches
            results["_fallback_stats"] = fb_stats

        if do_ai_review:
            step += 1
            _log.print(f"\n\033[1m[{step}/{n_steps}] AI review...\033[0m")
            results = self.ai_review(
                results,
                batch_size=ai_batch_size,
                gemini_api_key=gemini_api_key,
                ai_provider=ai_provider,
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
            "match_source": pl.col("match_source"),
            "match_threshold": pl.col("match_threshold"),
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
                    "match_source": None,
                    "match_threshold": None,
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
        n_search_terms = len(df_input)

        n_no_match = len(df_review.filter(pl.col("ai_verdict") == "no match"))

        # Rescue counts per category
        has_rescue = "is_rescued" in df_review.columns
        _rescue_filter = (pl.col("is_rescued") == True) if has_rescue else pl.lit(False)
        n_rescued_accepted = len(df_acc_flat.filter(_rescue_filter)) if has_rescue else 0
        n_rescued_human = len(df_human_review.filter(_rescue_filter)) if has_rescue else 0
        n_rescued_rejected = len(df_rejected.filter(
            _rescue_filter & (pl.col("ai_verdict") != "no match")
        )) if has_rescue else 0
        n_rescued_total = n_rescued_accepted + n_rescued_human + n_rescued_rejected

        # --- Per-verdict breakdown helpers ---
        def _verdict_counts(df_subset):
            """Return (n_exact, n_fuzzy, n_rescued, n_omop, n_cdc, n_both) for a df subset."""
            if len(df_subset) == 0:
                return 0, 0, 0, 0, 0, 0
            n_ex = len(df_subset.filter(pl.col("match_type") == "exact"))
            n_fz = len(df_subset.filter(pl.col("match_type") == "fuzzy"))
            n_res = len(df_subset.filter(_rescue_filter)) if has_rescue else 0
            has_src = "match_source" in df_subset.columns
            n_omop = len(df_subset.filter(pl.col("match_source") == "omop")) if has_src else 0
            n_cdc = len(df_subset.filter(pl.col("match_source") == "cdc")) if has_src else 0
            n_both = len(df_subset.filter(pl.col("match_source") == "cdc+omop")) if has_src else 0
            return n_ex, n_fz, n_res, n_omop, n_cdc, n_both

        acc_counts = _verdict_counts(df_acc_flat)
        hr_counts = _verdict_counts(df_human_review)
        # Rejected excluding no-match (no-match has no match_type/source)
        df_rej_matched = df_rejected.filter(pl.col("ai_verdict") != "no match")
        rej_counts = _verdict_counts(df_rej_matched)

        n_acc = len(df_acc_flat)
        n_hr = len(df_human_review)
        n_rej_matched = len(df_rej_matched)

        # Totals (matched rows only)
        t_hits = n_acc + n_hr + n_rej_matched
        t_exact = acc_counts[0] + hr_counts[0] + rej_counts[0]
        t_fuzzy = acc_counts[1] + hr_counts[1] + rej_counts[1]
        t_rescued = acc_counts[2] + hr_counts[2] + rej_counts[2]
        t_omop = acc_counts[3] + hr_counts[3] + rej_counts[3]
        t_cdc = acc_counts[4] + hr_counts[4] + rej_counts[4]
        t_both = acc_counts[5] + hr_counts[5] + rej_counts[5]

        # Determine which optional columns to show
        show_rescue = n_rescued_total > 0
        show_source = "match_source" in df_review.columns

        # Build header and rows
        hdr = ["Verdict", "Hits", "Exact", "Fuzzy"]
        if show_rescue:
            hdr.append("Rescued")
        if show_source:
            hdr.extend(["OMOP", "CDC", "Both"])

        def _row(label, hits, exact, fuzzy, rescued, omop, cdc, both, dash_extra=False):
            vals = [label, str(hits), str(exact), str(fuzzy)]
            if show_rescue:
                vals.append(str(rescued) if not dash_extra else "\u2014")
            if show_source:
                if dash_extra:
                    vals.extend(["\u2014", "\u2014", "\u2014"])
                else:
                    vals.extend([str(omop), str(cdc), str(both)])
            return vals

        rows = [
            _row("Accepted", n_acc, *acc_counts),
            _row("Human review", n_hr, *hr_counts),
            _row("Rejected", n_rej_matched, *rej_counts),
            _row("No match", n_no_match, "\u2014", "\u2014", 0, 0, 0, 0, dash_extra=True),
        ]
        totals = _row("Total", t_hits + n_no_match, t_exact, t_fuzzy,
                       t_rescued, t_omop, t_cdc, t_both)

        # Compute column widths
        col_w = [max(len(hdr[i]), *(len(r[i]) for r in rows), len(totals[i]))
                 for i in range(len(hdr))]
        # First column left-aligned, rest right-aligned
        def _fmt(vals):
            parts = [vals[0].ljust(col_w[0])]
            parts.extend(vals[i].rjust(col_w[i]) for i in range(1, len(vals)))
            return "  " + " | ".join(parts)

        sep = "  " + "-" * col_w[0] + "-+-" + "-+-".join("-" * col_w[i] for i in range(1, len(hdr)))

        _log.print(f"\n{'=' * 40}")
        _log.print(f"  FINAL SUMMARY")
        _log.print(f"{'=' * 40}")
        _log.print(f"  Conditions: {n_mapped} mapped, {n_unmapped} unmapped (of {n_conditions})")
        _log.print(f"  Search terms: {n_search_terms}")
        _log.print(f"{'-' * 40}")
        _log.print(_fmt(hdr))
        _log.print(sep)
        for r in rows:
            _log.print(_fmt(r))
        _log.print(sep)
        _log.print(_fmt(totals))
        _log.print(f"{'-' * 40}")

        # ICD version breakdown (accepted)
        if len(df_accepted) > 0:
            version_counts = df_accepted.group_by("icd_version").agg(
                pl.col("n_codes").sum().alias("total_codes"),
                pl.len().alias("n_condition_version_pairs"),
            ).sort("icd_version")
            for row in version_counts.iter_rows(named=True):
                label = f"ICD-{row['icd_version']}-CM"
                _log.print(f"  {label}: {row['total_codes']} codes, "
                      f"{row['n_condition_version_pairs']} conditions")

        # Hits per condition (only conditions with >=1 match)
        df_with_hits = df_review.filter(pl.col("icd_code").is_not_null())
        if len(df_with_hits) > 0:
            hits_per_cond = (
                df_with_hits.group_by("condition_name")
                .agg(pl.len().alias("n_hits"))
                ["n_hits"]
            )
            _log.print(f"  Hits/condition: "
                  f"min={hits_per_cond.min()} "
                  f"max={hits_per_cond.max()} "
                  f"med={hits_per_cond.median():.0f} "
                  f"\u03bc={hits_per_cond.mean():.1f}")

        if has_ai_cols:
            reviewed = df_review.filter(
                pl.col("ai_verdict").is_not_null()
                & (pl.col("ai_verdict") != "no match")
            )
            if len(reviewed) > 0:
                verdicts = reviewed.group_by("ai_verdict").len().sort("ai_verdict")
                parts = [f"{r['ai_verdict']}={r['len']}" for r in verdicts.iter_rows(named=True)]
                _log.print(f"  AI reviewed: {len(reviewed)} ({', '.join(parts)})")
                if "ai_combined_confidence" in reviewed.columns:
                    rel = reviewed.filter(
                        pl.col("ai_combined_confidence").is_not_null()
                    ).group_by("ai_combined_confidence").len().sort("ai_combined_confidence")
                    if len(rel) > 0:
                        rel_parts = [f"{r['ai_combined_confidence']}={r['len']}" for r in rel.iter_rows(named=True)]
                        _log.print(f"  AI reliability: {', '.join(rel_parts)}")

        fb_stats = results.get("_fallback_stats", [])
        if fb_stats:
            fb_total = sum(n for _, n in fb_stats)
            fb_parts = [f"{n} at {t}" for t, n in fb_stats]
            _log.print(f"  Fallback: {fb_total} conditions rescued ({', '.join(fb_parts)})")
        _log.print(f"{'=' * 40}")

        # --- Export ---
        if export_tsv:
            write_tsv_bom(df_review, f"{export_prefix}_full.tsv")
            write_tsv_bom(df_accepted, f"{export_prefix}_accepted.tsv")
            write_tsv_bom(df_human_review, f"{export_prefix}_human_review.tsv")
            write_tsv_bom(df_rejected, f"{export_prefix}_rejected.tsv")
            write_tsv_bom(df_unmatched_terms, f"{export_prefix}_unmatched_terms.tsv")
            write_tsv_bom(df_unmapped_conditions, f"{export_prefix}_unmapped_conditions.tsv")
            _log.print(f"\nExported:")
            _log.print(f"  {export_prefix}_full.tsv                  ({len(df_review)} matches)")
            _log.print(f"  {export_prefix}_accepted.tsv              ({len(df_accepted)} matches)")
            _log.print(f"  {export_prefix}_human_review.tsv          ({len(df_human_review)} matches)")
            _log.print(f"  {export_prefix}_rejected.tsv              ({len(df_rejected)} matches)")
            _log.print(f"  {export_prefix}_unmatched_terms.tsv       ({len(df_unmatched_terms)} terms)")
            _log.print(f"  {export_prefix}_unmapped_conditions.tsv   ({len(df_unmapped_conditions)} conditions)")

        # Store the log
        results["_run_log"] = _log.getvalue()

        return results

    # -------------------------------------------------------------------
    # ICD → SNOMED mapping
    # -------------------------------------------------------------------

    def icd_to_snomed(self, df_accepted: pl.DataFrame) -> dict:
        """Map accepted ICD codes (+ descendants) to SNOMED via OMOP concept_relationship.

        Takes the grouped ``df_accepted`` from :meth:`map` and:

        1. Explodes comma-separated ICD codes to flat rows
        2. Expands each code to all descendant codes (prefix match)
        3. Maps expanded ICD concept IDs → SNOMED via ``concept_relationship``
        4. Joins SNOMED results back to conditions
        5. Detects overlapping SNOMED concepts (shared by 2+ conditions)

        Args:
            df_accepted (pl.DataFrame): Grouped output from ``map()`` with
                comma-separated ``icd_codes``.

        Returns:
            dict: Dictionary with keys:

                - ``df_snomed`` — full condition → SNOMED mapping
                - ``df_overlaps`` — SNOMED concepts shared by 2+ conditions
        """
        _empty_snomed = pl.DataFrame(schema={
            "condition_name": pl.Utf8, "source_icd_code": pl.Utf8,
            "icd_version": pl.Utf8, "icd_code": pl.Utf8,
            "icd_concept_name": pl.Utf8,
            "snomed_concept_id": pl.Int64, "snomed_code": pl.Utf8,
            "snomed_name": pl.Utf8,
        })
        _empty_overlaps = pl.DataFrame(schema={
            "snomed_concept_id": pl.Int64, "snomed_code": pl.Utf8,
            "snomed_name": pl.Utf8, "n_conditions": pl.UInt32,
            "conditions": pl.Utf8, "icd_codes": pl.Utf8,
        })

        # ── Step 1: Explode comma-separated ICD codes to flat rows ──
        df_exploded = (
            df_accepted
            .select("condition_name", "icd_version", "icd_codes")
            .with_columns(pl.col("icd_codes").str.split(", ").alias("_codes"))
            .explode("_codes")
            .rename({"_codes": "icd_code"})
            .drop("icd_codes")
            .filter(pl.col("icd_code").is_not_null() & (pl.col("icd_code") != ""))
            .with_columns(
                pl.when(pl.col("icd_version").cast(pl.Utf8) == "10")
                .then(pl.lit("ICD10CM"))
                .otherwise(pl.lit("ICD9CM"))
                .alias("vocabulary_id")
            )
        )

        unique_codes = df_exploded["icd_code"].unique().to_list()
        if not unique_codes:
            return {"df_snomed": _empty_snomed, "df_overlaps": _empty_overlaps}

        # ── Step 2: Expand to descendants via vocab DuckDB ──────────
        starts = " OR ".join(
            f"STARTS_WITH(concept_code, '{sql_escape(c)}')"
            for c in unique_codes
        )
        df_icd_all = self._query(f"""
            SELECT concept_id, concept_code, concept_name, vocabulary_id
            FROM concept
            WHERE vocabulary_id IN ('ICD9CM', 'ICD10CM')
              AND invalid_reason IS NULL
              AND ({starts})
        """)

        icd_ids = df_icd_all["concept_id"].to_list()
        if not icd_ids:
            return {"df_snomed": _empty_snomed, "df_overlaps": _empty_overlaps}

        # ── Step 3: Map expanded ICD → SNOMED via concept_relationship
        id_list = ", ".join(str(i) for i in icd_ids)
        df_icd_snomed = self._query(f"""
            SELECT
                c_icd.concept_id   AS icd_concept_id,
                c_icd.concept_code AS icd_code,
                c_icd.concept_name AS icd_concept_name,
                c_icd.vocabulary_id,
                c_snomed.concept_id   AS snomed_concept_id,
                c_snomed.concept_code AS snomed_code,
                c_snomed.concept_name AS snomed_name
            FROM concept_relationship cr
            JOIN concept c_icd    ON cr.concept_id_1 = c_icd.concept_id
            JOIN concept c_snomed ON cr.concept_id_2 = c_snomed.concept_id
            WHERE cr.relationship_id = 'Maps to'
              AND c_snomed.vocabulary_id = 'SNOMED'
              AND c_snomed.standard_concept = 'S'
              AND cr.invalid_reason IS NULL
              AND c_icd.concept_id IN ({id_list})
        """)

        if len(df_icd_snomed) == 0:
            return {"df_snomed": _empty_snomed, "df_overlaps": _empty_overlaps}

        # ── Step 4: Join back to conditions (prefix match) ──────────
        # Cross-join exploded conditions with ICD→SNOMED results within
        # each vocabulary, then filter by prefix match.
        df_snomed = (
            df_exploded
            .rename({"icd_code": "source_icd_code"})
            .join(df_icd_snomed, on="vocabulary_id", how="inner")
            .filter(
                pl.col("icd_code").str.starts_with(pl.col("source_icd_code"))
            )
            .select(
                "condition_name", "source_icd_code", "icd_version",
                "icd_code", "icd_concept_name",
                "snomed_concept_id", "snomed_code", "snomed_name",
            )
            .unique()
        )

        # ── Step 5: Detect overlaps ─────────────────────────────────
        df_overlaps = (
            df_snomed
            .group_by("snomed_concept_id", "snomed_code", "snomed_name")
            .agg(
                pl.col("condition_name").n_unique().alias("n_conditions"),
                pl.col("condition_name").unique().sort()
                    .str.concat(", ").alias("conditions"),
                pl.col("icd_code").unique().sort()
                    .str.concat(", ").alias("icd_codes"),
            )
            .filter(pl.col("n_conditions") >= 2)
            .sort("n_conditions", descending=True)
        )

        # ── Step 6: Print summary ──────────────────────────────────
        n_icd10 = df_exploded.filter(
            pl.col("icd_version").cast(pl.Utf8) == "10"
        ).height
        n_icd9 = df_exploded.filter(
            pl.col("icd_version").cast(pl.Utf8) == "9"
        ).height
        n_expanded = df_icd_all["concept_id"].n_unique()
        n_snomed = (
            df_snomed["snomed_concept_id"].n_unique()
            if len(df_snomed) > 0 else 0
        )
        n_overlap = len(df_overlaps)
        n_overlap_conds = (
            df_overlaps["conditions"].str.split(", ").explode().n_unique()
            if n_overlap > 0 else 0
        )

        print(f"\n{'=' * 40}")
        print(f"  ICD \u2192 SNOMED MAPPING")
        print(f"{'=' * 40}")
        print(f"  Accepted ICD codes:    {len(unique_codes)}"
              f" ({n_icd10} ICD-10, {n_icd9} ICD-9)")
        print(f"  Expanded descendants:  {n_expanded}")
        print(f"  SNOMED concepts:       {n_snomed}")
        print(f"  Overlapping:           {n_overlap} concepts"
              f" across {n_overlap_conds} conditions")
        print(f"{'=' * 40}")

        # ── Condition-level summary ───────────────────────────────────
        # ICD codes per condition (from exploded input)
        icd_per_cond = (
            df_exploded
            .group_by("condition_name")
            .agg(pl.col("icd_code").n_unique().alias("n_icd"))
        )

        # SNOMED concepts per condition
        snomed_per_cond = (
            df_snomed
            .group_by("condition_name")
            .agg(pl.col("snomed_concept_id").n_unique().alias("n_snomed"))
        ) if len(df_snomed) > 0 else pl.DataFrame(
            schema={"condition_name": pl.Utf8, "n_snomed": pl.UInt32}
        )

        # Overlapping SNOMED concepts per condition
        if n_overlap > 0:
            overlap_per_cond = (
                df_overlaps
                .select(
                    pl.col("snomed_concept_id"),
                    pl.col("conditions").str.split(", "),
                )
                .explode("conditions")
                .rename({"conditions": "condition_name"})
                .group_by("condition_name")
                .agg(
                    pl.col("snomed_concept_id").n_unique()
                        .alias("n_overlaps"),
                )
            )
        else:
            overlap_per_cond = pl.DataFrame(
                schema={"condition_name": pl.Utf8, "n_overlaps": pl.UInt32}
            )

        df_condition_summary = (
            icd_per_cond
            .join(snomed_per_cond, on="condition_name", how="left")
            .join(overlap_per_cond, on="condition_name", how="left")
            .with_columns(
                pl.col("n_snomed").fill_null(0),
                pl.col("n_overlaps").fill_null(0),
            )
            .sort(
                pl.col("n_overlaps").cast(pl.Int64),
                "condition_name",
                descending=[True, False],
            )
        )

        # Print condition-level table
        cond_rows = df_condition_summary.iter_rows(named=True)
        max_name = max(
            len(r["condition_name"])
            for r in df_condition_summary.iter_rows(named=True)
        )
        max_name = min(max(max_name, 20), 44)  # clamp width

        hdr_cond = "Condition"
        hdr_icd = "ICD"
        hdr_sno = "SNOMED"
        hdr_ovl = "Overlaps"
        w_icd = max(len(hdr_icd), 3)
        w_sno = max(len(hdr_sno), 4)
        w_ovl = max(len(hdr_ovl), 4)

        sep_line = (
            f"  {'─' * max_name}─┼─"
            f"{'─' * w_icd}─┼─"
            f"{'─' * w_sno}─┼─"
            f"{'─' * w_ovl}"
        )

        print(f"\n  {hdr_cond:<{max_name}} | "
              f"{hdr_icd:>{w_icd}} | "
              f"{hdr_sno:>{w_sno}} | "
              f"{hdr_ovl:>{w_ovl}}")
        print(sep_line)
        for row in df_condition_summary.iter_rows(named=True):
            name = row["condition_name"]
            if len(name) > max_name:
                name = name[: max_name - 1] + "\u2026"
            print(f"  {name:<{max_name}} | "
                  f"{row['n_icd']:>{w_icd}} | "
                  f"{row['n_snomed']:>{w_sno}} | "
                  f"{row['n_overlaps']:>{w_ovl}}")
        print(sep_line)

        # Totals row (sum of per-condition unique counts)
        t_conds = df_condition_summary.height
        t_icd = df_condition_summary["n_icd"].sum()
        t_sno = df_condition_summary["n_snomed"].sum()
        t_ovl = df_condition_summary["n_overlaps"].sum()
        print(f"  {'Total (' + str(t_conds) + ' conditions)':<{max_name}} | "
              f"{t_icd:>{w_icd}} | "
              f"{t_sno:>{w_sno}} | "
              f"{t_ovl:>{w_ovl}}")
        print(f"{'=' * 40}")

        return {
            "df_snomed": df_snomed,
            "df_overlaps": df_overlaps,
            "df_condition_summary": df_condition_summary,
        }

    # -------------------------------------------------------------------
    # AI review of SNOMED overlap mappings
    # -------------------------------------------------------------------

    def review_snomed_overlaps(
        self,
        snomed_results: dict,
        ai_provider: str = "gemini",
        ai_tier: Optional[str] = None,
        ai_min_version: Optional[float] = None,
        ai_passes: int = 2,
        export_tsv: bool = False,
        export_prefix: str = "snomed_mapping",
    ) -> dict:
        """AI review of SNOMED concepts shared by 2+ conditions.

        For each (SNOMED concept, condition) pair in the overlap set,
        asks the AI whether the mapping is clinically valid or an artifact
        of broad ICD coding. Supports Gemini and Claude with auto-fallback.

        Args:
            snomed_results (dict): Output of :meth:`icd_to_snomed` containing
                ``df_snomed`` and ``df_overlaps``.
            ai_provider (str): Primary AI provider: ``"gemini"`` or ``"claude"``.
                Default ``"gemini"``.
            ai_tier (str, optional): Preferred model tier. Auto-resolves per provider
                if None: Gemini → "pro", Claude → "sonnet".
            ai_min_version (float, optional): Minimum model version. Gemini default 3.0,
                Claude default 4.6.
            ai_passes (int): Number of independent AI passes. Default 2.
            export_tsv (bool): Export results as TSV files. Default False.
            export_prefix (str): Filename prefix for exported TSVs. Default "snomed_mapping".

        Returns:
            dict: Updated ``snomed_results`` with added keys:

                - ``df_snomed`` — original + ``ai_overlap_verdict`` column
                - ``df_overlaps`` — unchanged
                - ``df_reviewed`` — per (snomed, condition) verdicts
                - ``df_snomed_accepted`` — grouped by condition, AI rejects removed
        """
        provider, model = self._resolve_model(
            ai_provider=ai_provider,
            ai_tier=ai_tier, ai_min_version=ai_min_version,
        )

        df_snomed = snomed_results["df_snomed"]
        df_overlaps = snomed_results["df_overlaps"]

        if len(df_overlaps) == 0:
            print("  No overlapping SNOMED concepts to review.")
            snomed_results["df_snomed"] = df_snomed.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("ai_overlap_verdict")
            )
            snomed_results["df_reviewed"] = pl.DataFrame(schema={
                "snomed_concept_id": pl.Int64, "snomed_code": pl.Utf8,
                "snomed_name": pl.Utf8, "condition_name": pl.Utf8,
                "icd_codes": pl.Utf8, "ai_verdict": pl.Utf8,
                "ai_comment": pl.Utf8, "ai_vote": pl.Utf8,
                "ai_vote_confidence": pl.Utf8,
                "ai_comment_consistency": pl.Int64,
                "ai_comment_consistency_tier": pl.Utf8,
                "ai_combined_confidence": pl.Utf8,
            })
            # No overlaps — all SNOMED mappings accepted
            df_snomed_full = snomed_results["df_snomed"]
            snomed_results["df_snomed_accepted"] = (
                df_snomed_full
                .group_by("condition_name")
                .agg(
                    pl.col("snomed_concept_id").unique().sort()
                        .cast(pl.Utf8).str.concat(", ")
                        .alias("snomed_concept_ids"),
                    pl.col("snomed_code").unique().sort()
                        .str.concat(", ").alias("snomed_codes"),
                    pl.col("snomed_name").unique().sort()
                        .str.concat(", ").alias("snomed_names"),
                    pl.col("snomed_concept_id").n_unique().alias("n_snomed"),
                )
                .sort("condition_name")
            )
            if export_tsv:
                write_tsv_bom(
                    df_snomed_full,
                    f"{export_prefix}_snomed_full.tsv",
                )
                write_tsv_bom(
                    snomed_results["df_snomed_accepted"],
                    f"{export_prefix}_snomed_accepted.tsv",
                )
                print(f"\nExported:")
                print(f"  {export_prefix}_snomed_full.tsv       "
                           f"({len(df_snomed_full)} rows)")
                print(f"  {export_prefix}_snomed_accepted.tsv   "
                           f"({len(snomed_results['df_snomed_accepted'])} conditions)")
            return snomed_results

        # ── Step 1: Build review items ────────────────────────────────
        # For each overlapping SNOMED concept, find which conditions
        # map to it and which ICD codes led there.
        overlap_ids = df_overlaps["snomed_concept_id"].to_list()

        df_overlap_detail = (
            df_snomed
            .filter(pl.col("snomed_concept_id").is_in(overlap_ids))
            .group_by("snomed_concept_id", "snomed_code", "snomed_name",
                       "condition_name")
            .agg(
                pl.col("icd_code").unique().sort()
                    .str.concat(", ").alias("icd_codes"),
            )
        )

        review_items = []
        for row in df_overlap_detail.iter_rows(named=True):
            review_items.append({
                "snomed_concept_id": row["snomed_concept_id"],
                "snomed_code": row["snomed_code"],
                "snomed_name": row["snomed_name"],
                "condition_name": row["condition_name"],
                "icd_codes": row["icd_codes"],
            })

        n_concepts = len(overlap_ids)
        n_pairs = len(review_items)
        print(f"\n  Overlapping SNOMED concepts: {n_concepts}")
        print(f"  (concept, condition) pairs:  {n_pairs}")
        print(f"  AI passes: {ai_passes}")
        print(f"  Model: {model}")

        # ── Step 2: Build prompts ─────────────────────────────────────
        system_prompt = (
            "You are a clinical terminologist reviewing SNOMED concept "
            "overlaps in an OMOP cohort definition.\n\n"
            "Each SNOMED concept below is mapped to 2+ conditions via ICD "
            "code expansion. For each (snomed_concept, condition) pair, "
            "decide:\n"
            '- "keep" — the SNOMED concept is clinically relevant to this '
            "condition\n"
            '- "remove" — the mapping is an artifact of broad ICD coding; '
            "this concept does not represent this condition\n"
            '- "review" — uncertain, needs human review\n\n'
            "Respond with a JSON array of verdicts. "
            "Comment: 2-5 word clinical rationale."
        )

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "verdicts": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "snomed_id": {"type": "STRING"},
                            "condition": {"type": "STRING"},
                            "v": {"type": "STRING",
                                   "enum": ["keep", "remove", "review"]},
                            "comment": {"type": "STRING"},
                        },
                        "required": ["snomed_id", "condition", "v", "comment"],
                    },
                },
            },
            "required": ["verdicts"],
        }

        # Group data prompt by SNOMED concept
        grouped: dict[int, list[dict]] = {}
        for item in review_items:
            sid = item["snomed_concept_id"]
            grouped.setdefault(sid, []).append(item)

        data_lines = []
        for sid, items in grouped.items():
            first = items[0]
            data_lines.append(
                f'SNOMED {sid} "{first["snomed_name"]}" '
                f"({first['snomed_code']})"
            )
            for it in items:
                data_lines.append(
                    f"  condition={it['condition_name']} "
                    f"| icd_codes={it['icd_codes']}"
                )
            data_lines.append("")

        data_prompt = "\n".join(data_lines)
        full_prompt = f"{system_prompt}\n\n{data_prompt}"

        # Show example
        example_lines = data_lines[:8]
        print(f"\n  Example prompt (first concepts):")
        for line in example_lines:
            print(f"    {line}")
        if len(data_lines) > 8:
            print(f"    ... ({len(data_lines)} total lines)")

        # ── Step 3: Multi-pass voting ─────────────────────────────────
        all_pass_results: list[list[dict]] = []

        for pass_num in tqdm(range(1, ai_passes + 1),
                             desc="  AI overlap review", unit="pass"):
            raw = None
            for attempt in range(3):
                try:
                    raw = self._call_ai(
                        provider, data_prompt, model,
                        system_prompt=system_prompt,
                        response_schema=response_schema,
                        temperature=0.0 if pass_num == 1 else 0.2,
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        if "timed out" in str(e):
                            print(f"    Timeout, retrying ({attempt + 1}/2)...")
                            continue
                        import time as _time
                        _time.sleep(30 * (attempt + 1))
                        continue
                    # Primary exhausted — try fallback
                    fallback = "claude" if provider == "gemini" else "gemini"
                    if self._has_provider(fallback):
                        print(f"  {provider} failed, trying {fallback}...")
                        try:
                            _, fb_model = self._resolve_model(
                                ai_provider=fallback,
                            )
                            raw = self._call_ai(
                                fallback, data_prompt, fb_model,
                                system_prompt=system_prompt,
                                response_schema=response_schema,
                                temperature=0.0 if pass_num == 1 else 0.2,
                            )
                        except Exception as fb_e:
                            print(f"  {fallback} also failed: {fb_e}")
                    if raw is None:
                        raise

            try:
                parsed = self._parse_ai_json(raw)
                verdicts = parsed.get("verdicts", [])
            except (json.JSONDecodeError, AttributeError, TypeError, RuntimeError) as exc:
                print(f"    Warning: pass {pass_num} returned unparseable response: {exc}")
                verdicts = []

            print(f"    Received {len(verdicts)} verdicts")
            all_pass_results.append(verdicts)

        # ── Tally votes per (snomed_id, condition) key ────────────────
        key_verdicts: dict[tuple, list[tuple[str, str]]] = {}
        for pass_verdicts in all_pass_results:
            for v in pass_verdicts:
                # Match snomed_id back to int — AI returns string
                try:
                    sid = int(v["snomed_id"])
                except (ValueError, KeyError):
                    continue
                cond = v.get("condition", "")
                verdict = v.get("v", "review")
                comment = v.get("comment", "")
                key = (sid, cond)
                key_verdicts.setdefault(key, []).append((verdict, comment))

        vote_results: dict[tuple, dict] = {}
        for key, vc_pairs in key_verdicts.items():
            n_total = len(vc_pairs)
            counts = {}
            for verdict, _ in vc_pairs:
                counts[verdict] = counts.get(verdict, 0) + 1

            # Majority verdict
            majority = max(counts, key=counts.get)
            majority_count = counts[majority]

            # For 2-pass: agreement = strong, disagreement = review/weak
            if n_total >= 2 and majority_count == n_total:
                confidence = "strong"
            elif n_total >= 2 and majority_count > n_total / 2:
                confidence = "moderate"
            else:
                # Disagreement — default to "review"
                majority = "review"
                confidence = "weak"

            vote_str = f"{majority_count}/{n_total}"
            majority_comments = [c for v, c in vc_pairs if v == majority]

            vote_results[key] = {
                "ai_verdict": majority,
                "ai_vote": vote_str,
                "ai_vote_confidence": confidence,
                "comments": majority_comments,
            }

        # ── Step 4: Comment consistency + combined confidence ─────────
        comment_results = self._compute_comment_consistency(vote_results)
        combined_results = self._compute_combined_confidence(
            vote_results, comment_results,
        )

        # ── Step 5: Build df_reviewed ─────────────────────────────────
        # Build a lookup from review_items for snomed metadata
        item_lookup = {
            (it["snomed_concept_id"], it["condition_name"]): it
            for it in review_items
        }

        reviewed_rows = []
        for key, vr in vote_results.items():
            sid, cond = key
            meta = item_lookup.get(key, {})
            cr = comment_results.get(key, {})
            cc = combined_results.get(key, {})
            reviewed_rows.append({
                "snomed_concept_id": sid,
                "snomed_code": meta.get("snomed_code", ""),
                "snomed_name": meta.get("snomed_name", ""),
                "condition_name": cond,
                "icd_codes": meta.get("icd_codes", ""),
                "ai_verdict": vr["ai_verdict"],
                "ai_comment": cr.get("ai_comment", ""),
                "ai_vote": vr["ai_vote"],
                "ai_vote_confidence": vr["ai_vote_confidence"],
                "ai_comment_consistency": cr.get("ai_comment_consistency", 0),
                "ai_comment_consistency_tier": cr.get(
                    "ai_comment_consistency_tier", "low"
                ),
                "ai_combined_confidence": cc.get(
                    "ai_combined_confidence", "inconclusive"
                ),
            })

        df_reviewed = pl.DataFrame(reviewed_rows, schema={
            "snomed_concept_id": pl.Int64,
            "snomed_code": pl.Utf8,
            "snomed_name": pl.Utf8,
            "condition_name": pl.Utf8,
            "icd_codes": pl.Utf8,
            "ai_verdict": pl.Utf8,
            "ai_comment": pl.Utf8,
            "ai_vote": pl.Utf8,
            "ai_vote_confidence": pl.Utf8,
            "ai_comment_consistency": pl.Int64,
            "ai_comment_consistency_tier": pl.Utf8,
            "ai_combined_confidence": pl.Utf8,
        })

        # Update df_snomed — left-join ai_overlap_verdict
        df_verdict_join = df_reviewed.select(
            "snomed_concept_id", "condition_name", "ai_verdict",
        ).rename({"ai_verdict": "ai_overlap_verdict"})

        df_snomed = df_snomed.join(
            df_verdict_join,
            on=["snomed_concept_id", "condition_name"],
            how="left",
        )

        snomed_results["df_snomed"] = df_snomed
        snomed_results["df_reviewed"] = df_reviewed

        # ── Step 6: Print summary ─────────────────────────────────────
        n_keep = len(df_reviewed.filter(pl.col("ai_verdict") == "keep"))
        n_remove = len(df_reviewed.filter(pl.col("ai_verdict") == "remove"))
        n_review = len(df_reviewed.filter(pl.col("ai_verdict") == "review"))
        n_strong = len(df_reviewed.filter(
            pl.col("ai_vote_confidence") == "strong"
        ))
        n_weak = len(df_reviewed.filter(
            pl.col("ai_vote_confidence") != "strong"
        ))

        print(f"\n{'=' * 40}")
        print(f"  SNOMED OVERLAP AI REVIEW")
        print(f"{'=' * 40}")
        print(f"  Model:                {model}")
        print(f"  Overlapping concepts: {n_concepts}")
        print(f"  Pairs reviewed:       {len(df_reviewed)}")
        print(f"    keep:               {n_keep}")
        print(f"    remove:             {n_remove}")
        print(f"    review:             {n_review}")
        print(f"  Confidence: strong={n_strong}, weak={n_weak}")
        print(f"{'=' * 40}")

        # ── Build df_snomed_accepted (grouped by condition) ───────────
        # Remove rows where AI said "remove"; keep everything else
        # (keep, review, and null = non-overlapping)
        df_snomed_clean = df_snomed.filter(
            pl.col("ai_overlap_verdict").is_null()
            | (pl.col("ai_overlap_verdict") != "remove")
        )

        df_snomed_accepted = (
            df_snomed_clean
            .group_by("condition_name")
            .agg(
                pl.col("snomed_concept_id").unique().sort()
                    .cast(pl.Utf8).str.concat(", ")
                    .alias("snomed_concept_ids"),
                pl.col("snomed_code").unique().sort()
                    .str.concat(", ").alias("snomed_codes"),
                pl.col("snomed_name").unique().sort()
                    .str.concat(", ").alias("snomed_names"),
                pl.col("snomed_concept_id").n_unique().alias("n_snomed"),
            )
            .sort("condition_name")
        )

        snomed_results["df_snomed_accepted"] = df_snomed_accepted

        # ── Export ────────────────────────────────────────────────────
        if export_tsv:
            write_tsv_bom(df_snomed, f"{export_prefix}_snomed_full.tsv")
            write_tsv_bom(
                df_snomed_accepted,
                f"{export_prefix}_snomed_accepted.tsv",
            )
            write_tsv_bom(df_reviewed, f"{export_prefix}_snomed_reviewed.tsv")
            print(f"\nExported:")
            print(f"  {export_prefix}_snomed_full.tsv       "
                       f"({len(df_snomed)} rows)")
            print(f"  {export_prefix}_snomed_accepted.tsv   "
                       f"({len(df_snomed_accepted)} conditions)")
            print(f"  {export_prefix}_snomed_reviewed.tsv   "
                       f"({len(df_reviewed)} overlap pairs)")

        return snomed_results
