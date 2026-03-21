"""
Condition2ICD — map free-text condition names to OMOP ICD Concept IDs.

Pipeline:
    1. Normalize input terms
    2. Match against ICD-9-CM / ICD-10-CM synonyms (exact + fuzzy via local DuckDB)
    3. (Optional) AI review of fuzzy matches via Gemini API

All steps can be run via a single ``map()`` call:

    mapper = Condition2ICD()

    # Matching only
    results = mapper.map(conditions, fuzzy_threshold=70)

    # Matching + AI review
    mapper.set_api_key(key_file="gemini_api_key.json")
    results = mapper.map(conditions, fuzzy_threshold=70, ai_review=True)

Setup:
    # Vocab database is auto-downloaded from Hugging Face on first use
"""

from typing import Optional

import polars as pl

from tctk._utils import (
    write_tsv_bom,
)
from tctk._omop_utils import ConditionMapperBase

__all__ = ["Condition2ICD"]


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
            "Rate confidence 0-100 (100 = certain).\n"
            "Comment: provide a short sentence or phrase explaining your decision "
            "(e.g. clinical rationale, what differs, why it matches).\n"
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
                            "c": {"type": "INTEGER"},
                            "comment": {"type": "STRING"},
                        },
                        "required": ["condition", "t", "id", "v", "c", "comment"],
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
            "ai_confidence": v.get("c"),
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
        """Print matching stats and return per-condition term counts."""
        df_term_counts = (
            df_input.group_by("condition_name").agg(
                pl.col("search_term").n_unique().alias("total_terms")
            )
        )

        all_matched_terms = set(df_matches["search_term"].unique().to_list())

        df_matched_counts = (
            df_input.filter(pl.col("search_term").is_in(all_matched_terms))
            .group_by("condition_name")
            .agg(pl.col("search_term").n_unique().alias("matched_terms"))
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

        n_conditions = len(all_conditions)
        n_search_terms = df_input["search_term"].n_unique()
        n_exact_terms = df_exact["search_term"].n_unique()
        n_fuzzy_terms = df_fuzzy["search_term"].n_unique() if len(df_fuzzy) > 0 else 0
        n_matched_terms = n_exact_terms + n_fuzzy_terms
        n_unmatched_terms = len(still_unmatched)
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
        print(f"  Matched: {n_exact_terms} exact + {n_fuzzy_terms} fuzzy "
              f"= {n_matched_terms} terms ({n_unmatched_terms} unmatched)")
        print(f"  Conditions with >=1 match: {n_cond_matched}")
        print(f"  Conditions with 0 matches: {n_cond_no_match}")
        print(f"")
        print(f"  Concept hits (each term can match multiple concepts):")
        print(f"    Exact: {n_exact_terms} terms -> {n_exact_hits} hits")
        print(f"    Fuzzy: {n_fuzzy_terms} terms -> {n_fuzzy_hits} hits")
        print(f"    Total: {n_matched_terms} terms -> {n_concept_hits} hits")
        print(f"")
        print(f"  ICD concepts: {n_icd} unique "
              f"({n_total_matches} total matches)")
        print(f"{'=' * 40}")

        return df_term_counts

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def map(
        self,
        conditions: dict[str, list[str]],
        fuzzy_threshold: int = 85,
        ai_review: bool = False,
        gemini_api_key: Optional[str] = None,
        ai_tier: str = "flash",
        ai_min_version: float = 3.0,
        config_path: Optional[str] = None,
        confidence_threshold: int = 80,
        ai_batch_size: Optional[int] = None,
        export_tsv: bool = False,
        export_prefix: str = "mapping",
    ) -> dict:
        """Map condition names and their synonyms to OMOP ICD Concept IDs.

        Parameters
        ----------
        conditions : dict[str, list[str]]
            Keys are condition names; values are lists of condition synonyms.
        fuzzy_threshold : int
            Minimum score (0-100) for rapidfuzz token_sort_ratio. Default 85.
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
        confidence_threshold : int
            Confidence score (0-100) below which AI verdicts are flagged as
            "human review". Default 80.
        ai_batch_size : int, optional
            Conditions per AI review API call. If None, auto-calculated.
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
                standard_concept (bool), exact_match (bool),
                top_level_code (first 3 chars of icd_code),
                has_confirmed_sibling (bool — True when a fuzzy match's
                top-level code also appears in an exact match for the same
                condition), fuzzy_score, ai_verdict, ai_confidence,
                ai_comment.
                Unmatched conditions appear with null ICD columns and
                ai_verdict="no match".
            df_accepted : pl.DataFrame
                Grouped table with max 2 rows per condition (one ICD-9,
                one ICD-10). Columns: condition_name, icd_version,
                icd_codes, top_level_codes, icd_concept_names,
                icd_concept_ids, n_codes. Code columns are comma-separated
                aggregations of unique values.
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
        n_steps = 3 + (1 if do_ai_review else 0)
        step = 0

        step += 1
        print(f"\033[1m[{step}/{n_steps}] Building search terms...\033[0m")
        df_input = self._build_input(conditions)
        print(
            f"  Conditions: {df_input['condition_name'].n_unique()}, "
            f"Search terms: {df_input['search_term'].n_unique()}"
        )

        step += 1
        print(f"\n\033[1m[{step}/{n_steps}] Matching against vocabulary...\033[0m")
        df_exact, df_fuzzy = self._match(df_input, vocab="ICD", fuzzy_threshold=fuzzy_threshold)

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

        # has_confirmed_sibling: True if this row's top-level code also appears
        # in an exact match for the same condition (signals same code family)
        exact_top_codes = (
            df_matches
            .filter(pl.col("match_type") == "exact")
            .select("condition_name", "top_level_code")
            .unique()
            .with_columns(pl.lit(True).alias("_has_sibling"))
        )
        df_matches = (
            df_matches
            .join(exact_top_codes, on=["condition_name", "top_level_code"], how="left")
            .with_columns(
                pl.col("_has_sibling").fill_null(False).alias("has_confirmed_sibling")
            )
            .drop("_has_sibling")
        )

        if not do_ai_review and fuzzy_threshold < 85 and len(df_fuzzy) > 0:
            print(f"\n  Warning: AI review is off and fuzzy_threshold={fuzzy_threshold}. "
                  f"Low-score fuzzy matches won't be vetted. "
                  f"Consider fuzzy_threshold >= 85 or enabling ai_review=True.")

        step += 1
        print(f"\n\033[1m[{step}/{n_steps}] Summarizing...\033[0m")
        df_term_counts = self._summarize(df_input, df_exact, df_fuzzy, df_matches)

        results = {
            "df_input": df_input,
            "df_exact": df_exact,
            "df_fuzzy": df_fuzzy,
            "df_matches": df_matches,
            "df_term_counts": df_term_counts,
        }

        if do_ai_review:
            step += 1
            print(f"\n\033[1m[{step}/{n_steps}] AI review...\033[0m")
            results = self.ai_review(
                results,
                batch_size=ai_batch_size,
                confidence_threshold=confidence_threshold,
                gemini_api_key=gemini_api_key,
                ai_tier=ai_tier,
                ai_min_version=ai_min_version,
                config_path=config_path,
            )

        # --- Build df_review: one row per match pair, flat columns ---
        df_final = results["df_matches"]

        # Recompute has_confirmed_sibling using all confirmed matches
        # (exact + AI-accepted) so the final output reflects post-review state
        if "ai_verdict" in df_final.columns:
            confirmed_top_codes = (
                df_final.filter(
                    (pl.col("match_type") == "exact")
                    | (pl.col("ai_verdict") == "accept")
                )
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
            "exact_match": (pl.col("match_type") == "exact"),
            "top_level_code": pl.col("top_level_code"),
            "has_confirmed_sibling": pl.col("has_confirmed_sibling"),
            "fuzzy_score": pl.col("match_score"),
        }

        df_review = df_final.select(**review_cols)

        # Add AI columns if present, otherwise nulls
        if "ai_verdict" in df_final.columns:
            df_review = df_review.with_columns(
                df_final["ai_verdict"],
                df_final["ai_confidence"],
                df_final["ai_comment"],
            )
        else:
            df_review = df_review.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("ai_verdict"),
                pl.lit(None).cast(pl.Int64).alias("ai_confidence"),
                pl.lit(None).cast(pl.Utf8).alias("ai_comment"),
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
                    "exact_match": None,
                    "top_level_code": None,
                    "has_confirmed_sibling": None,
                    "fuzzy_score": None,
                    "ai_verdict": "no match",
                    "ai_confidence": None,
                    "ai_comment": None,
                }
                for cond, term in unmatched_pairs
            ]
            df_unmatched_terms = pl.DataFrame(unmatched_rows, schema=df_review.schema)
            df_review = pl.concat([df_review, df_unmatched_terms], how="diagonal_relaxed")
        else:
            df_unmatched_terms = pl.DataFrame(schema=df_review.schema)

        # Build grouped accepted table (max 2 rows per condition: ICD-9 + ICD-10)
        df_acc_flat = df_review.filter(
            (pl.col("ai_verdict").is_null() | (pl.col("ai_verdict") == "accept"))
            & pl.col("icd_code").is_not_null()
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
        df_rejected = df_review.filter(
            pl.col("ai_verdict").is_in(["reject", "human review", "no match"])
        )

        # Conditions with no usable ICD mapping (no match or all rejected)
        mapped_conds = set(
            df_review.filter(
                pl.col("icd_concept_id").is_not_null()
                & (pl.col("ai_verdict").is_null() | (pl.col("ai_verdict") == "accept"))
            )["condition_name"].to_list()
        )
        all_conds = set(df_review["condition_name"].to_list())
        unmapped_conds = all_conds - mapped_conds
        df_unmapped_conditions = df_review.filter(pl.col("condition_name").is_in(list(unmapped_conds)))

        results["df_review"] = df_review
        results["df_accepted"] = df_accepted
        results["df_rejected"] = df_rejected
        results["df_unmatched_terms"] = df_unmatched_terms
        results["df_unmapped_conditions"] = df_unmapped_conditions

        # Remove old keys no longer produced
        results.pop("df_matches", None)

        # --- Export ---
        if export_tsv:
            write_tsv_bom(df_review, f"{export_prefix}_full.tsv")
            write_tsv_bom(df_accepted, f"{export_prefix}_accepted.tsv")
            write_tsv_bom(df_rejected, f"{export_prefix}_rejected.tsv")
            write_tsv_bom(df_unmatched_terms, f"{export_prefix}_unmatched_terms.tsv")
            write_tsv_bom(df_unmapped_conditions, f"{export_prefix}_unmapped_conditions.tsv")
            print(f"\nExported:")
            print(f"  {export_prefix}_full.tsv                  ({len(df_review)} rows)")
            print(f"  {export_prefix}_accepted.tsv              ({len(df_accepted)} rows)")
            print(f"  {export_prefix}_rejected.tsv              ({len(df_rejected)} rows)")
            print(f"  {export_prefix}_unmatched_terms.tsv       ({len(df_unmatched_terms)} rows)")
            print(f"  {export_prefix}_unmapped_conditions.tsv   ({len(df_unmapped_conditions)} rows)")

        return results
