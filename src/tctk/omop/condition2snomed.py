"""
Module: _omop_condition2snomed

Map condition names and their synonyms to OMOP Standard SNOMED Concept IDs
using a local DuckDB vocabulary database built from Athena CSV files.

Pipeline:
    1. Normalize input terms (lowercase, strip tags, expand parentheticals)
    2. Match against SNOMED standard synonyms (exact + fuzzy via local DuckDB)
    3. (Optional) Compute ancestor distances via concept_ancestor
    4. (Optional) AI review of fuzzy matches via Gemini API

All steps can be run via a single ``map()`` call:

    mapper = Condition2SNOMED()

    # Matching + ancestor distances
    results = mapper.map(conditions, fuzzy_threshold=70)

    # Matching + ancestor distances + AI review
    mapper.set_api_key(key_file="gemini_api_key.json")
    results = mapper.map(conditions, fuzzy_threshold=70, ai_review=True)

Setup:
    # Vocab database is auto-downloaded from Hugging Face on first use
"""

from typing import Optional

import polars as pl
from tqdm.auto import tqdm

from tctk._utils import (
    sql_escape,
    write_tsv_bom,
)
from tctk.omop._base import ConditionMapperBase

__all__ = ["Condition2SNOMED"]


class Condition2SNOMED(ConditionMapperBase):
    """Map condition names and synonyms to OMOP Standard SNOMED Concept IDs.

    Uses a local DuckDB vocabulary database built from Athena CSV files.
    No network access required for mapping — only for optional AI review.

    Args:
        vocab_db (str, optional): Path to the DuckDB vocabulary database.
            Default: auto-downloaded from Hugging Face
        force_download_db (bool): Force re-download of the vocabulary database. Default False.
    """

    _TARGET_ID_COL = "snomed_concept_id"
    _TARGET_NAME_COL = "snomed_concept_name"
    _VOCAB_LABEL = "SNOMED CT"

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

        n_snomed = df_matches["snomed_concept_id"].drop_nulls().n_unique()
        n_total_matches = len(df_matches.filter(pl.col("snomed_concept_id").is_not_null()))

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
        print(f"  SNOMED concepts: {n_snomed} unique "
              f"({n_total_matches} total matches)")
        print(f"{'=' * 40}")

        return df_term_counts

    # -------------------------------------------------------------------
    # Ancestor distance via concept_ancestor
    # -------------------------------------------------------------------

    def _compute_ancestors(self, results: dict) -> dict:
        """Compute ancestor distance between matches and reference anchors.

        For each match, finds the shortest path through shared SNOMED
        ancestors to the condition's reference anchors (exact matches
        or fuzzy >= 90).

        Adds columns: ancestor_distance, nearest_ancestor_name, nearest_anchor_type
        """
        df_matches = results["df_matches"].clone()

        snomed_ids = (
            df_matches.filter(pl.col("snomed_concept_id").is_not_null())
            .select("snomed_concept_id")
            .unique()["snomed_concept_id"]
            .to_list()
        )

        if not snomed_ids:
            print("  No SNOMED concepts found.")
            results["df_matches"] = df_matches
            return results

        # Query concept_ancestor
        print("  Querying concept_ancestor...")
        id_batches = [
            snomed_ids[i : i + self.BATCH_SIZE]
            for i in range(0, len(snomed_ids), self.BATCH_SIZE)
        ]

        ancestor_parts = []
        for batch in tqdm(id_batches, desc="Ancestor queries"):
            ids_sql = ", ".join([f"'{sql_escape(cid)}'" for cid in batch])

            sql = f"""
            SELECT DISTINCT
                CAST(ca.descendant_concept_id AS VARCHAR)  AS snomed_concept_id,
                CAST(ca.ancestor_concept_id AS VARCHAR)    AS ancestor_concept_id,
                anc.concept_name                           AS ancestor_name,
                CAST(ca.min_levels_of_separation AS INT)   AS min_separation
            FROM concept_ancestor ca
            JOIN concept anc
                ON ca.ancestor_concept_id = anc.concept_id
            WHERE CAST(ca.descendant_concept_id AS VARCHAR) IN ({ids_sql})
              AND ca.min_levels_of_separation > 0
              AND anc.domain_id = 'Condition'
              AND anc.invalid_reason IS NULL
            """

            df_part = self._query(sql)
            ancestor_parts.append(df_part)

        df_ancestors = (
            pl.concat(ancestor_parts, how="diagonal_relaxed")
            if ancestor_parts
            else pl.DataFrame()
        )

        # Reference anchors: exact matches only
        df_refs = (
            df_matches.filter(pl.col("match_type") == "exact")
            .filter(pl.col("snomed_concept_id").is_not_null())
            .select("condition_name", "snomed_concept_id", "match_type")
            .unique(subset=["condition_name", "snomed_concept_id"], keep="first")
        )

        n_cond_with_refs = df_refs["condition_name"].n_unique() if len(df_refs) > 0 else 0
        n_total_conds = df_matches.filter(
            pl.col("snomed_concept_id").is_not_null()
        )["condition_name"].n_unique()
        print(f"  Reference anchors: {len(df_refs)} across "
              f"{n_cond_with_refs}/{n_total_conds} conditions")

        # Compute ancestor distance per match
        distance_rows = []
        conditions_list = df_matches["condition_name"].unique().to_list()

        for cond in tqdm(conditions_list, desc="Computing distances"):
            cond_refs = df_refs.filter(pl.col("condition_name") == cond)
            cond_matches = df_matches.filter(
                (pl.col("condition_name") == cond)
                & (pl.col("snomed_concept_id").is_not_null())
            )
            ref_ids = set(cond_refs["snomed_concept_id"].to_list())
            ref_types = dict(zip(
                cond_refs["snomed_concept_id"].to_list(),
                cond_refs["match_type"].to_list(),
            ))

            for match_row in cond_matches.iter_rows(named=True):
                match_snomed = match_row["snomed_concept_id"]
                search_term = match_row["search_term"]

                # Reference anchor → distance 0
                if match_snomed in ref_ids:
                    distance_rows.append({
                        "condition_name": cond, "search_term": search_term,
                        "snomed_concept_id": match_snomed,
                        "ancestor_distance": 0,
                        "nearest_ancestor_name": "(self-reference)",
                        "nearest_anchor_type": ref_types[match_snomed],
                    })
                    continue

                # No references or no ancestor data → null
                if not ref_ids or len(df_ancestors) == 0:
                    distance_rows.append({
                        "condition_name": cond, "search_term": search_term,
                        "snomed_concept_id": match_snomed,
                        "ancestor_distance": None,
                        "nearest_ancestor_name": None,
                        "nearest_anchor_type": None,
                    })
                    continue

                # Find shortest path via shared ancestors
                match_anc = df_ancestors.filter(pl.col("snomed_concept_id") == match_snomed)
                match_ancestor_ids = set(match_anc["ancestor_concept_id"].to_list())

                best_dist = None
                best_ancestor = None
                best_ref_type = None

                for ref_id in ref_ids:
                    ref_anc = df_ancestors.filter(pl.col("snomed_concept_id") == ref_id)
                    shared = match_ancestor_ids & set(ref_anc["ancestor_concept_id"].to_list())
                    if not shared:
                        continue

                    for anc_id in shared:
                        m_rows = match_anc.filter(pl.col("ancestor_concept_id") == anc_id)
                        r_rows = ref_anc.filter(pl.col("ancestor_concept_id") == anc_id)
                        if len(m_rows) == 0 or len(r_rows) == 0:
                            continue
                        total = m_rows["min_separation"].min() + r_rows["min_separation"].min()
                        if best_dist is None or total < best_dist:
                            best_dist = total
                            best_ancestor = m_rows["ancestor_name"][0]
                            best_ref_type = ref_types[ref_id]

                distance_rows.append({
                    "condition_name": cond, "search_term": search_term,
                    "snomed_concept_id": match_snomed,
                    "ancestor_distance": best_dist,
                    "nearest_ancestor_name": best_ancestor,
                    "nearest_anchor_type": best_ref_type,
                })

        # Join results
        if distance_rows:
            df_distances = (
                pl.DataFrame(distance_rows)
                .unique(subset=["condition_name", "search_term", "snomed_concept_id"])
            )
            n_before = len(df_matches)
            df_matches = df_matches.join(
                df_distances,
                on=["condition_name", "search_term", "snomed_concept_id"],
                how="left",
            )
            if len(df_matches) != n_before:
                df_matches = df_matches.unique(
                    subset=["condition_name", "search_term", "snomed_concept_id"]
                )
        else:
            df_matches = df_matches.with_columns(
                pl.lit(None).cast(pl.Int64).alias("ancestor_distance"),
                pl.lit(None).cast(pl.Utf8).alias("nearest_ancestor_name"),
                pl.lit(None).cast(pl.Utf8).alias("nearest_anchor_type"),
            )

        # Summary
        df_with_dist = df_matches.filter(
            pl.col("ancestor_distance").is_not_null()
            & pl.col("snomed_concept_id").is_not_null()
        )
        n_no_path = len(df_matches.filter(
            pl.col("ancestor_distance").is_null()
            & pl.col("snomed_concept_id").is_not_null()
        ))
        if len(df_with_dist) > 0:
            n_ref = len(df_with_dist.filter(pl.col("ancestor_distance") == 0))
            n_close = len(df_with_dist.filter(
                (pl.col("ancestor_distance") > 0) & (pl.col("ancestor_distance") <= 4)
            ))
            n_mid = len(df_with_dist.filter(
                (pl.col("ancestor_distance") > 4) & (pl.col("ancestor_distance") <= 8)
            ))
            n_far = len(df_with_dist.filter(pl.col("ancestor_distance") > 8))
            print(f"\n  Ancestor distances ({len(df_with_dist)} matches):")
            print(f"    distance=0 (reference): {n_ref}")
            print(f"    distance 1-4 (close):   {n_close}")
            print(f"    distance 5-8 (mid):     {n_mid}")
            print(f"    distance >8 (far):      {n_far}")
            if n_no_path > 0:
                print(f"    no path found:          {n_no_path}")

        results["df_matches"] = df_matches
        return results

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
        ai_batch_size: Optional[int] = None,
        ai_passes: int = 2,
        export_tsv: bool = False,
        export_prefix: str = "mapping",
    ) -> dict:
        """Map condition names and their synonyms to OMOP Standard SNOMED Concept IDs.

        Args:
            conditions (dict[str, list[str]]): Keys are condition names; values are
                lists of condition synonyms.
            fuzzy_threshold (int): Minimum score (0-100) for rapidfuzz token_sort_ratio.
                Default 85.
            ai_review (bool): If True, run AI review of fuzzy matches via Gemini API.
                Default False.
            gemini_api_key (str, optional): Gemini API key for AI review. Falls back to
                key set via :meth:`set_api_key`, then env var, then config file.
            ai_tier (str): Preferred Gemini model tier: "pro", "flash", or "flash-lite".
                Default "flash".
            ai_min_version (float): Minimum Gemini model version. Default 3.0 (prefer
                Gemini 3.x+). Set to 2.5 to allow older models (e.g. gemini-2.5-flash).
            config_path (str, optional): Path to JSON config file for API key.
            ai_batch_size (int, optional): Conditions per AI review API call. If None,
                auto-calculated.
            ai_passes (int): Number of initial AI review passes. Default 2.
                Uses adaptive replication: 2 initial passes, then up to 5
                for disagreements. Set to 1 for single-pass mode.
            export_tsv (bool): If True, write five TSV files:
                ``{export_prefix}_full.tsv`` — all matched term pairs with
                scores, enrichment, and AI review columns (plus "no match" rows).
                ``{export_prefix}_accepted.tsv`` — one row per condition,
                aggregating only passed matches (not AI-rejected).
                ``{export_prefix}_rejected.tsv`` — rejected / human review /
                no-match rows.
                ``{export_prefix}_unmatched_terms.tsv`` — search terms with
                zero matches.
                ``{export_prefix}_unmapped_conditions.tsv`` — conditions with
                no usable SNOMED mapping.
                Default False.
            export_prefix (str): Filename prefix for exported TSV files. Default "mapping".

        Returns:
            dict: Dictionary with keys:

                - ``df_review`` (pl.DataFrame) — All matched (search_term, concept) pairs
                  plus "no match" rows, with scores, ancestor distances, and AI review columns.
                - ``df_accepted`` (pl.DataFrame) — One row per condition, codes aggregated
                  from passed matches only (ai_verdict != "reject").
                - ``df_rejected`` (pl.DataFrame) — Rows with ai_verdict in
                  ("reject", "human review", "no match").
                - ``df_unmatched_terms`` (pl.DataFrame) — Search terms with zero matches.
                - ``df_unmapped_conditions`` (pl.DataFrame) — Conditions with no usable
                  SNOMED mapping.
                - ``df_term_counts`` (pl.DataFrame) — Per-condition matching coverage
                  (total/matched/unmatched terms).
                - ``df_input``, ``df_exact``, ``df_fuzzy`` (pl.DataFrame) — Intermediate
                  DataFrames from the matching pipeline.
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
        n_steps = 4 + (1 if do_ai_review else 0)
        step = 0

        step += 1
        print(f"\033[1m[{step}/{n_steps}] Building search terms...\033[0m")
        df_input = self._build_input(conditions)
        print(
            f"  Conditions: {df_input['condition_name'].n_unique()}, "
            f"Search terms: {len(df_input)}"
        )

        step += 1
        print(f"\n\033[1m[{step}/{n_steps}] Matching against vocabulary...\033[0m")
        df_exact, df_fuzzy = self._omop_lookup(df_input, fuzzy_threshold=fuzzy_threshold)

        # Combine exact + fuzzy into single DataFrame
        df_matches = pl.concat([df_exact, df_fuzzy], how="diagonal_relaxed")
        df_matches = df_matches.with_columns(pl.col("concept_id").cast(pl.Utf8))
        # All matches are SNOMED standard — alias columns
        df_matches = df_matches.with_columns(
            pl.col("concept_id").alias("snomed_concept_id"),
            pl.col("concept_name").alias("snomed_concept_name"),
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

        step += 1
        print(f"\n\033[1m[{step}/{n_steps}] Computing ancestor distances...\033[0m")
        results = self._compute_ancestors(results)

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

        review_cols = {
            "condition_name": pl.col("condition_name"),
            "search_term": pl.col("search_term"),
            "matched_concept_synonym": pl.col("matched_concept_synonym"),
            "snomed_concept_name": pl.col("snomed_concept_name"),
            "snomed_concept_id": pl.col("snomed_concept_id"),
            "vocabulary_id": pl.col("vocabulary_id"),
            "exact_match": (pl.col("match_type") == "exact"),
            "fuzzy_score": pl.col("match_score"),
        }

        # Add ancestor columns if present
        if "ancestor_distance" in df_final.columns:
            review_cols["ancestor_distance"] = pl.col("ancestor_distance")
            review_cols["nearest_ancestor_name"] = pl.col("nearest_ancestor_name")

        df_review = df_final.select(**review_cols)

        # Add AI columns if present, otherwise nulls
        if "ai_verdict" in df_final.columns:
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

        # Append "no match" rows for search terms that didn't match anything
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
            unmatched_row_template = {col: None for col in df_review.columns}
            unmatched_rows = []
            for cond, term in unmatched_pairs:
                row = unmatched_row_template.copy()
                row["condition_name"] = cond
                row["search_term"] = term
                row["ai_verdict"] = "no match"
                unmatched_rows.append(row)
            df_unmatched_terms = pl.DataFrame(unmatched_rows, schema=df_review.schema)
            df_review = pl.concat([df_review, df_unmatched_terms], how="diagonal_relaxed")
        else:
            df_unmatched_terms = pl.DataFrame(schema=df_review.schema)

        # Build grouped accepted table (one row per condition)
        df_acc_flat = df_review.filter(
            (pl.col("ai_verdict").is_null() | (pl.col("ai_verdict") == "accept"))
            & pl.col("snomed_concept_id").is_not_null()
        )
        df_accepted = (
            df_acc_flat
            .group_by("condition_name")
            .agg(
                pl.col("snomed_concept_id").unique().sort().str.join(", ").alias("snomed_concept_ids"),
                pl.col("snomed_concept_name").unique().sort().str.join(", ").alias("snomed_concept_names"),
                pl.col("vocabulary_id").unique().sort().str.join(", ").alias("source_vocabularies"),
                pl.col("search_term").unique().sort().str.join(", ").alias("matched_via"),
                pl.col("matched_concept_synonym").unique().sort().str.join(", ").alias("matched_synonyms"),
                pl.col("snomed_concept_id").n_unique().alias("n_codes"),
            )
            .sort("condition_name")
        )

        # Rejected table stays flat (1 row per pair)
        df_rejected = df_review.filter(
            pl.col("ai_verdict").is_in(["reject", "human review", "no match"])
        )

        # Conditions with no usable SNOMED mapping (no match or all rejected)
        mapped_conds = set(
            df_review.filter(
                pl.col("snomed_concept_id").is_not_null()
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
        results.pop("df_condition_summary", None)

        # --- Export ---
        if export_tsv:
            write_tsv_bom(df_review, f"{export_prefix}_full.tsv")
            write_tsv_bom(df_accepted, f"{export_prefix}_accepted.tsv")
            write_tsv_bom(df_rejected, f"{export_prefix}_rejected.tsv")
            write_tsv_bom(df_unmatched_terms, f"{export_prefix}_unmatched_terms.tsv")
            write_tsv_bom(df_unmapped_conditions, f"{export_prefix}_unmapped_conditions.tsv")
            print(f"\nExported:")
            print(f"  {export_prefix}_full.tsv                  ({len(df_review)} matches)")
            print(f"  {export_prefix}_accepted.tsv              ({len(df_accepted)} matches)")
            print(f"  {export_prefix}_rejected.tsv              ({len(df_rejected)} matches)")
            print(f"  {export_prefix}_unmatched_terms.tsv       ({len(df_unmatched_terms)} terms)")
            print(f"  {export_prefix}_unmapped_conditions.tsv   ({len(df_unmapped_conditions)} conditions)")

        return results
