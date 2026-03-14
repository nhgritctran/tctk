"""
Module: _aou_condition2conceptid

Map condition names and their synonyms to OMOP Standard SNOMED Concept IDs
using a local DuckDB vocabulary database built from Athena CSV files.

Pipeline:
    1. Normalize input terms (lowercase, strip tags, expand parentheticals)
    2. Exact match against concept_synonym + concept (local DuckDB)
    3. Fuzzy match unmatched terms via rapidfuzz (token_sort_ratio)
    4. Map non-SNOMED matches to SNOMED via concept_relationship ('Maps to')
    5. Rank and return results
    6. (Optional) Enrich with biological validation via concept_ancestor
    7. (Optional) AI review of ambiguous matches via Gemini API

Setup:
    # Vocab database is auto-downloaded from Hugging Face on first use

Usage:
    from tctk._aou_condition2conceptid import Condition2ConceptID

    mapper = Condition2ConceptID()

    conditions = {
        "Cicatricial pemphigoid": [
            "Benign mucosal pemphigoid",
            "Mucous membrane pemphigoid",
        ],
        "Lupus": ["SLE", "Systemic lupus erythematosus"],
    }

    results = mapper.map(conditions)
    enriched = mapper.enrich(results, single_domain=True)
    reviewed = mapper.ai_review(enriched)  # requires Gemini API key
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

from tctk._utils import (
    sql_escape,
    write_tsv_bom,
    load_api_key,
    check_api_key,
    setup_credentials,
    detect_best_model,
    call_gemini,
)
from tctk.omop import get_vocab_db

__all__ = ["Condition2ConceptID"]


class Condition2ConceptID:
    """Map condition names and synonyms to OMOP Standard SNOMED Concept IDs.

    Uses a local DuckDB vocabulary database built from Athena CSV files.
    No network access required for mapping — only for optional AI review.

    Parameters
    ----------
    vocab_db : str, optional
        Path to the DuckDB vocabulary database.
        Default: auto-downloaded from Hugging Face
    fuzzy_threshold : int
        Minimum score (0-100) for rapidfuzz token_sort_ratio. Default 85.
    gemini_api_key : str, optional
        Gemini API key for AI review. If not provided, loaded from
        env var GEMINI_API_KEY or config file.
    ai_tier : str, optional
        Preferred Gemini model tier: "pro", "flash", or "flash-lite".
        Default "flash" (cost-effective).
    config_path : str, optional
        Path to JSON config file for API key.
    """

    BATCH_SIZE = 500

    def __init__(
        self,
        vocab_db: Optional[str] = None,
        force_download_db: bool = False,
        fuzzy_threshold: int = 85,
        gemini_api_key: Optional[str] = None,
        ai_tier: str = "flash",
        config_path: Optional[str] = None,
    ):
        self._vocab_db = Path(vocab_db) if vocab_db else Path(get_vocab_db(force_download=force_download_db))
        self.fuzzy_threshold = fuzzy_threshold
        self._api_key = load_api_key(
            api_key=gemini_api_key, config_path=config_path
        )
        self._ai_tier = ai_tier
        self._ai_model: Optional[str] = None

    # -------------------------------------------------------------------
    # DuckDB query helper
    # -------------------------------------------------------------------

    def _query(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query against the local vocab DuckDB and return Polars DataFrame."""
        conn = duckdb.connect(str(self._vocab_db), read_only=True)
        try:
            result = conn.execute(sql).pl()
        finally:
            conn.close()
        return result

    # -------------------------------------------------------------------
    # Credential / model management
    # -------------------------------------------------------------------

    def set_api_key(self, key: Optional[str] = None, key_file: Optional[str] = None) -> None:
        """Set Gemini API key for AI review.

        Parameters
        ----------
        key : str, optional
            API key string directly.
        key_file : str, optional
            Path to a JSON file containing {"gemini_api_key": "your-key"}.
        """
        if key:
            self._api_key = key
        elif key_file:
            config = json.loads(Path(key_file).read_text())
            self._api_key = config.get("gemini_api_key")
            if not self._api_key:
                raise ValueError(
                    f"Key file {key_file} missing 'gemini_api_key' field.\n"
                    'Expected format: {"gemini_api_key": "your-key-here"}'
                )
        else:
            raise ValueError("Provide either key or key_file.")
        self._ai_model = None

    @staticmethod
    def setup_credentials(path: Optional[str] = None) -> None:
        """Interactive helper to create a credentials file.

        Creates ~/.config/tctk/credentials.json (or custom path) with
        the Gemini API key. Input is hidden via getpass.

        Get a free key at: https://aistudio.google.com/apikey
        """
        setup_credentials(path)

    def _resolve_model(self) -> str:
        """Detect and cache the best Gemini model for the API key and tier."""
        if self._ai_model is None:
            # Re-check env var in case it was set after __init__
            if not self._api_key:
                self._api_key = load_api_key()
            api_key = check_api_key(self._api_key)
            self._ai_model = detect_best_model(api_key, ai_tier=self._ai_tier)
            print(f"  Gemini model selected: {self._ai_model}")
        return self._ai_model

    # -------------------------------------------------------------------
    # Step 1: Build normalised search terms from input dict
    # -------------------------------------------------------------------

    @staticmethod
    def _build_input(conditions: dict[str, list[str]]) -> pl.DataFrame:
        """
        Convert {condition_name: [condition_synonyms]} dict into a normalised
        DataFrame with columns: condition_name, search_term.

        - The condition_name itself is always included as a search term.
        - Tags like (subtype) and (synonym) are stripped.
        - Terms containing parentheses are expanded into two variants:
          one with and one without the parenthetical.
        """
        rows = []
        for condition_name, condition_synonyms in conditions.items():
            condition_name = condition_name.strip()
            all_terms = [condition_name] + [
                s.strip() for s in condition_synonyms if s.strip()
            ]
            for term in all_terms:
                rows.append({"condition_name": condition_name, "search_term_raw": term})

        df = pl.DataFrame(rows)

        df_normalized = (
            df.with_columns(
                pl.col("search_term_raw")
                .str.to_lowercase()
                .str.strip_chars()
                .str.replace_all(r"\s*\(subtype\)\.?", "")
                .str.replace_all(r"\s*\(synonym\)\.?", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias("search_term")
            ).filter(pl.col("search_term") != "")
        )

        df_has_parens = df_normalized.filter(
            pl.col("search_term").str.contains(r"\(.*\)")
        )
        df_no_parens = df_normalized.filter(
            ~pl.col("search_term").str.contains(r"\(.*\)")
        )

        df_parens_stripped = (
            df_has_parens.with_columns(
                pl.col("search_term")
                .str.replace_all(r"\s*\([^)]*\)", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias("search_term")
            ).filter(pl.col("search_term") != "")
        )

        df_input = (
            pl.concat(
                [df_no_parens, df_has_parens, df_parens_stripped],
                how="diagonal_relaxed",
            )
            .unique(subset=["condition_name", "search_term"])
            .select("condition_name", "search_term")
        )

        return df_input

    # -------------------------------------------------------------------
    # Step 2: Exact match via concept_synonym (DuckDB)
    # -------------------------------------------------------------------

    def _exact_match(self, df_input: pl.DataFrame) -> pl.DataFrame:
        """Query local vocab DB for exact (case-insensitive) matches."""
        unique_terms = df_input["search_term"].unique().sort().to_list()
        term_batches = [
            unique_terms[i : i + self.BATCH_SIZE]
            for i in range(0, len(unique_terms), self.BATCH_SIZE)
        ]

        parts = []
        for batch in tqdm(term_batches, desc="Exact match batches"):
            terms_sql = ", ".join([f"'{sql_escape(t)}'" for t in batch])

            sql = f"""
            SELECT
                i.term                                     AS search_term,
                LOWER(TRIM(cs.concept_synonym_name))       AS matched_concept_synonym,
                CAST(c.concept_id AS VARCHAR)              AS concept_id,
                c.concept_name,
                c.vocabulary_id,
                c.concept_class_id,
                c.standard_concept,
                c.domain_id
            FROM (SELECT UNNEST([{terms_sql}]) AS term) i
            JOIN concept_synonym cs
                ON LOWER(TRIM(cs.concept_synonym_name)) = i.term
            JOIN concept c
                ON cs.concept_id = c.concept_id
            WHERE c.domain_id = 'Condition'
              AND c.invalid_reason IS NULL
            """

            df_part = self._query(sql)
            parts.append(df_part)

        df_exact_bq = (
            pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
        )

        df_exact = (
            df_input.select("condition_name", "search_term")
            .join(df_exact_bq, on="search_term", how="inner")
            .with_columns(
                pl.lit("exact").alias("match_type"),
                pl.lit(100).alias("match_score"),
            )
        )

        n_matched = df_exact["search_term"].n_unique()
        n_total = df_input["search_term"].n_unique()
        print(f"  Exact match: {n_matched}/{n_total} unique search terms")
        return df_exact

    # -------------------------------------------------------------------
    # Step 3: Fuzzy match via rapidfuzz
    # -------------------------------------------------------------------

    def _fuzzy_match(
        self,
        df_input: pl.DataFrame,
        df_exact: pl.DataFrame,
    ) -> pl.DataFrame:
        """Pull candidates from local vocab DB, then fuzzy-match locally."""
        matched_terms = set(df_exact["search_term"].unique().to_list())
        df_unmatched = df_input.filter(~pl.col("search_term").is_in(matched_terms))

        if len(df_unmatched) == 0:
            print("  All terms matched exactly. Skipping fuzzy step.")
            return pl.DataFrame(schema=df_exact.schema)

        print(f"  Unmatched terms for fuzzy matching: {len(df_unmatched)}")

        keywords = set()
        for term in df_unmatched["search_term"].to_list():
            keywords.update(w for w in term.split() if len(w) >= 4)

        safe_keywords = sorted(sql_escape(kw) for kw in keywords)
        keyword_clauses = " OR ".join(
            [f"LOWER(cs.concept_synonym_name) LIKE '%{kw}%'" for kw in safe_keywords]
        )

        sql = f"""
        SELECT DISTINCT
            CAST(cs.concept_id AS VARCHAR)             AS concept_id,
            LOWER(TRIM(cs.concept_synonym_name))       AS concept_synonym_lower,
            c.concept_name,
            c.vocabulary_id,
            c.concept_class_id,
            c.standard_concept
        FROM concept_synonym cs
        JOIN concept c
            ON cs.concept_id = c.concept_id
        WHERE c.domain_id = 'Condition'
          AND c.invalid_reason IS NULL
          AND ({keyword_clauses})
        """

        print("  Pulling fuzzy candidates from vocab DB...")
        df_candidates = self._query(sql)
        print(f"  Candidate concept synonyms: {len(df_candidates)}")

        if len(df_candidates) == 0:
            print("  No candidates found. Skipping fuzzy step.")
            return pl.DataFrame(schema=df_exact.schema)

        candidate_synonyms = df_candidates["concept_synonym_lower"].to_list()
        fuzzy_results = []

        for row in tqdm(
            df_unmatched.iter_rows(named=True),
            total=len(df_unmatched),
            desc="Fuzzy matching terms",
        ):
            term = row["search_term"]
            matches = process.extract(
                term,
                candidate_synonyms,
                scorer=fuzz.token_sort_ratio,
                limit=5,
                score_cutoff=self.fuzzy_threshold,
            )
            for matched_text, score, idx in matches:
                cand = df_candidates.row(idx, named=True)
                fuzzy_results.append(
                    {
                        "condition_name": row["condition_name"],
                        "search_term": term,
                        "matched_concept_synonym": matched_text,
                        "concept_id": str(cand["concept_id"]),
                        "concept_name": cand["concept_name"],
                        "vocabulary_id": cand["vocabulary_id"],
                        "concept_class_id": cand["concept_class_id"],
                        "standard_concept": cand["standard_concept"],
                        "domain_id": "Condition",
                        "match_type": "fuzzy",
                        "match_score": int(score),
                    }
                )

        if fuzzy_results:
            df_fuzzy = pl.DataFrame(fuzzy_results)
            print(
                f"  Fuzzy matches: {len(df_fuzzy)} rows "
                f"for {df_fuzzy['search_term'].n_unique()} terms"
            )
            return df_fuzzy
        else:
            print("  No fuzzy matches found.")
            return pl.DataFrame(schema=df_exact.schema)

    # -------------------------------------------------------------------
    # Step 4: Combine, rank, and SNOMED mapping
    # -------------------------------------------------------------------

    @staticmethod
    def _rank_matches(
        df_exact: pl.DataFrame, df_fuzzy: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Combine exact + fuzzy, rank by standard_concept then match_score."""
        df_all = pl.concat([df_exact, df_fuzzy], how="diagonal_relaxed")
        df_all = df_all.with_columns(pl.col("concept_id").cast(pl.Utf8))

        df_ranked = (
            df_all.with_columns(
                pl.when(pl.col("standard_concept") == "S")
                .then(0)
                .otherwise(1)
                .alias("_std_rank")
            )
            .sort(
                ["condition_name", "search_term", "_std_rank", "match_score"],
                descending=[False, False, False, True],
            )
            .with_columns(
                pl.col("concept_id")
                .rank("ordinal")
                .over(["condition_name", "search_term"])
                .alias("_rank")
            )
            .with_columns((pl.col("_rank") == 1).alias("is_best_match"))
            .drop("_std_rank", "_rank")
        )

        return df_all, df_ranked

    def _map_to_snomed(self, df_ranked: pl.DataFrame) -> pl.DataFrame:
        """For non-SNOMED concepts, look up SNOMED equivalent via 'Maps to'."""
        df_snomed_self = (
            df_ranked.filter(pl.col("vocabulary_id") == "SNOMED")
            .select(
                pl.col("concept_id"),
                pl.col("concept_id").alias("snomed_concept_id"),
                pl.col("concept_name").alias("snomed_concept_name"),
            )
            .unique()
        )

        non_snomed_ids = (
            df_ranked.filter(pl.col("vocabulary_id") != "SNOMED")
            .select("concept_id")
            .unique()["concept_id"]
            .to_list()
        )

        print(f"  SNOMED concepts (self-map): {len(df_snomed_self)}")
        print(f"  Non-SNOMED concepts to map: {len(non_snomed_ids)}")

        if non_snomed_ids:
            id_batches = [
                non_snomed_ids[i : i + self.BATCH_SIZE]
                for i in range(0, len(non_snomed_ids), self.BATCH_SIZE)
            ]
            mapping_parts = []

            for batch in tqdm(id_batches, desc="SNOMED mapping batches"):
                ids_sql = ", ".join([f"'{sql_escape(str(cid))}'" for cid in batch])

                sql = f"""
                SELECT DISTINCT
                    CAST(cr.concept_id_1 AS VARCHAR) AS concept_id,
                    CAST(c2.concept_id AS VARCHAR)   AS snomed_concept_id,
                    c2.concept_name                  AS snomed_concept_name
                FROM (SELECT UNNEST([{ids_sql}]) AS id) s
                JOIN concept_relationship cr
                    ON CAST(cr.concept_id_1 AS VARCHAR) = s.id
                JOIN concept c2
                    ON cr.concept_id_2 = c2.concept_id
                WHERE cr.relationship_id = 'Maps to'
                  AND c2.standard_concept = 'S'
                  AND c2.domain_id = 'Condition'
                  AND c2.invalid_reason IS NULL
                  AND cr.invalid_reason IS NULL
                """

                df_part = self._query(sql)
                mapping_parts.append(df_part)

            df_snomed_map = (
                pl.concat(mapping_parts, how="diagonal_relaxed")
                if mapping_parts
                else pl.DataFrame()
            )

            if len(df_snomed_map) > 0:
                n_mapped = df_snomed_map["concept_id"].n_unique()
                n_unmapped = len(non_snomed_ids) - n_mapped
            else:
                n_unmapped = len(non_snomed_ids)

            if n_unmapped > 0:
                print(f"  Non-SNOMED concepts with no SNOMED mapping: {n_unmapped}")
        else:
            df_snomed_map = pl.DataFrame(
                schema={
                    "concept_id": pl.Utf8,
                    "snomed_concept_id": pl.Utf8,
                    "snomed_concept_name": pl.Utf8,
                }
            )

        df_snomed_lookup = pl.concat(
            [df_snomed_self, df_snomed_map], how="diagonal_relaxed"
        ).unique()
        df_ranked = df_ranked.join(df_snomed_lookup, on="concept_id", how="left")

        print(
            f"  Unique SNOMED concept IDs: "
            f"{df_ranked['snomed_concept_id'].drop_nulls().n_unique()}"
        )
        return df_ranked

    # -------------------------------------------------------------------
    # Step 5: Summarise
    # -------------------------------------------------------------------

    @staticmethod
    def _summarise(
        df_input: pl.DataFrame,
        df_all: pl.DataFrame,
        df_exact: pl.DataFrame,
        df_fuzzy: pl.DataFrame,
        df_ranked: pl.DataFrame,
    ) -> dict:
        """Compute term counts, condition-level summary, and unmatched lists."""
        df_term_counts = (
            df_input.group_by("condition_name").agg(
                pl.col("search_term").n_unique().alias("total_terms")
            )
        )

        all_matched_terms = set(df_all["search_term"].unique().to_list())

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
        conditions_with_any_match = set(df_all["condition_name"].unique().to_list())
        conditions_no_match = all_conditions - conditions_with_any_match

        df_condition_summary = (
            df_ranked.filter(pl.col("is_best_match"))
            .filter(pl.col("snomed_concept_id").is_not_null())
            .group_by("condition_name")
            .agg(
                [
                    pl.col("snomed_concept_id").unique().sort().str.join(", ").alias("snomed_concept_ids"),
                    pl.col("snomed_concept_name").unique().sort().str.join(", ").alias("snomed_concept_names"),
                    pl.col("concept_id").unique().sort().str.join(", ").alias("source_concept_ids"),
                    pl.col("vocabulary_id").unique().sort().str.join(", ").alias("source_vocabularies"),
                    pl.col("search_term").unique().sort().str.join(", ").alias("matched_via"),
                    pl.col("match_type").first().alias("primary_match_type"),
                    pl.col("match_score").min().alias("lowest_score"),
                ]
            )
            .join(df_term_counts, on="condition_name", how="left")
            .sort("lowest_score")
        )

        df_unmatched_conditions = (
            df_input.filter(pl.col("condition_name").is_in(conditions_no_match))
            .select("condition_name")
            .unique()
            .join(df_term_counts, on="condition_name", how="left")
        )

        still_unmatched = df_input.filter(~pl.col("search_term").is_in(all_matched_terms))

        print(f"\n{'=' * 40}")
        print(f"  MAPPING SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Total conditions:             {len(all_conditions)}")
        print(f"  Conditions with >=1 match:    {len(conditions_with_any_match)}")
        print(f"  Conditions with 0 matches:    {len(conditions_no_match)}")
        print(f"  Unique SNOMED concept IDs:    {df_ranked['snomed_concept_id'].drop_nulls().n_unique()}")
        print(f"  Total search terms:           {df_input['search_term'].n_unique()}")
        print(f"  Terms matched (exact):        {df_exact['search_term'].n_unique()}")
        print(f"  Terms matched (fuzzy):        {df_fuzzy['search_term'].n_unique() if len(df_fuzzy) > 0 else 0}")
        print(f"  Terms still unmatched:        {len(still_unmatched)}")
        print(f"{'=' * 40}")

        return {
            "df_term_counts": df_term_counts,
            "df_condition_summary": df_condition_summary,
            "df_unmatched_conditions": df_unmatched_conditions,
        }

    # -------------------------------------------------------------------
    # Step 6: Biological enrichment via concept_ancestor
    # -------------------------------------------------------------------

    def enrich(
        self,
        results: dict,
        single_domain: bool = False,
        ai_review_batch_size: Optional[int] = None,
    ) -> dict:
        """Enrich mapping results with biological validation data.

        Parameters
        ----------
        results : dict
            Output from map().
        single_domain : bool
            If True, assumes all conditions share a clinical domain.
            Enables cross-condition domain anchor discovery. Default False.
        ai_review_batch_size : int
            Conditions per AI review API call (for estimating calls). Default 5.

        Returns
        -------
        dict
            Updated results with df_ranked including:
            ancestor_distance, nearest_ancestor_name, finding_site,
            associated_morphology, validation_status
        """
        df_ranked = results["df_ranked"].clone()

        snomed_ids = (
            df_ranked.filter(pl.col("snomed_concept_id").is_not_null())
            .select("snomed_concept_id")
            .unique()["snomed_concept_id"]
            .to_list()
        )

        if not snomed_ids:
            print("  No SNOMED concepts to enrich.")
            results["df_ranked"] = df_ranked
            return results

        # --- A. Get ancestors ---
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

        # --- B. Get finding_site and morphology ---
        print("  Querying concept_relationship (finding_site, morphology)...")
        attr_parts = []
        for batch in tqdm(id_batches, desc="Attribute queries"):
            ids_sql = ", ".join([f"'{sql_escape(cid)}'" for cid in batch])

            sql = f"""
            SELECT DISTINCT
                CAST(cr.concept_id_1 AS VARCHAR) AS snomed_concept_id,
                cr.relationship_id,
                c2.concept_name                  AS related_concept_name
            FROM concept_relationship cr
            JOIN concept c2
                ON cr.concept_id_2 = c2.concept_id
            WHERE CAST(cr.concept_id_1 AS VARCHAR) IN ({ids_sql})
              AND cr.relationship_id IN ('Finding site', 'Associated morphology')
              AND cr.invalid_reason IS NULL
            """

            df_part = self._query(sql)
            attr_parts.append(df_part)

        df_attrs = (
            pl.concat(attr_parts, how="diagonal_relaxed")
            if attr_parts
            else pl.DataFrame()
        )

        df_finding_site = pl.DataFrame()
        df_morphology = pl.DataFrame()

        if len(df_attrs) > 0:
            df_finding_site = (
                df_attrs.filter(pl.col("relationship_id") == "Finding site")
                .group_by("snomed_concept_id")
                .agg(pl.col("related_concept_name").unique().sort().str.join(", ").alias("finding_site"))
            )
            df_morphology = (
                df_attrs.filter(pl.col("relationship_id") == "Associated morphology")
                .group_by("snomed_concept_id")
                .agg(pl.col("related_concept_name").unique().sort().str.join(", ").alias("associated_morphology"))
            )

        # --- C. Per-condition validation ---
        print("  Computing per-condition validation...")

        df_refs = (
            df_ranked.filter(
                (pl.col("match_type") == "exact") | (pl.col("match_score") >= 90)
            )
            .filter(pl.col("snomed_concept_id").is_not_null())
            .select("condition_name", "snomed_concept_id")
            .unique()
        )

        # Domain anchor discovery (single_domain mode)
        domain_anchors = set()
        if single_domain and len(df_ancestors) > 0 and len(df_refs) > 0:
            print("  Discovering domain anchors (single_domain=True)...")
            ref_ids = set(df_refs["snomed_concept_id"].to_list())
            n_conditions_with_refs = df_refs["condition_name"].n_unique()

            df_ref_ancestors = df_ancestors.filter(pl.col("snomed_concept_id").is_in(ref_ids))

            if len(df_ref_ancestors) > 0:
                df_ref_with_condition = df_refs.join(df_ref_ancestors, on="snomed_concept_id", how="inner")
                df_anchor_coverage = (
                    df_ref_with_condition.group_by("ancestor_concept_id", "ancestor_name")
                    .agg(pl.col("condition_name").n_unique().alias("n_conditions"))
                    .with_columns((pl.col("n_conditions") / n_conditions_with_refs).alias("coverage"))
                    .filter(pl.col("coverage") >= 0.5)
                    .sort("coverage", descending=True)
                )

                if len(df_anchor_coverage) > 0:
                    anchor_ids = df_anchor_coverage["ancestor_concept_id"].to_list()
                    anchor_ids_sql = ", ".join([f"'{sql_escape(a)}'" for a in anchor_ids])

                    breadth_sql = f"""
                    SELECT
                        CAST(ancestor_concept_id AS VARCHAR) AS ancestor_concept_id,
                        COUNT(DISTINCT descendant_concept_id) AS n_descendants
                    FROM concept_ancestor
                    WHERE CAST(ancestor_concept_id AS VARCHAR) IN ({anchor_ids_sql})
                    GROUP BY ancestor_concept_id
                    HAVING COUNT(DISTINCT descendant_concept_id) <= 10000
                    """

                    df_valid_anchors = self._query(breadth_sql)
                    domain_anchors = set(df_valid_anchors["ancestor_concept_id"].to_list())
                    print(f"  Domain anchors found: {len(domain_anchors)}")

        # Compute validation per match
        validation_rows = []
        conditions_list = df_ranked["condition_name"].unique().to_list()

        for cond in tqdm(conditions_list, desc="Validating conditions"):
            cond_refs = df_refs.filter(pl.col("condition_name") == cond)
            cond_matches = df_ranked.filter(
                (pl.col("condition_name") == cond)
                & (pl.col("snomed_concept_id").is_not_null())
            )
            ref_ids = set(cond_refs["snomed_concept_id"].to_list())

            for match_row in cond_matches.iter_rows(named=True):
                match_snomed = match_row["snomed_concept_id"]
                search_term = match_row["search_term"]

                if match_snomed in ref_ids:
                    validation_rows.append({
                        "condition_name": cond, "search_term": search_term,
                        "snomed_concept_id": match_snomed,
                        "ancestor_distance": 0, "nearest_ancestor_name": "(self-reference)",
                        "validation_status": "confirmed",
                    })
                    continue

                if not ref_ids:
                    status = "no_reference"
                    if domain_anchors and len(df_ancestors) > 0:
                        match_ancestors = set(
                            df_ancestors.filter(pl.col("snomed_concept_id") == match_snomed)
                            ["ancestor_concept_id"].to_list()
                        )
                        if match_ancestors & domain_anchors:
                            status = "plausible"
                    validation_rows.append({
                        "condition_name": cond, "search_term": search_term,
                        "snomed_concept_id": match_snomed,
                        "ancestor_distance": None, "nearest_ancestor_name": None,
                        "validation_status": status,
                    })
                    continue

                if len(df_ancestors) == 0:
                    validation_rows.append({
                        "condition_name": cond, "search_term": search_term,
                        "snomed_concept_id": match_snomed,
                        "ancestor_distance": None, "nearest_ancestor_name": None,
                        "validation_status": "weak_match",
                    })
                    continue

                match_anc = df_ancestors.filter(pl.col("snomed_concept_id") == match_snomed)
                match_ancestor_ids = set(match_anc["ancestor_concept_id"].to_list())

                best_dist = None
                best_ancestor = None

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

                if best_dist is None:
                    status = "weak_match"
                elif best_dist <= 4:
                    status = "confirmed"
                elif best_dist <= 8:
                    status = "plausible"
                else:
                    status = "weak_match"

                validation_rows.append({
                    "condition_name": cond, "search_term": search_term,
                    "snomed_concept_id": match_snomed,
                    "ancestor_distance": best_dist, "nearest_ancestor_name": best_ancestor,
                    "validation_status": status,
                })

        # Join results
        if validation_rows:
            df_validation = pl.DataFrame(validation_rows)
            df_ranked = df_ranked.join(
                df_validation,
                on=["condition_name", "search_term", "snomed_concept_id"],
                how="left",
            )
        else:
            df_ranked = df_ranked.with_columns(
                pl.lit(None).cast(pl.Int64).alias("ancestor_distance"),
                pl.lit(None).cast(pl.Utf8).alias("nearest_ancestor_name"),
                pl.lit("no_reference").alias("validation_status"),
            )

        if len(df_finding_site) > 0:
            df_ranked = df_ranked.join(df_finding_site, on="snomed_concept_id", how="left")
        else:
            df_ranked = df_ranked.with_columns(pl.lit(None).cast(pl.Utf8).alias("finding_site"))

        if len(df_morphology) > 0:
            df_ranked = df_ranked.join(df_morphology, on="snomed_concept_id", how="left")
        else:
            df_ranked = df_ranked.with_columns(pl.lit(None).cast(pl.Utf8).alias("associated_morphology"))

        # Summary
        if validation_rows:
            df_val = pl.DataFrame(validation_rows)
            status_order = ["confirmed", "plausible", "weak_match", "no_reference"]
            status_counts = dict(
                df_val.group_by("validation_status").len().iter_rows()
            )
            print(f"\n  Validation status distribution:")
            for status in status_order:
                if status in status_counts:
                    print(f"    {status}: {status_counts[status]}")

            review_rows = [
                r for r in validation_rows
                if r["validation_status"] in ("weak_match", "plausible", "no_reference")
            ]
            n_hits_review = len(review_rows)
            n_conditions_review = len(set(r["condition_name"] for r in review_rows))
            if ai_review_batch_size is not None:
                est_calls = math.ceil(n_conditions_review / ai_review_batch_size)
                print(f"\n  AI review estimate: {n_conditions_review} conditions, {n_hits_review} hits → ~{est_calls} API calls")
            else:
                print(f"\n  AI review estimate: {n_conditions_review} conditions, {n_hits_review} hits (batch size auto-calculated)")

        results["df_ranked"] = df_ranked
        return results

    # -------------------------------------------------------------------
    # Step 7: AI review via Gemini
    # -------------------------------------------------------------------

    # Estimated tokens per condition for batch size calculation
    # Input: ~100 tokens (condition name + match lines)
    # Output: ~50 tokens (verdict JSON per condition)
    _TOKENS_PER_CONDITION_INPUT = 100
    _TOKENS_PER_CONDITION_OUTPUT = 50
    _PROMPT_OVERHEAD_TOKENS = 200
    _CONTEXT_WINDOW = 1_000_000
    _MAX_OUTPUT_TOKENS = 65_536

    def _calculate_batch_size(self, n_matches_per_condition: float) -> int:
        """Calculate max batch size that fits within model limits."""
        input_per_cond = self._TOKENS_PER_CONDITION_INPUT * max(1, n_matches_per_condition)
        output_per_cond = self._TOKENS_PER_CONDITION_OUTPUT * max(1, n_matches_per_condition)

        max_by_context = (self._CONTEXT_WINDOW - self._PROMPT_OVERHEAD_TOKENS) // (input_per_cond + output_per_cond)
        max_by_output = self._MAX_OUTPUT_TOKENS // output_per_cond

        return int(max(1, min(max_by_context, max_by_output, 50)))

    def ai_review(
        self,
        results: dict,
        to_be_reviewed: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
        confidence_threshold: int = 80,
    ) -> dict:
        """AI-assisted review of ambiguous matches using Gemini API.

        Parameters
        ----------
        results : dict
            Output from enrich() (or map()).
        to_be_reviewed : list[str], optional
            Validation statuses to send for AI review.
            Default: ["weak_match", "plausible", "no_reference"].
        batch_size : int, optional
            Conditions per API call. If None, auto-calculated from model limits.
        confidence_threshold : int
            Confidence score (0-100) below which verdicts are flagged as
            "human review". Default 80.

        Returns
        -------
        dict
            Updated results with ai_verdict, ai_reason, and ai_confidence columns.
        """
        model = self._resolve_model()
        df_ranked = results["df_ranked"].clone()

        if to_be_reviewed is None:
            to_be_reviewed = ["weak_match", "plausible", "no_reference"]

        if "validation_status" in df_ranked.columns:
            df_to_review = df_ranked.filter(
                pl.col("validation_status").is_in(to_be_reviewed)
                & pl.col("snomed_concept_id").is_not_null()
            )
        else:
            df_to_review = df_ranked.filter(pl.col("match_type") == "fuzzy")

        if len(df_to_review) == 0:
            print("  No matches require AI review.")
            df_ranked = df_ranked.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("ai_verdict"),
                pl.lit(None).cast(pl.Utf8).alias("ai_reason"),
                pl.lit(None).cast(pl.Int64).alias("ai_confidence"),
            )
            results["df_ranked"] = df_ranked
            return results

        conditions_to_review = df_to_review["condition_name"].unique().to_list()
        n_conditions = len(conditions_to_review)

        # Auto-calculate batch size if not provided
        user_batch_size = batch_size is not None
        if not user_batch_size:
            avg_matches = len(df_to_review) / max(1, n_conditions)
            batch_size = self._calculate_batch_size(avg_matches)

        est_calls = math.ceil(n_conditions / batch_size)

        # Per-status hit counts
        if "validation_status" in df_to_review.columns:
            status_counts = dict(
                df_to_review.group_by("validation_status").len().iter_rows()
            )
            status_parts = [
                f"{s}: {status_counts[s]}" for s in to_be_reviewed if s in status_counts
            ]
            print(f"  Reviewing: {', '.join(status_parts)}")
        else:
            print(f"  Reviewing: {len(df_to_review)} fuzzy matches")

        print(f"  Conditions for AI review: {n_conditions}")
        print(f"  Batch size: {batch_size} ({'user-specified' if user_batch_size else 'auto-calculated'})")
        print(f"  Estimated API calls: {est_calls}")
        print(f"  Model: {model}")

        cond_batches = [
            conditions_to_review[i : i + batch_size]
            for i in range(0, len(conditions_to_review), batch_size)
        ]

        # Schema for structured Gemini response
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "verdicts": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "condition": {"type": "STRING"},
                            "id": {"type": "STRING"},
                            "v": {"type": "STRING", "enum": ["accept", "reject"]},
                            "r": {"type": "STRING"},
                            "c": {"type": "INTEGER"},
                        },
                        "required": ["condition", "id", "v", "r", "c"],
                    },
                }
            },
            "required": ["verdicts"],
        }

        all_verdicts = []
        calls_used = 0

        for batch in tqdm(cond_batches, desc="AI review batches"):
            prompt_parts = [
                "You are validating SNOMED concept matches for clinical conditions.\n"
                "For each condition-match pair, decide if the SNOMED concept is a "
                "clinically appropriate match for the condition.\n"
                "Accept if the concept captures the same clinical meaning. "
                "Reject if it refers to a different condition, wrong body site, "
                "wrong specificity, or unrelated finding.\n"
                "Rate confidence 0-100 (100 = certain).\n"
                "Reason must be 10 words or fewer.\n\n"
            ]

            for cond in batch:
                cond_rows = df_to_review.filter(pl.col("condition_name") == cond)
                matches_lines = []
                for row in cond_rows.iter_rows(named=True):
                    parts_line = [
                        f"id={row['snomed_concept_id']}",
                        f"name={row.get('snomed_concept_name', '')}",
                        f"fuzzy={row['match_score']}",
                    ]
                    if row.get("ancestor_distance") is not None:
                        parts_line.append(f"ancestor_dist={row['ancestor_distance']}")
                    if row.get("finding_site"):
                        parts_line.append(f"site={row['finding_site']}")
                    if row.get("associated_morphology"):
                        parts_line.append(f"morphology={row['associated_morphology']}")
                    matches_lines.append(" | ".join(parts_line))

                prompt_parts.append(f"Condition: {cond}")
                prompt_parts.append("Matches:")
                prompt_parts.extend(matches_lines)
                prompt_parts.append("")

            prompt = "\n".join(prompt_parts)

            try:
                response_text = call_gemini(
                    prompt, check_api_key(self._api_key), model,
                    response_schema=response_schema,
                )
                calls_used += 1

                parsed = json.loads(response_text.strip())
                verdicts = parsed.get("verdicts", [])
                all_verdicts.extend(verdicts)
            except Exception as e:
                calls_used += 1
                print(f"  AI review error for batch: {e}")

        if all_verdicts:
            verdict_rows = [
                {
                    "condition_name": v.get("condition", ""),
                    "snomed_concept_id": str(v.get("id", "")),
                    "ai_verdict": v.get("v", "human review"),
                    "ai_reason": v.get("r", ""),
                    "ai_confidence": v.get("c"),
                }
                for v in all_verdicts
            ]
            df_verdicts = pl.DataFrame(verdict_rows).unique(
                subset=["condition_name", "snomed_concept_id"]
            )

            # Flag low-confidence verdicts as "human review"
            df_verdicts = df_verdicts.with_columns(
                pl.when(
                    pl.col("ai_confidence").is_not_null()
                    & (pl.col("ai_confidence") < confidence_threshold)
                )
                .then(pl.lit("human review"))
                .otherwise(pl.col("ai_verdict"))
                .alias("ai_verdict")
            )

            df_ranked = df_ranked.join(
                df_verdicts, on=["condition_name", "snomed_concept_id"], how="left"
            )
        else:
            df_ranked = df_ranked.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("ai_verdict"),
                pl.lit(None).cast(pl.Utf8).alias("ai_reason"),
                pl.lit(None).cast(pl.Int64).alias("ai_confidence"),
            )

        # Count final verdicts (only rows that were sent for review)
        df_reviewed = df_ranked.filter(
            pl.col("ai_verdict").is_not_null()
            & pl.col("validation_status").is_in(to_be_reviewed)
        )

        n_accepted_hits = len(df_reviewed.filter(pl.col("ai_verdict") == "accept"))
        n_rejected_hits = len(df_reviewed.filter(pl.col("ai_verdict") == "reject"))
        n_human_review_hits = len(df_reviewed.filter(pl.col("ai_verdict") == "human review"))
        n_total = n_accepted_hits + n_rejected_hits + n_human_review_hits

        n_cond_accepted = df_reviewed.filter(
            pl.col("ai_verdict") == "accept"
        )["condition_name"].n_unique()
        conds_with_accept = set(
            df_reviewed.filter(pl.col("ai_verdict") == "accept")["condition_name"].to_list()
        )
        n_cond_rejected = df_reviewed.filter(
            (pl.col("ai_verdict") == "reject")
            & ~pl.col("condition_name").is_in(list(conds_with_accept))
        )["condition_name"].n_unique()
        n_cond_human_review = df_reviewed.filter(
            pl.col("ai_verdict") == "human review"
        )["condition_name"].n_unique()

        print(f"\n{'=' * 40}")
        print(f"  AI REVIEW SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Model used:        {model}")
        print(f"  API calls used:    {calls_used}")
        print(f"  Confidence cutoff: {confidence_threshold}")
        print(f"  Total hits reviewed: {n_total}")
        print(f"    accepted:        {n_accepted_hits} hits ({n_cond_accepted} conditions with at least 1 hit)")
        print(f"    rejected:        {n_rejected_hits} hits ({n_cond_rejected} conditions lost all hits)")
        print(f"    human review:    {n_human_review_hits} hits ({n_cond_human_review} conditions)")
        print(f"{'=' * 40}")

        results["df_ranked"] = df_ranked
        return results

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def map(
        self,
        conditions: dict[str, list[str]],
        export_tsv: bool = False,
        export_prefix: str = "mapping",
    ) -> dict:
        """
        Map condition names and their synonyms to OMOP Standard SNOMED Concept IDs.

        Parameters
        ----------
        conditions : dict[str, list[str]]
            Keys are condition names; values are lists of condition synonyms.
        export_tsv : bool
            If True, write TSV files with UTF-8 BOM. Default False.
        export_prefix : str
            Filename prefix for exported TSV files. Default "mapping".

        Returns
        -------
        dict with keys:
            df_input, df_exact, df_fuzzy, df_ranked,
            df_condition_summary, df_unmatched_conditions, df_term_counts
        """
        print("[1/5] Building search terms...")
        df_input = self._build_input(conditions)
        print(
            f"  Conditions: {df_input['condition_name'].n_unique()}, "
            f"Search terms: {df_input['search_term'].n_unique()}"
        )

        print("[2/5] Exact matching against concept_synonym...")
        df_exact = self._exact_match(df_input)

        print("[3/5] Fuzzy matching unmatched terms...")
        df_fuzzy = self._fuzzy_match(df_input, df_exact)

        print("[4/5] Ranking and SNOMED mapping...")
        df_all, df_ranked = self._rank_matches(df_exact, df_fuzzy)
        df_ranked = self._map_to_snomed(df_ranked)

        print("[5/5] Summarising results...")
        summary = self._summarise(df_input, df_all, df_exact, df_fuzzy, df_ranked)

        results = {
            "df_input": df_input,
            "df_exact": df_exact,
            "df_fuzzy": df_fuzzy,
            "df_ranked": df_ranked,
            "df_condition_summary": summary["df_condition_summary"],
            "df_unmatched_conditions": summary["df_unmatched_conditions"],
            "df_term_counts": summary["df_term_counts"],
        }

        if export_tsv:
            print(f"\nExporting TSV files (prefix: '{export_prefix}_')...")
            write_tsv_bom(df_ranked, f"{export_prefix}_full_review.tsv")
            write_tsv_bom(summary["df_condition_summary"], f"{export_prefix}_condition_summary.tsv")
            if len(summary["df_unmatched_conditions"]) > 0:
                write_tsv_bom(summary["df_unmatched_conditions"], f"{export_prefix}_unmatched_conditions.tsv")
            print("  Done.")

        return results
