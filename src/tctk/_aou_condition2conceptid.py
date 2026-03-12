"""
Module: _aou_condition2conceptid

Map condition names and their synonyms to OMOP Standard SNOMED Concept IDs
via the concept_synonym table in BigQuery (read-only).

Pipeline:
    1. Normalize input terms (lowercase, strip tags, expand parentheticals)
    2. Exact match against concept_synonym + concept via UNNEST
    3. Fuzzy match unmatched terms via rapidfuzz (token_sort_ratio)
    4. Map non-SNOMED matches to SNOMED via concept_relationship ('Maps to')
    5. Rank and return results

Usage:
    from tctk._aou_condition2conceptid import Condition2ConceptID

    mapper = Condition2ConceptID()

    conditions = {
        "Cicatricial pemphigoid": [
            "Benign mucosal pemphigoid",
            "Mucous membrane pemphigoid",
            "Ocular cicatricial pemphigoid (subtype)",
        ],
        "Lupus": ["SLE", "Systemic lupus erythematosus"],
    }

    results = mapper.map(conditions)
    results["df_ranked"]                # full ranked matches
    results["df_condition_summary"]     # per-condition SNOMED summary
    results["df_unmatched_conditions"]  # conditions with 0 matches
    results["df_term_counts"]           # per-condition term match counts
"""

import os
from typing import Optional

import polars as pl
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm
import tctk.polars_tools as pt

__all__ = ["Condition2ConceptID"]


class Condition2ConceptID:
    """Map condition names and synonyms to OMOP Standard SNOMED Concept IDs.

    Parameters
    ----------
    ds : str, optional
        BigQuery dataset prefix for OMOP vocab tables.
        Defaults to os.getenv("WORKSPACE_CDR").
    bucket : str, optional
        GCS bucket path. Defaults to os.getenv("WORKSPACE_BUCKET").
    fuzzy_threshold : int
        Minimum score (0-100) for rapidfuzz token_sort_ratio. Default 85.
    """

    BATCH_SIZE = 500

    def __init__(
        self,
        ds: Optional[str] = None,
        bucket: Optional[str] = None,
        fuzzy_threshold: int = 85,
    ):
        self.ds = ds or os.getenv("WORKSPACE_CDR")
        self.bucket = bucket or os.getenv("WORKSPACE_BUCKET")
        self.fuzzy_threshold = fuzzy_threshold

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _sql_escape(s: str) -> str:
        """Escape backslashes and single quotes for BigQuery string literals."""
        return s.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def write_tsv_bom(df: pl.DataFrame, path: str) -> None:
        """Write a Polars DataFrame as TSV with UTF-8 BOM for Excel/Mac compatibility."""
        with open(path, "wb") as f:
            f.write(b"\xef\xbb\xbf")
            f.write(df.write_csv(separator="\t").encode("utf-8"))

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
            all_terms = [condition_name] + [s.strip() for s in condition_synonyms if s.strip()]
            for term in all_terms:
                rows.append({"condition_name": condition_name, "search_term_raw": term})

        df = pl.DataFrame(rows)

        # --- Normalize ---
        df_normalized = (
            df
            .with_columns(
                pl.col("search_term_raw")
                .str.to_lowercase()
                .str.strip_chars()
                .str.replace_all(r"\s*\(subtype\)\.?", "")
                .str.replace_all(r"\s*\(synonym\)\.?", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias("search_term")
            )
            .filter(pl.col("search_term") != "")
        )

        # --- Expand parenthetical terms into two variants ---
        df_has_parens = df_normalized.filter(pl.col("search_term").str.contains(r"\(.*\)"))
        df_no_parens = df_normalized.filter(~pl.col("search_term").str.contains(r"\(.*\)"))

        df_parens_stripped = (
            df_has_parens
            .with_columns(
                pl.col("search_term")
                .str.replace_all(r"\s*\([^)]*\)", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias("search_term")
            )
            .filter(pl.col("search_term") != "")
        )

        df_input = (
            pl.concat([df_no_parens, df_has_parens, df_parens_stripped], how="diagonal_relaxed")
            .unique(subset=["condition_name", "search_term"])
            .select("condition_name", "search_term")
        )

        return df_input

    # -------------------------------------------------------------------
    # Step 2: Exact match via concept_synonym
    # -------------------------------------------------------------------

    def _exact_match(self, df_input: pl.DataFrame) -> pl.DataFrame:
        """Query BQ concept_synonym for exact (case-insensitive) matches."""
        unique_terms = df_input["search_term"].unique().sort().to_list()
        term_batches = [
            unique_terms[i:i + self.BATCH_SIZE]
            for i in range(0, len(unique_terms), self.BATCH_SIZE)
        ]

        parts = []
        for batch in tqdm(term_batches, desc="Exact match batches"):
            terms_sql = ", ".join([f"'{self._sql_escape(t)}'" for t in batch])

            sql = f"""
            WITH input_terms AS (
                SELECT term
                FROM UNNEST([{terms_sql}]) AS term
            )
            SELECT
                i.term                                       AS search_term,
                LOWER(TRIM(cs.concept_synonym_name))         AS matched_concept_synonym,
                CAST(c.concept_id AS STRING)                 AS concept_id,
                c.concept_name,
                c.vocabulary_id,
                c.concept_class_id,
                c.standard_concept,
                c.domain_id
            FROM input_terms i
            JOIN {self.ds}.concept_synonym AS cs
                ON LOWER(TRIM(cs.concept_synonym_name)) = i.term
            JOIN {self.ds}.concept AS c
                ON cs.concept_id = c.concept_id
            WHERE c.domain_id = 'Condition'
              AND c.invalid_reason IS NULL
            """

            df_part = pt.polars_gbq(sql)
            parts.append(df_part)

        df_exact_bq = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()

        # Join back to input to attach condition_name
        df_exact = (
            df_input
            .select("condition_name", "search_term")
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
        """Pull candidates from BQ, then fuzzy-match unmatched terms locally."""
        matched_terms = set(df_exact["search_term"].unique().to_list())
        df_unmatched = df_input.filter(~pl.col("search_term").is_in(matched_terms))

        if len(df_unmatched) == 0:
            print("  All terms matched exactly. Skipping fuzzy step.")
            return pl.DataFrame(schema=df_exact.schema)

        print(f"  Unmatched terms for fuzzy matching: {len(df_unmatched)}")

        # Extract keywords (>=4 chars) for BQ pre-filter
        keywords = set()
        for term in df_unmatched["search_term"].to_list():
            keywords.update(w for w in term.split() if len(w) >= 4)

        safe_keywords = sorted(self._sql_escape(kw) for kw in keywords)
        keyword_clauses = " OR ".join(
            [f"LOWER(cs.concept_synonym_name) LIKE '%{kw}%'" for kw in safe_keywords]
        )

        sql = f"""
        SELECT DISTINCT
            CAST(cs.concept_id AS STRING)                AS concept_id,
            LOWER(TRIM(cs.concept_synonym_name))         AS concept_synonym_lower,
            c.concept_name,
            c.vocabulary_id,
            c.concept_class_id,
            c.standard_concept
        FROM {self.ds}.concept_synonym AS cs
        JOIN {self.ds}.concept AS c
            ON cs.concept_id = c.concept_id
        WHERE c.domain_id = 'Condition'
          AND c.invalid_reason IS NULL
          AND ({keyword_clauses})
        """

        print("  Pulling fuzzy candidates from BigQuery...")
        df_candidates = pt.polars_gbq(sql)
        print(f"  Candidate concept synonyms pulled: {len(df_candidates)}")

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
                fuzzy_results.append({
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
                })

        if fuzzy_results:
            df_fuzzy = pl.DataFrame(fuzzy_results)
            print(f"  Fuzzy matches: {len(df_fuzzy)} rows for {df_fuzzy['search_term'].n_unique()} terms")
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
            df_all
            .with_columns(
                pl.when(pl.col("standard_concept") == "S").then(0).otherwise(1).alias("_std_rank")
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
        # Self-mapping for SNOMED concepts
        df_snomed_self = (
            df_ranked
            .filter(pl.col("vocabulary_id") == "SNOMED")
            .select(
                pl.col("concept_id"),
                pl.col("concept_id").alias("snomed_concept_id"),
                pl.col("concept_name").alias("snomed_concept_name"),
            )
            .unique()
        )

        # Non-SNOMED concepts
        non_snomed_ids = (
            df_ranked
            .filter(pl.col("vocabulary_id") != "SNOMED")
            .select("concept_id")
            .unique()
            ["concept_id"].to_list()
        )

        print(f"  SNOMED concepts (self-map): {len(df_snomed_self)}")
        print(f"  Non-SNOMED concepts to map: {len(non_snomed_ids)}")

        if non_snomed_ids:
            id_batches = [
                non_snomed_ids[i:i + self.BATCH_SIZE]
                for i in range(0, len(non_snomed_ids), self.BATCH_SIZE)
            ]
            mapping_parts = []

            for batch in tqdm(id_batches, desc="SNOMED mapping batches"):
                ids_sql = ", ".join([f"'{self._sql_escape(str(cid))}'" for cid in batch])

                sql = f"""
                WITH source_ids AS (
                    SELECT id FROM UNNEST([{ids_sql}]) AS id
                )
                SELECT DISTINCT
                    CAST(cr.concept_id_1 AS STRING)  AS concept_id,
                    CAST(c2.concept_id AS STRING)    AS snomed_concept_id,
                    c2.concept_name                  AS snomed_concept_name
                FROM source_ids s
                JOIN {self.ds}.concept_relationship AS cr
                    ON CAST(cr.concept_id_1 AS STRING) = s.id
                JOIN {self.ds}.concept AS c2
                    ON cr.concept_id_2 = c2.concept_id
                WHERE cr.relationship_id = 'Maps to'
                  AND c2.standard_concept = 'S'
                  AND c2.domain_id = 'Condition'
                  AND c2.invalid_reason IS NULL
                  AND cr.invalid_reason IS NULL
                """

                df_part = pt.polars_gbq(sql)
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
                schema={"concept_id": pl.Utf8, "snomed_concept_id": pl.Utf8, "snomed_concept_name": pl.Utf8}
            )

        # Combine and join
        df_snomed_lookup = pl.concat([df_snomed_self, df_snomed_map], how="diagonal_relaxed").unique()
        df_ranked = df_ranked.join(df_snomed_lookup, on="concept_id", how="left")

        print(f"  Unique SNOMED concept IDs: {df_ranked['snomed_concept_id'].drop_nulls().n_unique()}")
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
        # Per-condition term counts
        df_term_counts = (
            df_input
            .group_by("condition_name")
            .agg(pl.col("search_term").n_unique().alias("total_terms"))
        )

        all_matched_terms = set(df_all["search_term"].unique().to_list())

        df_matched_counts = (
            df_input
            .filter(pl.col("search_term").is_in(all_matched_terms))
            .group_by("condition_name")
            .agg(pl.col("search_term").n_unique().alias("matched_terms"))
        )

        df_term_counts = (
            df_term_counts
            .join(df_matched_counts, on="condition_name", how="left")
            .with_columns(pl.col("matched_terms").fill_null(0))
            .with_columns(
                (pl.col("total_terms") - pl.col("matched_terms")).alias("unmatched_terms")
            )
        )

        # Condition-level match status
        all_conditions = set(df_input["condition_name"].unique().to_list())
        conditions_with_any_match = set(df_all["condition_name"].unique().to_list())
        conditions_no_match = all_conditions - conditions_with_any_match

        # Per-condition summary (SNOMED)
        df_condition_summary = (
            df_ranked
            .filter(pl.col("is_best_match"))
            .filter(pl.col("snomed_concept_id").is_not_null())
            .group_by("condition_name")
            .agg([
                pl.col("snomed_concept_id").unique().sort().str.join(", ").alias("snomed_concept_ids"),
                pl.col("snomed_concept_name").unique().sort().str.join(", ").alias("snomed_concept_names"),
                pl.col("concept_id").unique().sort().str.join(", ").alias("source_concept_ids"),
                pl.col("vocabulary_id").unique().sort().str.join(", ").alias("source_vocabularies"),
                pl.col("search_term").unique().sort().str.join(", ").alias("matched_via"),
                pl.col("match_type").first().alias("primary_match_type"),
                pl.col("match_score").min().alias("lowest_score"),
            ])
            .join(df_term_counts, on="condition_name", how="left")
            .sort("lowest_score")
        )

        # Unmatched conditions
        df_unmatched_conditions = (
            df_input
            .filter(pl.col("condition_name").is_in(conditions_no_match))
            .select("condition_name")
            .unique()
            .join(df_term_counts, on="condition_name", how="left")
        )

        # Print summary
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
            Both the key and all values are used as search terms.
        export_tsv : bool
            If True, write TSV files with UTF-8 BOM. Default False.
        export_prefix : str
            Filename prefix for exported TSV files. Default "mapping".

        Returns
        -------
        dict with keys:
            df_input                : normalised search terms
            df_exact                : exact match results
            df_fuzzy                : fuzzy match results
            df_ranked               : combined ranked results with SNOMED mapping
            df_condition_summary    : per-condition SNOMED summary
            df_unmatched_conditions : conditions with 0 matches
            df_term_counts          : per-condition term match counts
        """
        print("[1/5] Building search terms...")
        df_input = self._build_input(conditions)
        print(f"  Conditions: {df_input['condition_name'].n_unique()}, "
              f"Search terms: {df_input['search_term'].n_unique()}")

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
            self.write_tsv_bom(df_ranked, f"{export_prefix}_full_review.tsv")
            self.write_tsv_bom(summary["df_condition_summary"], f"{export_prefix}_condition_summary.tsv")
            if len(summary["df_unmatched_conditions"]) > 0:
                self.write_tsv_bom(summary["df_unmatched_conditions"], f"{export_prefix}_unmatched_conditions.tsv")
            print("  Done.")

        return results