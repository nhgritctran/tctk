"""
Module: _omop_utils

Base class for OMOP condition mappers.

Provides the general condition-mapping workflow shared by vocabulary-specific
subclasses (e.g. Condition2SNOMED, Condition2ICD):

    - Input normalization (lowercase, strip tags, expand parentheticals)
    - Synonym loading (single CTE query against local DuckDB)
    - Exact + fuzzy matching via Polars joins and rapidfuzz
    - AI review infrastructure (credential management, batch sizing)
"""

import json
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

from tctk._utils import (
    sql_escape,
    load_api_key,
    check_api_key,
    call_gemini,
    setup_credentials,
    detect_best_model,
)
from tctk.omop import get_vocab_db

__all__ = ["ConditionMapperBase"]


class ConditionMapperBase:
    """Base class for OMOP condition mappers.

    Provides input normalization, exact/fuzzy matching against a local
    DuckDB vocabulary database, ranking, and AI review infrastructure.

    Parameters
    ----------
    vocab_db : str, optional
        Path to the DuckDB vocabulary database.
        Default: auto-downloaded from Hugging Face
    force_download_db : bool
        Force re-download of the vocabulary database. Default False.
    """

    BATCH_SIZE = 500

    # Subclasses must define these for vocabulary-specific AI review
    _TARGET_ID_COL: str    # e.g. "snomed_concept_id" or "icd_concept_id"
    _TARGET_NAME_COL: str  # e.g. "snomed_concept_name" or "icd_concept_name"
    _VOCAB_LABEL: str      # e.g. "SNOMED CT" or "ICD-10-CM"

    # Generic medical nouns excluded from fuzzy scoring to prevent
    # false matches driven by shared non-discriminative tokens.
    _FUZZY_STOPWORDS = {
        "disease", "disorder", "syndrome", "condition", "infection",
        "of", "the", "and", "in", "with", "by", "to", "a", "an",
    }

    @staticmethod
    def _strip_stopwords(text: str, stopwords: set[str]) -> str:
        """Remove stopwords from text for fuzzy scoring."""
        tokens = [t for t in text.split() if t not in stopwords]
        return " ".join(tokens) if tokens else text

    # Batch sizing: target match lines per batch (~150 ≈ 3 min response)
    _TARGET_LINES_PER_BATCH = 150
    _MAX_OUTPUT_TOKENS = 65_536  # kept for safety reference

    def __init__(
        self,
        vocab_db: Optional[str] = None,
        force_download_db: bool = False,
    ):
        self._vocab_db = Path(vocab_db) if vocab_db else Path(get_vocab_db(force_download=force_download_db))
        self._api_key: Optional[str] = None
        self._ai_tier: Optional[str] = None
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

    def _resolve_model(
        self,
        gemini_api_key: Optional[str] = None,
        ai_tier: Optional[str] = None,
        ai_min_version: float = 3.0,
        config_path: Optional[str] = None,
    ) -> str:
        """Detect and cache the best Gemini model for the API key and tier.

        Parameters
        ----------
        gemini_api_key : str, optional
            Explicit API key (highest priority).
        ai_tier : str, optional
            Model tier: "pro", "flash", or "flash-lite". Default "flash".
        ai_min_version : float
            Minimum model version. Default 3.0 (prefer Gemini 3.x+).
            Set to 2.5 to allow older models.
        config_path : str, optional
            Path to JSON config file for API key.
        """
        api_key = (
            gemini_api_key
            or self._api_key
            or load_api_key(config_path=config_path)
        )
        api_key = check_api_key(api_key)
        tier = ai_tier or self._ai_tier or "flash"

        # Reuse cached model if key, tier, and min_version haven't changed
        if (
            self._ai_model is not None
            and self._api_key == api_key
            and self._ai_tier == tier
            and getattr(self, "_ai_min_version", None) == ai_min_version
        ):
            return self._ai_model

        self._api_key = api_key
        self._ai_tier = tier
        self._ai_min_version = ai_min_version
        self._ai_model = detect_best_model(
            api_key, ai_tier=tier, min_version=ai_min_version
        )
        print(f"  Gemini model selected: {self._ai_model}")
        return self._ai_model

    # -------------------------------------------------------------------
    # Step 1: Build normalized search terms from input dict
    # -------------------------------------------------------------------

    @staticmethod
    def _build_input(conditions: dict[str, list[str]]) -> pl.DataFrame:
        """
        Convert {condition_name: [condition_synonyms]} dict into a normalized
        DataFrame with columns: condition_name, search_term.

        - The condition_name itself is always included as a search term.
        - Tags like (subtype) and (synonym) are stripped.
        - Terms containing parentheses are expanded into two variants:
          one with and one without the parenthetical.
        - Terms containing "/" are split into separate search terms
          (e.g. "myositis/polymyositis" -> "myositis", "polymyositis").
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
                .str.replace_all("-", " ")
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

        df_combined = pl.concat(
            [df_no_parens, df_has_parens, df_parens_stripped],
            how="diagonal_relaxed",
        )

        # Expand terms containing "/" into separate search terms
        # e.g. "myositis/polymyositis" -> "myositis", "polymyositis"
        df_has_slash = df_combined.filter(pl.col("search_term").str.contains("/"))
        slash_rows = []
        for row in df_has_slash.iter_rows(named=True):
            for part in row["search_term"].split("/"):
                part = part.strip()
                if part:
                    slash_rows.append({
                        "condition_name": row["condition_name"],
                        "search_term_raw": row.get("search_term_raw", ""),
                        "search_term": part,
                    })
        df_slash_expanded = pl.DataFrame(slash_rows) if slash_rows else pl.DataFrame(schema=df_combined.schema)

        df_input = (
            pl.concat(
                [df_combined, df_slash_expanded],
                how="diagonal_relaxed",
            )
            .unique(subset=["condition_name", "search_term"])
            .select("condition_name", "search_term")
        )

        return df_input

    # -------------------------------------------------------------------
    # Step 2: Load synonyms and match (exact + fuzzy)
    # -------------------------------------------------------------------

    def _load_synonyms(self, vocab: str = "SNOMED") -> pl.DataFrame:
        """Load candidate concept synonyms from the vocabulary database.

        Runs a single CTE query that collects concept names and synonyms,
        then generates parenthetical-stripped variants.

        Parameters
        ----------
        vocab : str
            Vocabulary strategy: ``"SNOMED"`` (default) or ``"ICD"``.

        Returns
        -------
        pl.DataFrame
            Columns: concept_id, concept_code, vocabulary_id, concept_name,
            concept_class_id, standard_concept, synonym_lower
        """
        if vocab == "SNOMED":
            vocab_filter = "vocabulary_id IN ('SNOMED')"
            standard_filter = "AND standard_concept = 'S'"
        elif vocab == "ICD":
            vocab_filter = "vocabulary_id IN ('ICD9CM', 'ICD10CM')"
            standard_filter = ""
        else:
            raise NotImplementedError(
                f"_load_synonyms not implemented for vocab={vocab!r}"
            )

        sql = f"""
        WITH target_concepts AS (
            SELECT concept_id, concept_code, vocabulary_id,
                   concept_name, concept_class_id, standard_concept
            FROM concept
            WHERE {vocab_filter}
              AND domain_id = 'Condition'
              AND invalid_reason IS NULL
              {standard_filter}
        ),
        all_raw_terms AS (
            SELECT concept_id, concept_code, vocabulary_id,
                   concept_name, concept_class_id, standard_concept,
                   concept_name AS raw_term
            FROM target_concepts
            UNION DISTINCT
            SELECT c.concept_id, c.concept_code, c.vocabulary_id,
                   c.concept_name, c.concept_class_id, c.standard_concept,
                   cs.concept_synonym_name AS raw_term
            FROM target_concepts c
            JOIN concept_synonym cs ON c.concept_id = cs.concept_id
        )
        SELECT DISTINCT
            CAST(concept_id AS VARCHAR) AS concept_id,
            concept_code, vocabulary_id, concept_name,
            concept_class_id, standard_concept,
            TRIM(REGEXP_REPLACE(REGEXP_REPLACE(LOWER(TRIM(processed_term)), '-', ' ', 'g'), '\s+', ' ', 'g')) AS synonym_lower
        FROM all_raw_terms,
            UNNEST([raw_term,
                    REGEXP_REPLACE(raw_term, '\\s*\\([^)]*\\)', '', 'g')
            ]) AS t(processed_term)
        WHERE processed_term IS NOT NULL
          AND TRIM(processed_term) != ''
        """

        df = self._query(sql)
        print(
            f"  Loaded {len(df)} candidate synonyms "
            f"({df['concept_id'].n_unique()} concepts)"
        )
        return df

    def _match(
        self,
        df_input: pl.DataFrame,
        vocab: str = "SNOMED",
        fuzzy_threshold: int = 85,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Match search terms against vocabulary synonyms (exact + fuzzy).

        1. Loads all candidate synonyms via :meth:`_load_synonyms`
        2. Exact match: Polars inner join on search_term == synonym_lower
        3. Fuzzy match: keyword pre-filter + rapidfuzz for unmatched terms

        Parameters
        ----------
        df_input : pl.DataFrame
            Input with ``condition_name`` and ``search_term`` columns.
        vocab : str
            Vocabulary strategy: ``"SNOMED"`` or ``"ICD"``.
        fuzzy_threshold : int
            Minimum score (0-100) for rapidfuzz token_sort_ratio.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            ``(df_exact, df_fuzzy)`` with columns: condition_name,
            search_term, matched_concept_synonym, concept_id, concept_code,
            concept_name, vocabulary_id, concept_class_id, standard_concept,
            match_type, match_score
        """
        # 1. Load synonyms (single SQL query)
        print("  Loading synonyms from vocab DB...")
        df_candidates = self._load_synonyms(vocab)

        # 2. Exact match via Polars inner join
        df_exact = (
            df_input.join(
                df_candidates,
                left_on="search_term",
                right_on="synonym_lower",
                how="inner",
            )
            .unique(subset=["condition_name", "search_term", "concept_id"])
            .with_columns(
                pl.col("search_term").alias("matched_concept_synonym"),
                pl.lit("exact").alias("match_type"),
                pl.lit(100).alias("match_score"),
            )
        )

        n_exact = df_exact["search_term"].n_unique()
        n_total = df_input["search_term"].n_unique()
        print(f"  Exact match: {n_exact}/{n_total} unique search terms")

        # 3. Identify unmatched terms
        matched_terms = set(df_exact["search_term"].unique().to_list())
        df_unmatched = df_input.filter(~pl.col("search_term").is_in(matched_terms))

        if len(df_unmatched) == 0:
            print("  All terms matched exactly. Skipping fuzzy step.")
            return df_exact, pl.DataFrame(schema=df_exact.schema)

        print(f"  Unmatched terms for fuzzy matching: {len(df_unmatched)}")

        # 4. Keyword pre-filter: extract 4+ char keywords, filter via Polars
        keywords = set()
        for term in df_unmatched["search_term"].to_list():
            keywords.update(w for w in term.split() if len(w) >= 4)

        if not keywords:
            print("  No keywords for fuzzy filtering. Skipping fuzzy step.")
            return df_exact, pl.DataFrame(schema=df_exact.schema)

        keyword_list = sorted(keywords)
        filter_expr = pl.lit(False)
        for kw in keyword_list:
            filter_expr = filter_expr | pl.col("synonym_lower").str.contains(
                kw, literal=True
            )

        df_fuzzy_candidates = df_candidates.filter(filter_expr)
        print(f"  Fuzzy candidates after keyword filter: {len(df_fuzzy_candidates)}")

        if len(df_fuzzy_candidates) == 0:
            print("  No candidates found. Skipping fuzzy step.")
            return df_exact, pl.DataFrame(schema=df_exact.schema)

        # 5. Fuzzy match via rapidfuzz
        candidate_synonyms = df_fuzzy_candidates["synonym_lower"].to_list()
        candidate_stripped = [
            self._strip_stopwords(s, self._FUZZY_STOPWORDS)
            for s in candidate_synonyms
        ]
        fuzzy_results = []

        for row in tqdm(
            df_unmatched.iter_rows(named=True),
            total=len(df_unmatched),
            desc="Fuzzy matching terms",
        ):
            term = row["search_term"]
            term_stripped = self._strip_stopwords(term, self._FUZZY_STOPWORDS)
            matches = process.extract(
                term_stripped,
                candidate_stripped,
                scorer=fuzz.token_sort_ratio,
                limit=5,
                score_cutoff=fuzzy_threshold,
            )
            for _, score, idx in matches:
                cand = df_fuzzy_candidates.row(idx, named=True)
                fuzzy_results.append(
                    {
                        "condition_name": row["condition_name"],
                        "search_term": term,
                        "matched_concept_synonym": candidate_synonyms[idx],
                        "concept_id": str(cand["concept_id"]),
                        "concept_code": cand["concept_code"],
                        "concept_name": cand["concept_name"],
                        "vocabulary_id": cand["vocabulary_id"],
                        "concept_class_id": cand["concept_class_id"],
                        "standard_concept": cand["standard_concept"],
                        "match_type": "fuzzy",
                        "match_score": int(score),
                    }
                )

        if fuzzy_results:
            df_fuzzy = pl.DataFrame(fuzzy_results)
            print(
                f"  Fuzzy matches: {len(df_fuzzy)} synonyms "
                f"for {df_fuzzy['search_term'].n_unique()} terms"
            )
        else:
            print("  No fuzzy matches found.")
            df_fuzzy = pl.DataFrame(schema=df_exact.schema)

        return df_exact, df_fuzzy

    # -------------------------------------------------------------------
    # AI review batch sizing
    # -------------------------------------------------------------------

    def _build_review_batches(
        self,
        df_to_review: pl.DataFrame,
        conditions_to_review: list[str],
    ) -> list[list[str]]:
        """Pack conditions into batches by match line count.

        Greedily adds conditions to the current batch until
        ``_TARGET_LINES_PER_BATCH`` would be exceeded, then starts a new
        batch.  A single condition always gets its own batch even if it
        exceeds the target (no splitting within a condition).
        """
        counts = dict(
            df_to_review.group_by("condition_name")
            .agg(pl.len().alias("n"))
            .iter_rows()
        )
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_lines = 0
        for cond in conditions_to_review:
            cond_lines = counts.get(cond, 0)
            if current_batch and current_lines + cond_lines > self._TARGET_LINES_PER_BATCH:
                batches.append(current_batch)
                current_batch, current_lines = [], 0
            current_batch.append(cond)
            current_lines += cond_lines
        if current_batch:
            batches.append(current_batch)
        return batches

    # -------------------------------------------------------------------
    # AI review hooks (override in subclass for vocabulary-specific text)
    # -------------------------------------------------------------------

    def _ai_review_system_prompt(self) -> str:
        """Return the instruction preamble for AI review prompts."""
        return (
            f"You are a clinical terminologist with expertise in OMOP and {self._VOCAB_LABEL} vocabularies.\n"
            f"You are validating fuzzy matches between search terms and {self._VOCAB_LABEL} concepts.\n"
            "Each entry shows: term (search term), fuzzy_synonym (the vocabulary "
            f"synonym text that fuzzy-matched), and target (the {self._VOCAB_LABEL} concept it maps to).\n"
            "Accept if the concept captures the same clinical meaning "
            "as the search term in the context of the condition.\n"
            "Reject if it refers to a different condition, wrong body site, "
            "wrong specificity, or unrelated finding.\n"
            "Rate confidence 0-100 (100 = certain).\n"
            "Reason must be 10 words or fewer.\n"
            "Comment: provide a short sentence or phrase explaining your decision "
            "(e.g. clinical rationale, what differs, why it matches).\n"
        )

    def _ai_review_format_match_line(self, row: dict) -> str:
        """Format one match row into a pipe-delimited prompt line."""
        parts = [
            f"term={row.get('search_term', '')}",
            f"fuzzy_synonym={row.get('matched_concept_synonym', '')}",
            f"target={row.get(self._TARGET_NAME_COL, '')}",
            f"id={row[self._TARGET_ID_COL]}",
            f"fuzzy={row['match_score']}",
        ]
        if row.get("ancestor_distance") is not None:
            parts.append(f"ancestor_dist={row['ancestor_distance']}")
        return " | ".join(parts)

    def _ai_review_response_schema(self) -> dict:
        """Return the structured response schema for Gemini AI review."""
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
                            "r": {"type": "STRING"},
                            "c": {"type": "INTEGER"},
                            "comment": {"type": "STRING"},
                        },
                        "required": ["condition", "t", "id", "v", "r", "c", "comment"],
                    },
                }
            },
            "required": ["verdicts"],
        }

    def _ai_review_parse_verdict(self, v: dict) -> dict:
        """Parse one AI verdict JSON object into a DataFrame row dict."""
        return {
            "condition_name": v.get("condition", ""),
            "search_term": v.get("t", ""),
            self._TARGET_ID_COL: str(v.get("id", "")),
            "ai_verdict": v.get("v", "human review"),
            "ai_reason": v.get("r", ""),
            "ai_confidence": v.get("c"),
            "ai_comment": v.get("comment", ""),
        }

    # -------------------------------------------------------------------
    # AI review via Gemini
    # -------------------------------------------------------------------

    def ai_review(
        self,
        results: dict,
        batch_size: Optional[int] = None,
        confidence_threshold: int = 80,
        gemini_api_key: Optional[str] = None,
        ai_tier: str = "flash",
        ai_min_version: float = 3.0,
        config_path: Optional[str] = None,
    ) -> dict:
        """AI-assisted review of fuzzy matches using Gemini API.

        Parameters
        ----------
        results : dict
            Output from map() pipeline.
        batch_size : int, optional
            Conditions per API call. If None, auto-calculated from model limits.
        confidence_threshold : int
            Confidence score (0-100) below which verdicts are flagged as
            "human review". Default 80.
        gemini_api_key : str, optional
            Gemini API key. Falls back to key set via :meth:`set_api_key`,
            then env var GEMINI_API_KEY, then config file.
        ai_tier : str
            Preferred Gemini model tier: "pro", "flash", or "flash-lite".
            Default "flash".
        ai_min_version : float
            Minimum Gemini model version. Default 3.0 (prefer Gemini 3.x+).
            Set to 2.5 to allow older models (e.g. gemini-2.5-flash).
        config_path : str, optional
            Path to JSON config file for API key.

        Returns
        -------
        dict
            Updated results with ai_verdict, ai_reason, ai_confidence,
            and ai_comment columns.
        """
        model = self._resolve_model(
            gemini_api_key=gemini_api_key,
            ai_tier=ai_tier,
            ai_min_version=ai_min_version,
            config_path=config_path,
        )
        df_matches = results["df_matches"].clone()

        # All fuzzy matches with a target concept
        df_fuzzy_all = df_matches.filter(
            (pl.col("match_type") == "fuzzy")
            & pl.col(self._TARGET_ID_COL).is_not_null()
        )

        df_to_review = df_fuzzy_all
        print(f"  Fuzzy matches: {len(df_to_review)} for AI review")

        if len(df_to_review) == 0:
            print("  No matches require AI review.")
            df_matches = df_matches.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("ai_verdict"),
                pl.lit(None).cast(pl.Utf8).alias("ai_reason"),
                pl.lit(None).cast(pl.Int64).alias("ai_confidence"),
                pl.lit(None).cast(pl.Utf8).alias("ai_comment"),
            )
            results["df_matches"] = df_matches
            return results

        conditions_to_review = df_to_review["condition_name"].unique().to_list()
        n_conditions = len(conditions_to_review)

        # Build batches
        print(f"  Reviewing: {len(df_to_review)} fuzzy matches")
        print(f"  Conditions for AI review: {n_conditions}")

        if batch_size is not None:
            # User-specified fixed condition count
            cond_batches = [
                conditions_to_review[i : i + batch_size]
                for i in range(0, len(conditions_to_review), batch_size)
            ]
            print(f"  Batch size: {batch_size} conditions (user-specified)")
            print(f"  Estimated API calls: {len(cond_batches)}")
        else:
            # Auto: line-budget packing
            cond_batches = self._build_review_batches(df_to_review, conditions_to_review)
            avg_lines = len(df_to_review) / max(1, len(cond_batches))
            print(
                f"  Batches: {len(cond_batches)} "
                f"(auto, ~{avg_lines:.0f} lines/batch, "
                f"target \u2264{self._TARGET_LINES_PER_BATCH})"
            )
            print(f"  Estimated API calls: {len(cond_batches)}")

        print(f"  Model: {model}")

        # Schema for structured Gemini response
        response_schema = self._ai_review_response_schema()

        all_verdicts = []
        calls_used = 0

        # Verify fuzzy scores match stored values
        df_fuzzy_rows = df_to_review.filter(pl.col("match_type") == "fuzzy")
        if len(df_fuzzy_rows) > 0:
            mismatches = []
            for row in df_fuzzy_rows.iter_rows(named=True):
                synonym = row.get("matched_concept_synonym", "")
                if not synonym:
                    continue
                term_stripped = self._strip_stopwords(row["search_term"], self._FUZZY_STOPWORDS)
                syn_stripped = self._strip_stopwords(synonym, self._FUZZY_STOPWORDS)
                recomputed = int(fuzz.token_sort_ratio(term_stripped, syn_stripped))
                if recomputed != row["match_score"]:
                    mismatches.append(
                        f"    '{row['search_term']}' <-> '{synonym}': "
                        f"stored={row['match_score']}, recomputed={recomputed}"
                    )
            if mismatches:
                print(f"  WARNING: {len(mismatches)} fuzzy score mismatches:")
                for m in mismatches[:10]:
                    print(m)
            else:
                print(f"  Fuzzy score check: all {len(df_fuzzy_rows)} scores verified")

        # Build and print 1 example prompt (first condition only)
        system_prompt = self._ai_review_system_prompt()
        _ex_parts = [system_prompt]
        _ex_cond = conditions_to_review[0]
        _ex_rows = df_to_review.filter(pl.col("condition_name") == _ex_cond)
        _ex_parts.append(f"Condition: {_ex_cond}")
        for row in _ex_rows.iter_rows(named=True):
            _ex_parts.append(self._ai_review_format_match_line(row))

        print(f"\n  Example prompt (condition 1 of {n_conditions}):")
        for line in "\n".join(_ex_parts).splitlines():
            print(f"    {line}")

        for batch in tqdm(cond_batches, desc="AI review batches"):
            prompt_parts = [system_prompt + "\n"]

            for cond in batch:
                cond_rows = df_to_review.filter(pl.col("condition_name") == cond)
                prompt_parts.append(f"Condition: {cond}")

                for row in cond_rows.iter_rows(named=True):
                    prompt_parts.append(self._ai_review_format_match_line(row))

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
                self._ai_review_parse_verdict(v)
                for v in all_verdicts
            ]
            df_verdicts = pl.DataFrame(verdict_rows).unique(
                subset=["condition_name", "search_term", self._TARGET_ID_COL]
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

            df_matches = df_matches.join(
                df_verdicts,
                on=["condition_name", "search_term", self._TARGET_ID_COL],
                how="left",
            )
        else:
            df_matches = df_matches.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("ai_verdict"),
                pl.lit(None).cast(pl.Utf8).alias("ai_reason"),
                pl.lit(None).cast(pl.Int64).alias("ai_confidence"),
                pl.lit(None).cast(pl.Utf8).alias("ai_comment"),
            )

        # Count final verdicts
        df_reviewed = df_matches.filter(pl.col("ai_verdict").is_not_null())

        n_accepted = len(df_reviewed.filter(pl.col("ai_verdict") == "accept"))
        n_rejected = len(df_reviewed.filter(pl.col("ai_verdict") == "reject"))
        n_human_review = len(df_reviewed.filter(pl.col("ai_verdict") == "human review"))
        n_total = n_accepted + n_rejected + n_human_review

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
        print(f"  Total reviewed:    {n_total} matches")
        print(f"    accepted:        {n_accepted} matches ({n_cond_accepted} conditions with at least 1 accepted)")
        print(f"    rejected:        {n_rejected} matches ({n_cond_rejected} conditions lost all hits)")
        print(f"    human review:    {n_human_review} matches ({n_cond_human_review} conditions)")
        print(f"{'=' * 40}")

        results["df_matches"] = df_matches
        return results
