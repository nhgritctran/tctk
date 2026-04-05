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
import re
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

from tctk._utils import (
    sql_escape,
    strip_accents,
    load_api_key,
    load_anthropic_api_key,
    check_api_key,
    call_gemini,
    call_claude,
    setup_credentials,
    detect_best_model,
    create_gemini_cache,
    call_gemini_cached,
    delete_gemini_cache,
)
from tctk.omop.vocab import get_vocab_db

__all__ = ["ConditionMapperBase"]


class ConditionMapperBase:
    """Base class for OMOP condition mappers.

    Provides input normalization, exact/fuzzy matching against a local
    DuckDB vocabulary database, ranking, and AI review infrastructure.

    Args:
        vocab_db (str, optional): Path to the DuckDB vocabulary database.
            Default: auto-downloaded from Hugging Face
        force_download_db (bool): Force re-download of the vocabulary database. Default False.
    """

    BATCH_SIZE = 500

    # Claude model map (fixed names — no API call needed)
    _CLAUDE_MODELS = {
        "opus":   "claude-opus-4-6",
        "sonnet": "claude-sonnet-4-6",
        "haiku":  "claude-haiku-4-5-20251001",
    }

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
        self._claude_api_key: Optional[str] = None
        self._claude_model: Optional[str] = None

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

    def set_api_key(
        self,
        key: Optional[str] = None,
        key_file: Optional[str] = None,
        claude_key: Optional[str] = None,
        claude_key_file: Optional[str] = None,
    ) -> None:
        """Set API keys for AI review.

        Args:
            key (str, optional): Gemini API key string directly.
            key_file (str, optional): Path to a JSON file containing
                ``{"gemini_api_key": "..."}``. May also contain
                ``{"anthropic_api_key": "..."}`` for Claude.
            claude_key (str, optional): Claude (Anthropic) API key string directly.
            claude_key_file (str, optional): Path to a JSON file containing
                ``{"anthropic_api_key": "..."}``.
        """
        # --- Gemini key ---
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
        elif not claude_key and not claude_key_file:
            raise ValueError(
                "Provide at least one API key:\n"
                "  Gemini: key=... or key_file=...\n"
                "  Claude: claude_key=... or claude_key_file=..."
            )
        self._ai_model = None

        # --- Claude key (optional) ---
        if claude_key:
            self._claude_api_key = claude_key
        elif claude_key_file:
            claude_config = json.loads(Path(claude_key_file).read_text())
            self._claude_api_key = claude_config.get("anthropic_api_key")
        elif key_file:
            # Also check the Gemini key file for anthropic_api_key
            gemini_config = json.loads(Path(key_file).read_text())
            claude_key_found = gemini_config.get("anthropic_api_key")
            if claude_key_found:
                self._claude_api_key = claude_key_found
        self._claude_model = None

    @staticmethod
    def setup_credentials(path: Optional[str] = None) -> None:
        """Interactive helper to create a credentials file.

        Creates ~/.config/tctk/credentials.json (or custom path) with
        the Gemini API key. Input is hidden via getpass.

        Get a free key at: https://aistudio.google.com/apikey
        """
        setup_credentials(path)

    def _resolve_claude_model(
        self,
        ai_tier: Optional[str] = None,
        ai_min_version: Optional[float] = None,
    ) -> Optional[str]:
        """Select a Claude model by tier and minimum version.

        Returns None if no Claude API key is set.
        """
        if not self._claude_api_key:
            return None

        tier = (ai_tier or "sonnet").lower().strip()
        min_ver = ai_min_version if ai_min_version is not None else 4.6

        # Filter by min_version  (model names use hyphens: claude-opus-4-6)
        candidates = {}
        for t, m in self._CLAUDE_MODELS.items():
            match = re.search(r"(\d+)[.-](\d+)", m)
            ver = float(f"{match.group(1)}.{match.group(2)}") if match else 0.0
            if ver >= min_ver:
                candidates[t] = (m, ver)

        if not candidates:
            available = ", ".join(
                f"{t} ({m})" for t, m in self._CLAUDE_MODELS.items()
            )
            raise RuntimeError(
                f"No Claude models >= {min_ver}. Available: {available}"
            )

        if tier in candidates:
            model = candidates[tier][0]
        else:
            # Pick best available tier
            model = max(candidates.values(), key=lambda x: x[1])[0]
            print(
                f"  Warning: Claude tier '{tier}' not available >= {min_ver}. "
                f"Using {model}."
            )

        if self._claude_model != model:
            self._claude_model = model
            print(f"  Claude model selected: {model}")
        return model

    def _has_provider(self, provider: str) -> bool:
        """Check if an API key is configured for the given provider."""
        if provider == "claude":
            return self._claude_api_key is not None
        elif provider == "gemini":
            return self._api_key is not None
        return False

    def _resolve_model(
        self,
        ai_provider: str = "gemini",
        gemini_api_key: Optional[str] = None,
        ai_tier: Optional[str] = None,
        ai_min_version: Optional[float] = None,
        config_path: Optional[str] = None,
    ) -> tuple:
        """Detect and cache the best model for the given provider.

        Parameters
        ----------
        ai_provider : str
            ``"gemini"`` or ``"claude"``. Default ``"gemini"``.
        gemini_api_key : str, optional
            Explicit Gemini API key (highest priority).
        ai_tier : str, optional
            Model tier. For Gemini: "pro", "flash", "flash-lite" (default "pro").
            For Claude: "opus", "sonnet", "haiku" (default "sonnet").
        ai_min_version : float, optional
            Minimum model version. Gemini default 3.0, Claude default 4.6.
        config_path : str, optional
            Path to JSON config file for API key.

        Returns
        -------
        tuple[str, str]
            ``(provider, model)`` — e.g. ``("gemini", "gemini-3.0-pro")`` or
            ``("claude", "claude-sonnet-4-6")``.
        """
        if ai_provider == "claude":
            tier = ai_tier or "sonnet"
            min_ver = ai_min_version if ai_min_version is not None else 4.6
            model = self._resolve_claude_model(
                ai_tier=tier, ai_min_version=min_ver,
            )
            if model is None:
                raise ValueError(
                    "Claude API key not set.\n"
                    "Use set_api_key(claude_key=...) or "
                    "set_api_key(claude_key_file=...)"
                )
            return ("claude", model)

        # --- Gemini ---
        tier = ai_tier or self._ai_tier or "pro"
        min_ver = ai_min_version if ai_min_version is not None else 3.0

        api_key = (
            gemini_api_key
            or self._api_key
            or load_api_key(config_path=config_path)
        )
        api_key = check_api_key(api_key)

        # Reuse cached model if key, tier, and min_version haven't changed
        if (
            self._ai_model is not None
            and self._api_key == api_key
            and self._ai_tier == tier
            and getattr(self, "_ai_min_version", None) == min_ver
        ):
            return ("gemini", self._ai_model)

        self._api_key = api_key
        self._ai_tier = tier
        self._ai_min_version = min_ver
        self._ai_model = detect_best_model(
            api_key, ai_tier=tier, min_version=min_ver,
        )
        print(f"  Gemini model selected: {self._ai_model}")
        return ("gemini", self._ai_model)

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
                # Smart quotes → straight
                .str.replace_all("\u2018", "'")
                .str.replace_all("\u2019", "'")
                .str.replace_all("\u201c", '"')
                .str.replace_all("\u201d", '"')
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
    # Step 2: OMOP vocabulary lookup (exact + fuzzy)
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
                    REGEXP_REPLACE(raw_term, '\\s*\\([^)]*\\)', '', 'g'),
                    REGEXP_REPLACE(raw_term, '\\s*\\[[^\\]]*\\]', '', 'g'),
                    REGEXP_REPLACE(REGEXP_REPLACE(raw_term, '\\s*\\([^)]*\\)', '', 'g'), '\\s*\\[[^\\]]*\\]', '', 'g')
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

    def _omop_lookup(
        self,
        df_input: pl.DataFrame,
        vocab: str = "SNOMED",
        fuzzy_threshold: int = 85,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Match search terms against OMOP vocabulary synonyms (exact + fuzzy).

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
            .with_columns(
                pl.col("search_term").alias("matched_concept_synonym"),
                pl.lit("exact").alias("match_type"),
                pl.lit(100).alias("match_score"),
            )
            .unique(subset=["condition_name", "search_term",
                            "matched_concept_synonym", "concept_code",
                            "vocabulary_id"])
        )

        # 3. Identify unmatched terms
        matched_terms = set(df_exact["search_term"].unique().to_list())
        df_unmatched = df_input.filter(~pl.col("search_term").is_in(matched_terms))
        n_input = len(df_input)
        n_matched = n_input - len(df_unmatched)
        print(f"  Exact match: {n_matched}/{n_input} search terms")

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
                        "matched_concept_synonym": candidate_stripped[idx],
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
            n_fuzzy_pairs = (
                df_fuzzy.select("condition_name", "search_term")
                .unique().height
            )
            print(
                f"  Fuzzy matches: {len(df_fuzzy)} concept matches "
                f"for {n_fuzzy_pairs} search terms"
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
            "Comment: 2-5 word clinical rationale "
            "(e.g. 'exact anatomical match', 'wrong body site', 'different etiology').\n"
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
                            "comment": {"type": "STRING"},
                        },
                        "required": ["condition", "t", "id", "v", "comment"],
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
            "ai_comment": v.get("comment", ""),
        }

    # -------------------------------------------------------------------
    # AI dispatch helper
    # -------------------------------------------------------------------

    @staticmethod
    def _parse_ai_json(response_text: str) -> dict:
        """Parse a JSON response from an AI model.

        Handles empty responses and Claude's tendency to wrap JSON in
        markdown code fences (```json ... ```) despite instructions.
        """
        if not response_text or not response_text.strip():
            raise RuntimeError("API returned empty response")

        text = response_text.strip()

        # Strip markdown code fences if present (Claude sometimes adds these)
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            # Find the outermost JSON object if surrounded by extra text
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            if obj_match and not text.startswith("{"):
                text = obj_match.group(0)

        return json.loads(text)

    def _call_ai(
        self,
        provider: str,
        prompt: str,
        model: str,
        system_prompt: str = "",
        response_schema: Optional[dict] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 65536,
        timeout: int = 300,
        max_retries: int = 3,
        cache_name: Optional[str] = None,
    ) -> str:
        """Dispatch an AI call to Gemini or Claude.

        For Claude, the response schema is included in the system prompt
        as JSON instructions (Claude doesn't support native schema enforcement).
        """
        if provider == "claude":
            claude_system = system_prompt or ""
            if response_schema:
                claude_system += (
                    "\n\nYou MUST respond with ONLY valid JSON (no markdown "
                    "fences, no explanation) matching this schema:\n"
                    + json.dumps(response_schema, indent=2)
                )
            return call_claude(
                prompt=prompt,
                api_key=self._claude_api_key,
                model=model,
                system_prompt=claude_system,
                temperature=temperature,
                max_output_tokens=min(max_output_tokens, 16384),
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            # Gemini
            if cache_name:
                return call_gemini_cached(
                    prompt, check_api_key(self._api_key), model,
                    cache_name=cache_name,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    max_retries=max_retries,
                    response_schema=response_schema,
                )
            else:
                full_prompt = (
                    (system_prompt + "\n" + prompt)
                    if system_prompt else prompt
                )
                return call_gemini(
                    full_prompt, check_api_key(self._api_key), model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    max_retries=max_retries,
                    response_schema=response_schema,
                )

    # -------------------------------------------------------------------
    # AI review: single-pass execution
    # -------------------------------------------------------------------

    def _run_single_pass(
        self,
        df_to_review: pl.DataFrame,
        cond_batches: list[list[str]],
        system_prompt: str,
        response_schema: dict,
        model: str,
        pass_num: int,
        cache_name: Optional[str] = None,
        provider: str = "gemini",
    ) -> list[dict]:
        """Run a single AI review pass over all batches.

        Parameters
        ----------
        df_to_review : pl.DataFrame
            Fuzzy matches to review.
        cond_batches : list[list[str]]
            Condition name batches.
        system_prompt : str
            System prompt text (included in prompt if no cache).
        response_schema : dict
            Gemini structured output schema.
        model : str
            Gemini model name.
        pass_num : int
            Pass number (for display).
        cache_name : str, optional
            Gemini cache name. If provided, uses cached API call.

        Returns
        -------
        list[dict]
            Parsed verdict dicts with keys: condition_name, search_term,
            <target_id_col>, ai_verdict, ai_comment.
        """
        all_verdicts = []

        # Build lookup: accent-stripped search_term -> original search_term
        # so we can fix up Gemini responses that strip accents
        original_terms = set(df_to_review["search_term"].unique().to_list())
        norm_to_original = {}
        for t in original_terms:
            norm_to_original[strip_accents(t)] = t

        for batch in tqdm(cond_batches, desc=f"AI pass {pass_num}"):
            # Build data prompt (conditions + match lines)
            data_parts = []
            for cond in batch:
                cond_rows = df_to_review.filter(pl.col("condition_name") == cond)
                data_parts.append(f"Condition: {cond}")
                for row in cond_rows.iter_rows(named=True):
                    data_parts.append(self._ai_review_format_match_line(row))
                data_parts.append("")

            data_prompt = "\n".join(data_parts)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response_text = self._call_ai(
                        provider, data_prompt, model,
                        system_prompt=system_prompt,
                        response_schema=response_schema,
                        cache_name=cache_name if provider == "gemini" else None,
                    )

                    parsed = self._parse_ai_json(response_text)
                    verdicts = parsed.get("verdicts", [])
                    for v in verdicts:
                        row = self._ai_review_parse_verdict(v)
                        # Fix search_term if AI stripped accents
                        st = row["search_term"]
                        if st not in original_terms:
                            row["search_term"] = norm_to_original.get(strip_accents(st), st)
                        all_verdicts.append(row)
                    break  # success
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time
                        wait = 30 * (attempt + 1)
                        print(f"  Pass {pass_num} batch error (attempt "
                              f"{attempt + 1}/{max_retries}): {e}")
                        print(f"    Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        # Primary exhausted — try fallback provider
                        fallback = "claude" if provider == "gemini" else "gemini"
                        if self._has_provider(fallback):
                            print(f"  {provider} failed, trying {fallback}...")
                            try:
                                _, fb_model = self._resolve_model(
                                    ai_provider=fallback,
                                )
                                response_text = self._call_ai(
                                    fallback, data_prompt, fb_model,
                                    system_prompt=system_prompt,
                                    response_schema=response_schema,
                                )
                                parsed = self._parse_ai_json(response_text)
                                verdicts = parsed.get("verdicts", [])
                                for v in verdicts:
                                    row = self._ai_review_parse_verdict(v)
                                    st = row["search_term"]
                                    if st not in original_terms:
                                        row["search_term"] = norm_to_original.get(
                                            strip_accents(st), st,
                                        )
                                    all_verdicts.append(row)
                            except Exception as fb_e:
                                print(f"  {fallback} also failed: {fb_e}")
                                print(f"  Pass {pass_num} batch FAILED after "
                                      f"{max_retries} attempts + fallback")
                        else:
                            print(f"  Pass {pass_num} batch FAILED after "
                                  f"{max_retries} attempts: {e}")

        return all_verdicts

    # -------------------------------------------------------------------
    # AI review: vote confidence computation
    # -------------------------------------------------------------------

    @staticmethod
    def _find_disagreements(
        pass1: list[dict],
        pass2: list[dict],
        target_id_col: str,
    ) -> tuple[dict, dict, set]:
        """Compare two passes and find agreements vs disagreements.

        Returns
        -------
        tuple[dict, dict, set]
            (agreements, disagreements, disagreement_keys)
            agreements: match_key -> {verdict, vote, vote_confidence, comments}
            disagreements: match_key -> {pass_verdicts: [v1, v2], comments: [c1, c2]}
            disagreement_keys: set of (condition_name, search_term, target_id)
        """
        # Index pass results by match key
        def _index_pass(verdicts):
            indexed = {}
            for v in verdicts:
                key = (v["condition_name"], v["search_term"], v[target_id_col])
                indexed[key] = v
            return indexed

        idx1 = _index_pass(pass1)
        idx2 = _index_pass(pass2)
        all_keys = set(idx1.keys()) | set(idx2.keys())

        agreements = {}
        disagreements = {}
        disagreement_keys = set()

        for key in all_keys:
            v1 = idx1.get(key)
            v2 = idx2.get(key)

            if v1 is None or v2 is None:
                # Only appeared in one pass — treat as disagreement
                verdict = (v1 or v2)["ai_verdict"]
                comment = (v1 or v2)["ai_comment"]
                disagreements[key] = {
                    "pass_verdicts": [
                        v1["ai_verdict"] if v1 else None,
                        v2["ai_verdict"] if v2 else None,
                    ],
                    "comments": [
                        v1["ai_comment"] if v1 else None,
                        v2["ai_comment"] if v2 else None,
                    ],
                }
                disagreement_keys.add(key)
                continue

            if v1["ai_verdict"] == v2["ai_verdict"]:
                agreements[key] = {
                    "ai_verdict": v1["ai_verdict"],
                    "ai_vote": "2/2",
                    "ai_vote_confidence": "strong",
                    "comments": [v1["ai_comment"], v2["ai_comment"]],
                }
            else:
                disagreements[key] = {
                    "pass_verdicts": [v1["ai_verdict"], v2["ai_verdict"]],
                    "comments": [v1["ai_comment"], v2["ai_comment"]],
                }
                disagreement_keys.add(key)

        return agreements, disagreements, disagreement_keys

    @staticmethod
    def _compute_vote_confidence(
        all_pass_verdicts: list[list[dict]],
        target_id_col: str,
    ) -> dict:
        """Compute majority vote and confidence from multiple passes.

        Parameters
        ----------
        all_pass_verdicts : list[list[dict]]
            List of verdict lists (one per pass).
        target_id_col : str
            Name of the target ID column.

        Returns
        -------
        dict
            match_key -> {ai_verdict, ai_vote, ai_vote_confidence, comments}
        """
        # Collect all verdicts per match key
        key_verdicts: dict[tuple, list[tuple[str, str]]] = {}
        for pass_verdicts in all_pass_verdicts:
            for v in pass_verdicts:
                key = (v["condition_name"], v["search_term"], v[target_id_col])
                if key not in key_verdicts:
                    key_verdicts[key] = []
                key_verdicts[key].append((v["ai_verdict"], v.get("ai_comment", "")))

        results = {}
        for key, verdict_comment_pairs in key_verdicts.items():
            n_total = len(verdict_comment_pairs)
            n_accept = sum(1 for vc, _ in verdict_comment_pairs if vc == "accept")
            n_reject = n_total - n_accept

            if n_accept >= n_reject:
                majority = "accept"
                majority_count = n_accept
            else:
                majority = "reject"
                majority_count = n_reject

            vote_str = f"{majority_count}/{n_total}"

            # Confidence tiers
            if n_total == 2 and majority_count == 2:
                confidence = "strong"
            elif majority_count >= 4 or (n_total <= 2 and majority_count == n_total):
                confidence = "high"
            elif majority_count == 3 and n_total == 5:
                confidence = "moderate"
            else:
                confidence = "inconclusive"

            # Collect majority-side comments
            majority_comments = [
                c for vc, c in verdict_comment_pairs if vc == majority
            ]

            results[key] = {
                "ai_verdict": majority,
                "ai_vote": vote_str,
                "ai_vote_confidence": confidence,
                "comments": majority_comments,
            }

        return results

    @staticmethod
    def _compute_comment_consistency(vote_results: dict) -> dict:
        """Compute pairwise comment consistency for each match.

        Parameters
        ----------
        vote_results : dict
            Output from ``_compute_vote_confidence()``.

        Returns
        -------
        dict
            match_key -> {ai_comment_consistency, ai_comment_consistency_tier, ai_comment}
        """
        results = {}
        for key, vr in vote_results.items():
            comments = [c for c in vr.get("comments", []) if c]

            if len(comments) == 0:
                results[key] = {
                    "ai_comment_consistency": 0,
                    "ai_comment_consistency_tier": "low",
                    "ai_comment": "",
                }
                continue

            if len(comments) == 1:
                results[key] = {
                    "ai_comment_consistency": 100,
                    "ai_comment_consistency_tier": "high",
                    "ai_comment": comments[0],
                }
                continue

            # Pairwise token_sort_ratio
            pair_scores = []
            for i in range(len(comments)):
                for j in range(i + 1, len(comments)):
                    pair_scores.append(fuzz.token_sort_ratio(comments[i], comments[j]))

            avg_score = int(sum(pair_scores) / len(pair_scores)) if pair_scores else 0

            # Tier assignment
            if avg_score >= 85:
                tier = "high"
            elif avg_score >= 65:
                tier = "moderate"
            else:
                tier = "low"

            # Pick the comment most similar to others (highest avg pairwise score)
            best_comment = comments[0]
            if len(comments) > 2:
                best_avg = -1
                for i, c in enumerate(comments):
                    scores = [
                        fuzz.token_sort_ratio(c, comments[j])
                        for j in range(len(comments)) if j != i
                    ]
                    avg = sum(scores) / len(scores) if scores else 0
                    if avg > best_avg:
                        best_avg = avg
                        best_comment = c
            else:
                best_comment = comments[0]

            results[key] = {
                "ai_comment_consistency": avg_score,
                "ai_comment_consistency_tier": tier,
                "ai_comment": best_comment,
            }

        return results

    @staticmethod
    def _compute_combined_confidence(vote_results: dict, comment_results: dict) -> dict:
        """Combine vote confidence + comment consistency into ai_combined_confidence.

        Mapping table:
            strong  + high     -> strong
            strong  + moderate -> high
            strong  + low      -> inconclusive
            high    + high     -> high
            high    + moderate -> moderate
            high    + low      -> inconclusive
            moderate + high    -> moderate
            moderate + moderate -> low
            moderate + low     -> inconclusive
            inconclusive + any -> inconclusive
            any    + low       -> inconclusive

        Parameters
        ----------
        vote_results : dict
            Output from ``_compute_vote_confidence()``.
        comment_results : dict
            Output from ``_compute_comment_consistency()``.

        Returns
        -------
        dict
            match_key -> {ai_combined_confidence}
        """
        results = {}
        for key in vote_results:
            vc = vote_results[key]["ai_vote_confidence"]
            cc_tier = comment_results.get(key, {}).get("ai_comment_consistency_tier", "low")

            # Low comments or inconclusive votes -> inconclusive
            if cc_tier == "low" or vc == "inconclusive":
                combined = "inconclusive"
            elif vc == "strong":
                combined = "strong" if cc_tier == "high" else "high"
            elif vc == "high":
                combined = "high" if cc_tier == "high" else "moderate"
            elif vc == "moderate":
                combined = "moderate" if cc_tier == "high" else "low"
            else:
                combined = "inconclusive"

            results[key] = {"ai_combined_confidence": combined}

        return results

    # -------------------------------------------------------------------
    # AI review via Gemini (multi-pass with adaptive replication)
    # -------------------------------------------------------------------

    def ai_review(
        self,
        results: dict,
        batch_size: Optional[int] = None,
        gemini_api_key: Optional[str] = None,
        ai_provider: str = "gemini",
        ai_tier: Optional[str] = None,
        ai_min_version: Optional[float] = None,
        config_path: Optional[str] = None,
        ai_passes: int = 2,
    ) -> dict:
        """AI-assisted review of fuzzy matches.

        Uses multi-pass adaptive replication: runs ``ai_passes`` initial
        passes on all fuzzy matches, then up to 5 total passes on
        disagreements. Vote confidence, comment consistency, and overall
        AI reliability are computed from the results.

        Supports Gemini and Claude as AI providers with auto-fallback:
        if all retries for a batch fail with the primary provider,
        the other provider is tried if its key is configured.

        Parameters
        ----------
        results : dict
            Output from map() pipeline.
        batch_size : int, optional
            Conditions per API call. If None, auto-calculated.
        gemini_api_key : str, optional
            Gemini API key.
        ai_provider : str
            Primary AI provider: ``"gemini"`` or ``"claude"``.
            Default ``"gemini"``.
        ai_tier : str, optional
            Preferred model tier. Auto-resolves per provider if None:
            Gemini → "pro", Claude → "sonnet".
        ai_min_version : float, optional
            Minimum model version. Gemini default 3.0, Claude default 4.6.
        config_path : str, optional
            Path to JSON config file for API key.
        ai_passes : int
            Number of initial passes. Default 2. If 1, runs single-pass
            (no vote confidence computed).

        Returns
        -------
        dict
            Updated results with ai_verdict, ai_vote, ai_vote_confidence,
            ai_comment, ai_comment_consistency, ai_comment_consistency_tier,
            and ai_combined_confidence columns.
        """
        provider, model = self._resolve_model(
            ai_provider=ai_provider,
            gemini_api_key=gemini_api_key,
            ai_tier=ai_tier,
            ai_min_version=ai_min_version,
            config_path=config_path,
        )

        # Clear any zombie cached sessions (Gemini only)
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=self._api_key)
            zombie_count = 0
            for cache in client.caches.list():
                client.caches.delete(name=cache.name)
                zombie_count += 1
            if zombie_count:
                print(f"  Cleared {zombie_count} zombie cached session(s)")

        df_matches = results["df_matches"].clone()

        # All fuzzy matches with a target concept
        df_to_review = df_matches.filter(
            (pl.col("match_type") == "fuzzy")
            & pl.col(self._TARGET_ID_COL).is_not_null()
        )
        print(f"  Fuzzy matches: {len(df_to_review)} for AI review")

        # Null columns for when there's nothing to review
        _null_cols = [
            pl.lit(None).cast(pl.Utf8).alias("ai_verdict"),
            pl.lit(None).cast(pl.Utf8).alias("ai_vote"),
            pl.lit(None).cast(pl.Utf8).alias("ai_vote_confidence"),
            pl.lit(None).cast(pl.Utf8).alias("ai_comment"),
            pl.lit(None).cast(pl.Int64).alias("ai_comment_consistency"),
            pl.lit(None).cast(pl.Utf8).alias("ai_comment_consistency_tier"),
            pl.lit(None).cast(pl.Utf8).alias("ai_combined_confidence"),
        ]

        if len(df_to_review) == 0:
            print("  No matches require AI review.")
            df_matches = df_matches.with_columns(*_null_cols)
            results["df_matches"] = df_matches
            return results

        conditions_to_review = df_to_review["condition_name"].unique().to_list()
        n_conditions = len(conditions_to_review)

        # Build batches
        print(f"  Reviewing: {len(df_to_review)} fuzzy matches")
        print(f"  Conditions for AI review: {n_conditions}")
        print(f"  AI passes: {ai_passes}")

        if batch_size is not None:
            cond_batches = [
                conditions_to_review[i : i + batch_size]
                for i in range(0, len(conditions_to_review), batch_size)
            ]
            print(f"  Batch size: {batch_size} conditions (user-specified)")
        else:
            cond_batches = self._build_review_batches(df_to_review, conditions_to_review)
            avg_lines = len(df_to_review) / max(1, len(cond_batches))
            print(
                f"  Batches: {len(cond_batches)} "
                f"(auto, ~{avg_lines:.0f} lines/batch, "
                f"target \u2264{self._TARGET_LINES_PER_BATCH})"
            )

        est_calls = len(cond_batches) * min(ai_passes, 2)
        print(f"  Estimated API calls (initial): {est_calls}")
        print(f"  Model: {model}")

        system_prompt = self._ai_review_system_prompt()
        response_schema = self._ai_review_response_schema()

        # Example prompt
        _ex_parts = [system_prompt]
        _ex_cond = conditions_to_review[0]
        _ex_rows = df_to_review.filter(pl.col("condition_name") == _ex_cond)
        _ex_parts.append(f"Condition: {_ex_cond}")
        for row in _ex_rows.iter_rows(named=True):
            _ex_parts.append(self._ai_review_format_match_line(row))

        print(f"\n  Example prompt (condition 1 of {n_conditions}):")
        for line in "\n".join(_ex_parts).splitlines():
            print(f"    {line}")

        # --- Create Gemini cache (only if prompt is large enough) ---
        cache_name = None
        if provider == "gemini":
            from tctk._utils import _GEMINI_CACHE_MIN_TOKENS
            est_tokens = len(system_prompt) // 4
            cache_name = create_gemini_cache(
                system_prompt, check_api_key(self._api_key), model,
            )
            if cache_name:
                print(f"\n  Gemini cache created: {cache_name}")
            else:
                print(f"\n  System prompt ~{est_tokens} tokens, below Gemini cache minimum "
                      f"({_GEMINI_CACHE_MIN_TOKENS:,} tokens). Falling back to full prompts.")

        calls_used = 0

        try:
            if ai_passes == 1:
                # --- Single-pass mode (backward compatible) ---
                print(f"\n  Running single pass...")
                pass1 = self._run_single_pass(
                    df_to_review, cond_batches, system_prompt,
                    response_schema, model, 1, cache_name,
                    provider=provider,
                )
                calls_used += len(cond_batches)

                # Build vote results directly (no voting for single pass)
                vote_results = {}
                for v in pass1:
                    key = (v["condition_name"], v["search_term"], v[self._TARGET_ID_COL])
                    vote_results[key] = {
                        "ai_verdict": v["ai_verdict"],
                        "ai_vote": "1/1",
                        "ai_vote_confidence": "single",
                        "comments": [v.get("ai_comment", "")],
                    }

            else:
                # --- Multi-pass with adaptive replication ---
                print(f"\n  Running pass 1 of {ai_passes}...")
                pass1 = self._run_single_pass(
                    df_to_review, cond_batches, system_prompt,
                    response_schema, model, 1, cache_name,
                    provider=provider,
                )
                calls_used += len(cond_batches)

                print(f"  Running pass 2 of {ai_passes}...")
                pass2 = self._run_single_pass(
                    df_to_review, cond_batches, system_prompt,
                    response_schema, model, 2, cache_name,
                    provider=provider,
                )
                calls_used += len(cond_batches)

                # Find agreements and disagreements
                agreements, disagreements, disagreement_keys = self._find_disagreements(
                    pass1, pass2, self._TARGET_ID_COL,
                )

                n_agree = len(agreements)
                n_disagree = len(disagreements)
                print(f"\n  Pass 1+2 results: {n_agree} agreements, {n_disagree} disagreements")

                if disagreement_keys:
                    # Build subset DataFrame for disagreements only
                    disagree_filter = pl.lit(False)
                    for cond, term, tid in disagreement_keys:
                        disagree_filter = disagree_filter | (
                            (pl.col("condition_name") == cond)
                            & (pl.col("search_term") == term)
                            & (pl.col(self._TARGET_ID_COL) == tid)
                        )
                    df_disagreements = df_to_review.filter(disagree_filter)

                    # Build batches for disagreement subset
                    disagree_conditions = df_disagreements["condition_name"].unique().to_list()
                    if batch_size is not None:
                        disagree_batches = [
                            disagree_conditions[i : i + batch_size]
                            for i in range(0, len(disagree_conditions), batch_size)
                        ]
                    else:
                        disagree_batches = self._build_review_batches(
                            df_disagreements, disagree_conditions,
                        )

                    print(f"  Running passes 3-5 on {n_disagree} disagreements "
                          f"({len(disagree_batches)} batches each)...")

                    pass3 = self._run_single_pass(
                        df_disagreements, disagree_batches, system_prompt,
                        response_schema, model, 3, cache_name,
                        provider=provider,
                    )
                    calls_used += len(disagree_batches)

                    pass4 = self._run_single_pass(
                        df_disagreements, disagree_batches, system_prompt,
                        response_schema, model, 4, cache_name,
                        provider=provider,
                    )
                    calls_used += len(disagree_batches)

                    pass5 = self._run_single_pass(
                        df_disagreements, disagree_batches, system_prompt,
                        response_schema, model, 5, cache_name,
                        provider=provider,
                    )
                    calls_used += len(disagree_batches)

                    # Compute vote confidence from all 5 passes for disagreements
                    disagree_vote = self._compute_vote_confidence(
                        [pass1, pass2, pass3, pass4, pass5],
                        self._TARGET_ID_COL,
                    )
                    # Filter to only disagreement keys
                    disagree_vote = {k: v for k, v in disagree_vote.items() if k in disagreement_keys}
                else:
                    disagree_vote = {}

                # Merge agreements + disagreement votes
                vote_results = dict(agreements)
                vote_results.update(disagree_vote)

            # --- Retry missing verdicts ---
            # Detect matches that were sent but got no verdict back
            sent_keys = set()
            for row in df_to_review.iter_rows(named=True):
                key = (row["condition_name"], row["search_term"],
                       row[self._TARGET_ID_COL])
                sent_keys.add(key)
            missing_keys = sent_keys - set(vote_results.keys())

            if missing_keys:
                print(f"\n  Retrying {len(missing_keys)} matches with missing verdicts...")
                # Build a DataFrame of missing matches
                missing_filter = pl.lit(False)
                for cond, term, tid in missing_keys:
                    missing_filter = missing_filter | (
                        (pl.col("condition_name") == cond)
                        & (pl.col("search_term") == term)
                        & (pl.col(self._TARGET_ID_COL) == tid)
                    )
                df_missing = df_to_review.filter(missing_filter)
                missing_conds = df_missing["condition_name"].unique().to_list()
                missing_batches = self._build_review_batches(df_missing, missing_conds)

                # Run 2 retry passes on missing matches
                for retry_pass in range(1, 3):
                    if not missing_keys:
                        break
                    retry_verdicts = self._run_single_pass(
                        df_missing, missing_batches, system_prompt,
                        response_schema, model, f"retry-{retry_pass}",
                        cache_name, provider=provider,
                    )
                    calls_used += len(missing_batches)

                    for v in retry_verdicts:
                        key = (v["condition_name"], v["search_term"],
                               v[self._TARGET_ID_COL])
                        if key in missing_keys:
                            vote_results[key] = {
                                "ai_verdict": v["ai_verdict"],
                                "ai_vote": "1/1",
                                "ai_vote_confidence": "single",
                                "comments": [v.get("ai_comment", "")],
                            }
                            missing_keys.discard(key)

                if missing_keys:
                    print(f"  Still missing after retries: {len(missing_keys)}")
                else:
                    print(f"  All missing verdicts recovered.")

            # --- Compute comment consistency ---
            comment_results = self._compute_comment_consistency(vote_results)

            # --- Compute combined confidence ---
            combined_results = self._compute_combined_confidence(vote_results, comment_results)

            # --- Build verdict DataFrame ---
            verdict_rows = []
            for key, vr in vote_results.items():
                cr = comment_results.get(key, {})
                cc = combined_results.get(key, {})

                # Flag low/inconclusive combined confidence as "human review"
                combined = cc.get("ai_combined_confidence", "inconclusive")
                verdict = vr["ai_verdict"]
                if combined in ("low", "inconclusive"):
                    verdict = "human review"

                verdict_rows.append({
                    "condition_name": key[0],
                    "search_term": key[1],
                    self._TARGET_ID_COL: key[2],
                    "ai_verdict": verdict,
                    "ai_vote": vr["ai_vote"],
                    "ai_vote_confidence": vr["ai_vote_confidence"],
                    "ai_comment": cr.get("ai_comment", ""),
                    "ai_comment_consistency": cr.get("ai_comment_consistency", 0),
                    "ai_comment_consistency_tier": cr.get("ai_comment_consistency_tier", "low"),
                    "ai_combined_confidence": combined,
                })

            if verdict_rows:
                df_verdicts = pl.DataFrame(verdict_rows).unique(
                    subset=["condition_name", "search_term", self._TARGET_ID_COL]
                )
                df_matches = df_matches.join(
                    df_verdicts,
                    on=["condition_name", "search_term", self._TARGET_ID_COL],
                    how="left",
                )
            else:
                df_matches = df_matches.with_columns(*_null_cols)

        finally:
            # --- Clean up Gemini cache ---
            if cache_name and provider == "gemini":
                delete_gemini_cache(cache_name, check_api_key(self._api_key))
                print(f"  Gemini cache deleted")

        # --- Summary ---
        df_reviewed = df_matches.filter(pl.col("ai_verdict").is_not_null())

        n_accepted = len(df_reviewed.filter(pl.col("ai_verdict") == "accept"))
        n_rejected = len(df_reviewed.filter(pl.col("ai_verdict") == "reject"))
        n_human_review = len(df_reviewed.filter(pl.col("ai_verdict") == "human review"))
        n_total = n_accepted + n_rejected + n_human_review

        # Vote confidence breakdown
        vc_counts = {}
        if "ai_vote_confidence" in df_reviewed.columns:
            for row in (
                df_reviewed.filter(pl.col("ai_vote_confidence").is_not_null())
                .group_by("ai_vote_confidence").len()
                .iter_rows(named=True)
            ):
                vc_counts[row["ai_vote_confidence"]] = row["len"]

        # Combined confidence breakdown
        rel_counts = {}
        if "ai_combined_confidence" in df_reviewed.columns:
            for row in (
                df_reviewed.filter(pl.col("ai_combined_confidence").is_not_null())
                .group_by("ai_combined_confidence").len()
                .iter_rows(named=True)
            ):
                rel_counts[row["ai_combined_confidence"]] = row["len"]

        print(f"\n{'=' * 40}")
        print(f"  AI REVIEW SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Provider:          {provider}")
        print(f"  Model used:        {model}")
        print(f"  API calls used:    {calls_used}")
        print(f"  AI passes:         {ai_passes}")
        print(f"  Total reviewed:    {n_total} matches")
        print(f"    accepted:        {n_accepted} matches")
        print(f"    rejected:        {n_rejected} matches")
        print(f"    human review:    {n_human_review} matches")
        if vc_counts:
            print(f"  Vote confidence:")
            for tier in ("strong", "high", "moderate", "inconclusive", "single"):
                if tier in vc_counts:
                    print(f"    {tier}: {vc_counts[tier]}")
        if rel_counts:
            print(f"  Combined confidence:")
            for tier in ("strong", "high", "moderate", "low", "inconclusive"):
                if tier in rel_counts:
                    print(f"    {tier}: {rel_counts[tier]}")
        print(f"{'=' * 40}")

        results["df_matches"] = df_matches
        return results
