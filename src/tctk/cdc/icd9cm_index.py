"""
CDC ICD-9-CM Disease Index — download, parse, and lookup.

The CDC publishes the ICD-9-CM Disease Index as an RTF file inside a ZIP
archive (~1.5 MB).  This module downloads it once, converts RTF to plain
text via ``striprtf`` (cross-platform) with a macOS ``textutil`` fallback,
caches the result, and provides instant dict-based lookups against ~31K
unique terms.

Usage::

    from tctk.cdc import CDCIndex9

    idx = CDCIndex9()                        # auto-downloads on first use
    idx.lookup("diabetes mellitus")          # -> [{"code": "250.0-", ...}]
    idx.fuzzy_lookup("autoimmune hepatitis") # -> [{"code": "571.42", ...}]
"""

import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from tctk._utils import strip_accents

__all__ = ["CDCIndex9", "get_cdc_icd9_index"]

# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "tctk" / "cdc"

_CDC_ICD9_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/"
    "Publications/ICD9-CM/2011/DINDEX12.ZIP"
)

_CACHE_TXT = _CACHE_DIR / "icd9cm-dindex.txt"


def _rtf_to_text(rtf_bytes: bytes) -> str:
    r"""Convert ICD-9 Disease Index RTF to indented plain text.

    Parses ``\pard`` / ``\li`` RTF control words to recover the
    hierarchical indentation that generic converters discard.

    State machine: ``\pard`` resets the left-indent to 0, ``\li<N>``
    sets it.  Subsequent ``\par`` lines inherit the current indent
    until the next ``\pard``.
    """
    raw = rtf_bytes.decode("cp1252", errors="replace")

    # Determine the indent unit (smallest non-zero \li value)
    li_values = sorted(
        set(int(v) for v in re.findall(r"\\li(\d+)", raw)) - {0}
    )
    indent_unit = li_values[0] if li_values else 360

    # Split on \par (paragraph break)
    segments = re.split(r"\\par\b\s*", raw)

    current_li = 0
    lines: list[str] = []

    for seg in segments:
        # \pard resets paragraph formatting
        if re.search(r"\\pard\b", seg):
            current_li = 0

        # \li<N> sets left indent (may follow \pard in same segment)
        li_matches = re.findall(r"\\li(\d+)", seg)
        if li_matches:
            current_li = int(li_matches[-1])

        # Strip RTF markup to plain text
        text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes.fromhex(m.group(1)).decode("cp1252", errors="replace"), seg)   # hex escapes -> actual chars
        text = re.sub(r"\\[a-z]+-?\d*\s?", "", text)     # control words (delimiter space consumed)
        text = re.sub(r"\\.", "", text)                    # other escapes
        text = re.sub(r"[{}]", "", text)                   # braces
        # Fix single-char fragments from mid-word RTF breaks (e.g. "c omplicating" -> "complicating")
        text = re.sub(r"(?<!\S)([A-Za-z])\s+(?=[a-z])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        indent_level = current_li // indent_unit if indent_unit else 0
        lines.append("  " * indent_level + text)

    return "\n".join(lines) + "\n"


def get_cdc_icd9_index(txt_path: Optional[str | Path] = None) -> Path:
    """Download and cache the CDC ICD-9-CM Disease Index as plain text.

    Args:
        txt_path (str or Path, optional): Path to a pre-extracted plain-text file. If provided,
            the file is used directly (offline mode) — no download is attempted.

    Returns:
        Path: Absolute path to the cached text file.
    """
    if txt_path is not None:
        return Path(txt_path)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _CACHE_TXT.exists():
        return _CACHE_TXT

    import io
    import zipfile

    url = _CDC_ICD9_URL
    print(f"  Downloading CDC ICD-9-CM Disease Index...")
    print(f"  URL: {url}")

    try:
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            zip_bytes = resp.read()
    except Exception as exc:
        print(
            f"\n  Failed to download CDC ICD-9-CM index.\n"
            f"  Error: {exc}\n"
            f"  Download manually from:\n"
            f"    {_CDC_ICD9_URL}\n"
            f"  Then pass the extracted text file path:\n"
            f'    CDCIndex9(txt_path="/path/to/dindex.txt")'
        )
        raise

    # Find the RTF file inside the ZIP
    rtf_pattern = re.compile(r"dindex.*\.rtf$", re.IGNORECASE)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        rtf_name = None
        for name in zf.namelist():
            basename = name.rsplit("/", 1)[-1] if "/" in name else name
            if rtf_pattern.match(basename):
                rtf_name = name
                break

        if rtf_name is None:
            raise FileNotFoundError(
                f"Could not find Disease Index RTF in ZIP. Contents: {zf.namelist()}"
            )

        rtf_bytes = zf.read(rtf_name)

    # Convert RTF -> text and cache
    text = _rtf_to_text(rtf_bytes)
    _CACHE_TXT.write_text(text, encoding="utf-8")

    size_kb = _CACHE_TXT.stat().st_size / 1024
    print(f"  Cached: {_CACHE_TXT} ({size_kb:.0f} KB)")
    return _CACHE_TXT


# ---------------------------------------------------------------------------
# CDCIndex9 — parse + lookup
# ---------------------------------------------------------------------------

class CDCIndex9:
    """Parsed ICD-9-CM Disease Index with instant dict-based lookups.

    Args:
        txt_path (str or Path, optional): Path to the plain-text disease index. If None,
            auto-downloads via ``get_cdc_icd9_index()``.
        normalize_fn (callable, optional): Default normalize function for :meth:`lookup`. If None,
            a built-in normalizer is used (lowercase, hyphens→spaces, strip parentheticals,
            collapse whitespace).
    """

    def __init__(
        self,
        txt_path: Optional[str | Path] = None,
        normalize_fn: Optional[Callable[[str], str]] = None,
    ):
        self._txt_path = get_cdc_icd9_index(txt_path)
        self._default_normalize_fn = normalize_fn or self._default_normalize

        # Primary lookup: normalized_title -> [(code, original_title), ...]
        self._code_lookup: dict[str, list[tuple[str, str]]] = {}

        # See references: normalized_title -> see_target_text
        self._see_refs: dict[str, str] = {}

        # See-also references (informational, not resolved)
        self._see_also_refs: dict[str, str] = {}

        self._parse()
        self._resolve_see_refs()
        self._build_tokensort_index()
        self._build_code_names()

    # -------------------------------------------------------------------
    # Default normalizer (identical to CDCIndex._default_normalize)
    # -------------------------------------------------------------------

    @staticmethod
    def _default_normalize(text: str) -> str:
        """Normalize text: strip accents, lowercase, hyphens→spaces,
        strip possessives ('s), strip parentheticals/brackets,
        strip punctuation, collapse whitespace."""
        t = strip_accents(text)
        t = t.lower()
        t = t.replace("-", " ")
        t = re.sub(r"'s\b", "", t)
        t = re.sub(r"\s*\([^)]*\)", "", t)
        t = t.replace(")", "")  # strip stray ) from nested parens
        t = re.sub(r"\s*\[[^\]]*\]", "", t)      # strip []
        t = t.replace("[", "")                   # strip stray [
        t = t.replace("]", "")                   # strip stray ]
        for ch in ',:;"/':
            t = t.replace(ch, " ")
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    # -------------------------------------------------------------------
    # Text parsing (indentation-based ICD-9-CM format)
    # -------------------------------------------------------------------

    # Regex to match an ICD-9-CM code at the end of a line
    # e.g. "Diabetes mellitus 250.0" or "  with coma 250.30"
    _CODE_RE = re.compile(r"\b(\d{3}(?:\.\d{1,2})?)\s*(?:\[\d{3}(?:\.\d{1,2})?\])?\s*$")

    # Regex to detect "Note" instructional headers (not clinical terms).
    # Matches "Note " and "N ote" (RTF may break the word).
    _NOTE_RE = re.compile(r"N\s*ote\b", re.IGNORECASE)

    def _parse(self) -> None:
        """Parse the plain-text disease index and build lookup dicts.

        Uses two-level hierarchy tracking:
        - **Main terms** start with an uppercase letter (e.g. "Diabetes").
        - **Sub-terms** start with a lowercase letter (e.g. "with coma")
          and are prefixed with the current main term.
        - **Deeper sub-entries** (indent > base) use an indent stack to
          build multi-level terms (e.g. "Diabetes coma due to secondary
          diabetes").
        """
        normalize = self._default_normalize
        text = self._txt_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        # Current main term (last uppercase-starting entry at base indent)
        main_term = ""
        main_indent = -1

        # Stack for entries deeper than the main term's base level
        sub_stack: list[tuple[int, str]] = []

        for raw_line in lines:
            stripped = raw_line.rstrip()
            if not stripped:
                continue

            content = stripped.lstrip()
            indent = len(stripped) - len(content)

            # Skip indent-0 lines: file headers, section letters, Notes
            if indent == 0:
                continue

            # Skip short headers (single letter/number)
            if len(content) <= 2 and not self._CODE_RE.search(content):
                continue

            # Skip "Note ..." instructional headers
            if self._NOTE_RE.match(content):
                continue

            # --- Determine if this is a main term or sub-term ---
            first_alpha = ""
            for ch in content:
                if ch.isalpha():
                    first_alpha = ch
                    break

            is_main = first_alpha.isupper() and (
                main_indent < 0 or indent <= main_indent + 2
            )

            if is_main:
                main_indent = indent
                sub_stack = []
            else:
                # Sub-entry: pop sub_stack to current indent
                while sub_stack and sub_stack[-1][0] >= indent:
                    sub_stack.pop()

            # --- Extract term, see refs, codes ---
            term_part = content

            see_match = re.search(r"\s*-\s*see\s+", content, re.IGNORECASE)
            see_also_match = re.search(
                r"(?:\s*-\s*see\s+also\s+|\(see\s+also\s+)",
                content, re.IGNORECASE,
            )
            code_match = self._CODE_RE.search(content)

            if see_also_match:
                pos = see_also_match.start()
                term_part = content[:pos].strip()
                target = content[see_also_match.end():].rstrip().rstrip(")")
                if term_part:
                    full_term = self._build_full_term_v2(
                        main_term, sub_stack, term_part, is_main,
                    )
                    norm = normalize(full_term)
                    if norm:
                        self._see_also_refs[norm] = target
            elif see_match and not see_also_match:
                pos = see_match.start()
                term_part = content[:pos].strip()
                target = content[see_match.end():].rstrip()
                if term_part:
                    full_term = self._build_full_term_v2(
                        main_term, sub_stack, term_part, is_main,
                    )
                    norm = normalize(full_term)
                    if norm:
                        self._see_refs[norm] = target

            if code_match:
                code = code_match.group(1)
                term_part = content[:code_match.start()].strip()
                if term_part:
                    full_term = self._build_full_term_v2(
                        main_term, sub_stack, term_part, is_main,
                    )
                    norm = normalize(full_term)
                    if norm:
                        self._code_lookup.setdefault(norm, []).append(
                            (code, full_term)
                        )

            # --- Update state for subsequent lines ---
            clean_term = re.sub(
                r"\s*\d{3}(?:\.\d{1,2})?\s*$", "", term_part,
            ).strip()
            clean_term = re.sub(
                r"\s*-\s*see\s+.*$", "", clean_term, flags=re.IGNORECASE,
            ).strip()
            clean_term = re.sub(
                r"\s*\(see\s+also\s+.*$", "", clean_term, flags=re.IGNORECASE,
            ).strip()

            if not clean_term or self._NOTE_RE.match(clean_term):
                continue

            if is_main:
                main_term = clean_term
            else:
                sub_stack.append((indent, clean_term))

    @staticmethod
    def _build_full_term_v2(
        main_term: str,
        sub_stack: list[tuple[int, str]],
        current_term: str,
        is_main: bool,
    ) -> str:
        """Build the full hierarchical term."""
        if is_main:
            return current_term
        parts = []
        if main_term:
            parts.append(main_term)
        parts.extend(term for _, term in sub_stack)
        parts.append(current_term)
        return " ".join(parts)

    @staticmethod
    def _build_full_term(
        indent_stack: list[tuple[int, str]], current_term: str
    ) -> str:
        """Build the full hierarchical term from indent stack + current."""
        parts = [term for _, term in indent_stack]
        parts.append(current_term)
        return " ".join(parts)

    # -------------------------------------------------------------------
    # See reference resolution
    # -------------------------------------------------------------------

    def _resolve_see_refs(self) -> None:
        """Resolve 'see' references and add resolved codes to _code_lookup."""
        normalize = self._default_normalize

        # Count incoming see-refs per target main term.
        # Targets referenced by many terms are generic navigational hubs
        # (e.g. "condition" ×479, "injury" ×207, "fracture" ×216)
        # and should NOT be resolved — they produce massive false positives.
        from collections import Counter
        target_counts: Counter[str] = Counter()
        for see_target in self._see_refs.values():
            main = normalize(see_target.split(",")[0].strip())
            target_counts[main] += 1

        _GENERIC_THRESHOLD = 30
        generic_targets = {t for t, c in target_counts.items() if c > _GENERIC_THRESHOLD}

        for norm_title, see_target in self._see_refs.items():
            # Already has direct codes — skip
            if norm_title in self._code_lookup:
                continue

            # Skip single-word generic keys (e.g. "chronic" -> see condition)
            if len(norm_title.split()) <= 1:
                continue

            # Parse comma-separated target: "Abortion, induced" →
            # try "abortion induced" as a single normalized key
            parts = [p.strip() for p in see_target.split(",")]
            combined = " ".join(parts)
            target_norm = normalize(combined)

            # Skip generic navigational targets (condition, injury, fracture, ...)
            main_norm = normalize(parts[0])
            if main_norm in generic_targets and len(parts) == 1:
                continue

            codes = self._code_lookup.get(target_norm)
            if codes:
                self._code_lookup[norm_title] = list(codes)
                continue

            # Try just the main term (only if not generic)
            if main_norm not in generic_targets:
                codes = self._code_lookup.get(main_norm)
                if codes:
                    self._code_lookup[norm_title] = list(codes)

    # -------------------------------------------------------------------
    # Code → names index (for full-name lookup)
    # -------------------------------------------------------------------

    def _build_code_names(self) -> None:
        """Build code → list of original names for full-name lookup."""
        self._code_to_names: dict[str, list[str]] = {}
        for entries in self._code_lookup.values():
            for code, name in entries:
                self._code_to_names.setdefault(code, []).append(name)

    def _best_cdc_name(self, code: str, search_term: str) -> str:
        """Find the most descriptive CDC name for *code* relevant to *search_term*.

        Picks the name with the highest token overlap with the search term,
        breaking ties by length (longer = more descriptive).
        """
        candidates = self._code_to_names.get(code, [])
        if not candidates:
            return search_term
        if len(candidates) == 1:
            return candidates[0]
        search_tokens = set(self._default_normalize(search_term).split())
        best = candidates[0]
        best_score = (-1, 0)
        for name in candidates:
            name_tokens = set(self._default_normalize(name).split())
            overlap = len(search_tokens & name_tokens)
            score = (overlap, len(name))
            if score > best_score:
                best_score = score
                best = name
        return best

    # -------------------------------------------------------------------
    # Token-sort index (word-order-invariant lookup)
    # -------------------------------------------------------------------

    def _build_tokensort_index(self) -> None:
        """Build token-sorted lookup and collision index.

        For each normalized key in ``_code_lookup``, computes a canonical
        form by sorting tokens alphabetically.  Keys whose sorted form
        maps to different code sets are marked as *collisions* — these
        require AI disambiguation at lookup time (returned as score=99).
        """
        sorted_to_originals: dict[str, list[str]] = {}
        for key in self._code_lookup:
            sorted_key = " ".join(sorted(key.split()))
            sorted_to_originals.setdefault(sorted_key, []).append(key)

        self._tokensort_lookup = sorted_to_originals

        # Collision = sorted_key maps to original keys with different code sets
        self._tokensort_collisions: set[str] = set()
        for sorted_key, orig_keys in sorted_to_originals.items():
            if len(orig_keys) > 1:
                code_sets = [
                    frozenset(c for c, _ in self._code_lookup[k])
                    for k in orig_keys
                ]
                if len(set(code_sets)) > 1:
                    self._tokensort_collisions.add(sorted_key)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def lookup(
        self,
        term: str,
        normalize_fn: Optional[Callable[[str], str]] = None,
    ) -> list[dict]:
        """Look up a term in the ICD-9-CM index (exact normalized match).

        Args:
            term (str): The search term (e.g. "diabetes mellitus").
            normalize_fn (callable, optional): Normalization function. If None, uses the instance default.

        Returns:
            list[dict]: Each dict has keys ``"code"`` and ``"name"``.
        """
        fn = normalize_fn or self._default_normalize_fn
        norm = fn(term)
        entries = self._code_lookup.get(norm)
        if entries:
            return [{"code": c, "name": norm, "matched_query": norm,
                     "cdc_name": self._best_cdc_name(c, term)}
                    for c, n in entries]

        # Token-sorted fallback (non-collision only)
        sorted_norm = " ".join(sorted(norm.split()))
        ts_originals = self._tokensort_lookup.get(sorted_norm)
        if ts_originals and sorted_norm not in self._tokensort_collisions:
            results = []
            seen: set[str] = set()
            for orig_key in ts_originals:
                for c, n in self._code_lookup[orig_key]:
                    if c not in seen:
                        results.append({"code": c, "name": orig_key,
                                        "matched_query": norm,
                                        "cdc_name": self._best_cdc_name(c, term)})
                        seen.add(c)
            return results

        return []

    @staticmethod
    def _strip_stopwords(text: str, stopwords: set[str]) -> str:
        """Remove stopwords from text for fuzzy scoring."""
        tokens = [t for t in text.split() if t not in stopwords]
        return " ".join(tokens) if tokens else text

    # Generic medical nouns excluded from fuzzy scoring to prevent
    # false matches driven by shared non-discriminative tokens.
    _FUZZY_STOPWORDS = {
        "disease", "disorder", "syndrome", "condition", "infection",
        "of", "the", "and", "in", "with", "by", "to", "a", "an",
    }

    def fuzzy_lookup(
        self,
        term: str,
        threshold: int = 85,
        limit: int = 5,
        normalize_fn: Optional[Callable[[str], str]] = None,
        stopwords: Optional[set[str]] = None,
    ) -> list[dict]:
        """Fuzzy-match a term against all ICD-9-CM index entries.

        Uses keyword pre-filtering (4+ char tokens) to narrow candidates,
        stopword stripping, then rapidfuzz ``token_sort_ratio`` for scoring
        — same approach as CDCIndex (ICD-10) and the OMOP vocab fuzzy matching.

        Args:
            term (str): The search term.
            threshold (int): Minimum fuzzy score (0-100). Default 85.
            limit (int): Max results to return. Default 5.
            normalize_fn (callable, optional): Normalization function. If None, uses the instance default.
            stopwords (set[str], optional): Words to strip before fuzzy scoring. If None, uses the built-in
                medical stopword set (disease, disorder, syndrome, ...).

        Returns:
            list[dict]: Each dict has ``"code"``, ``"name"``, and ``"score"`` keys,
            sorted by score descending.
        """
        from rapidfuzz import fuzz, process

        fn = normalize_fn or self._default_normalize_fn
        sw = stopwords if stopwords is not None else self._FUZZY_STOPWORDS
        norm = fn(term)

        # Exact match first — return immediately if found
        exact = self._code_lookup.get(norm)
        if exact:
            return [{"code": c, "name": norm, "matched_query": norm,
                     "cdc_name": self._best_cdc_name(c, term), "score": 100}
                    for c, n in exact]

        # Token-sorted exact match (catches word-order variants)
        # Collect as exact hits, then continue to fuzzy for additional matches.
        ts_results: list[dict] = []
        ts_seen: set[str] = set()
        sorted_norm = " ".join(sorted(norm.split()))
        ts_originals = self._tokensort_lookup.get(sorted_norm)
        if ts_originals:
            is_collision = sorted_norm in self._tokensort_collisions
            ts_score = 99 if is_collision else 100
            for orig_key in ts_originals:
                for c, n in self._code_lookup[orig_key]:
                    if c not in ts_seen:
                        ts_results.append({
                            "code": c, "name": orig_key,
                            "matched_query": norm,
                            "cdc_name": self._best_cdc_name(c, term),
                            "score": ts_score})
                        ts_seen.add(c)

        # Build candidate list on first call (cached per stopword set)
        cache_key = frozenset(sw) if sw else frozenset()
        if not hasattr(self, "_fuzzy_cache_key") or self._fuzzy_cache_key != cache_key:
            keys = list(self._code_lookup.keys())
            self._fuzzy_candidate_keys = keys
            self._fuzzy_candidates_stripped = [
                self._strip_stopwords(k, sw) for k in keys
            ]
            self._fuzzy_cache_key = cache_key

        keys = self._fuzzy_candidate_keys
        candidates_stripped = self._fuzzy_candidates_stripped

        # Strip stopwords from query too
        norm_stripped = self._strip_stopwords(norm, sw)

        # Keyword pre-filter: require at least one 4+ char token to appear
        keywords = [w for w in norm_stripped.split() if len(w) >= 4]
        if keywords:
            indices = [
                i for i, c in enumerate(candidates_stripped)
                if any(kw in c for kw in keywords)
            ]
        else:
            indices = list(range(len(candidates_stripped)))

        if not indices:
            return ts_results

        filtered_stripped = [candidates_stripped[i] for i in indices]

        matches = process.extract(
            norm_stripped,
            filtered_stripped,
            scorer=fuzz.token_sort_ratio,
            limit=limit,
            score_cutoff=threshold,
        )

        results = list(ts_results)
        for _, score, match_idx in matches:
            original_idx = indices[match_idx]
            original_key = keys[original_idx]
            stripped_key = candidates_stripped[original_idx]
            for code, name in self._code_lookup[original_key]:
                if code not in ts_seen:
                    results.append({
                        "code": code, "name": stripped_key,
                        "matched_query": norm_stripped,
                        "cdc_name": self._best_cdc_name(code, term),
                        "score": int(score)})
        return results

    @property
    def keys(self) -> list[str]:
        """All normalized term keys in the index."""
        return list(self._code_lookup.keys())

    def __repr__(self) -> str:
        n_terms = len(self._code_lookup)
        n_see = len(self._see_refs)
        return f"CDCIndex9({self._txt_path.name}, {n_terms} terms, {n_see} see refs)"
