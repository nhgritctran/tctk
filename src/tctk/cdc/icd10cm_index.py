"""
CDC ICD-10-CM Alphabetic Index — download, parse, and lookup.

The CDC publishes the official ICD-10-CM Alphabetic Index as XML
(~9.2 MB). This module downloads it once, caches it locally, and
provides instant dict-based lookups against ~63K code entries.

Usage::

    from tctk.cdc import CDCIndex

    idx = CDCIndex()                       # auto-downloads on first use
    idx.lookup("pandas", normalize_fn)     # -> [{"code": "D89.89", ...}]
    idx.lookup("cogan's syndrome", fn)     # -> [{"code": "H16.32-", ...}]
"""

import datetime
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from xml.etree import ElementTree as ET

from tctk._utils import strip_accents

__all__ = ["CDCIndex", "get_cdc_index"]

# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "tctk" / "cdc"

_CDC_URL_TEMPLATE = (
    "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/"
    "ICD10CM/{year}/icd10cm-table and index-{year}.zip"
)

_INDEX_FILENAME_TEMPLATE = "icd10cm-index-{year}.xml"


def _detect_fiscal_year() -> int:
    """Return the ICD-10-CM fiscal year for today's date.

    ICD-10-CM updates take effect October 1 each year, so if the current
    month is October or later, the fiscal year is next calendar year.
    """
    today = datetime.date.today()
    return today.year + 1 if today.month >= 10 else today.year


def get_cdc_index(
    year: Optional[int] = None,
    xml_path: Optional[str | Path] = None,
) -> Path:
    """Download and cache the CDC ICD-10-CM Alphabetic Index XML.

    Args:
        year (int, optional): Fiscal year (e.g. 2026). If None, auto-detect from current date.
        xml_path (str or Path, optional): Path to a pre-extracted XML file. If provided, the file
            is used directly (offline mode) — no download is attempted.

    Returns:
        Path: Absolute path to the cached XML file.
    """
    if xml_path is not None:
        return Path(xml_path)

    if year is None:
        year = _detect_fiscal_year()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_path = _CACHE_DIR / _INDEX_FILENAME_TEMPLATE.format(year=year)

    if cached_path.exists():
        return cached_path

    url = _CDC_URL_TEMPLATE.format(year=year)
    print(f"  Downloading CDC ICD-10-CM index ({year})...")
    print(f"  URL: {url}")

    import urllib.parse
    import urllib.request
    # Percent-encode spaces (and other special chars) in the URL path
    url = urllib.parse.quote(url, safe=":/?#[]@!$&'()*+,;=-._~")
    try:
        with urllib.request.urlopen(url) as resp:
            zip_bytes = resp.read()
    except Exception as exc:
        print(
            f"\n  Failed to download CDC ICD-10-CM index.\n"
            f"  Error: {exc}\n"
            f"  Download manually from:\n"
            f"    https://ftp.cdc.gov/pub/health_statistics/nchs/publications/"
            f"ICD10CM/{year}/icd10cm-table and index-{year}.zip\n"
            f"  Then pass the extracted XML path:\n"
            f'    CDCIndex(xml_path="/path/to/icd10cm-index.xml")'
        )
        raise

    # Extract just the index XML from the ZIP
    index_pattern = re.compile(r"icd10cm[_-]?index.*\.xml$", re.IGNORECASE)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_name = None
        for name in zf.namelist():
            basename = name.rsplit("/", 1)[-1] if "/" in name else name
            if index_pattern.match(basename):
                xml_name = name
                break

        if xml_name is None:
            raise FileNotFoundError(
                f"Could not find index XML in ZIP. Contents: {zf.namelist()}"
            )

        cached_path.write_bytes(zf.read(xml_name))

    size_mb = cached_path.stat().st_size / (1024 * 1024)
    print(f"  Cached: {cached_path} ({size_mb:.1f} MB)")
    return cached_path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TermNode:
    """A node in the ICD-10-CM index term tree."""
    title: str
    code: Optional[str] = None
    see: Optional[str] = None
    see_also: Optional[str] = None
    children: list["TermNode"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CDCIndex — parse + lookup
# ---------------------------------------------------------------------------

class CDCIndex:
    """Parsed ICD-10-CM Alphabetic Index with instant dict-based lookups.

    Args:
        xml_path (str or Path, optional): Path to the index XML. If None, auto-downloads via ``get_cdc_index()``.
        year (int, optional): Fiscal year, passed to ``get_cdc_index()`` when ``xml_path`` is None.
        normalize_fn (callable, optional): Default normalize function for :meth:`lookup`. If None, a built-in
            normalizer is used (lowercase, hyphens→spaces, strip parentheticals,
            collapse whitespace).
    """

    def __init__(
        self,
        xml_path: Optional[str | Path] = None,
        year: Optional[int] = None,
        normalize_fn: Optional[Callable[[str], str]] = None,
    ):
        self._xml_path = get_cdc_index(year=year, xml_path=xml_path)
        self._default_normalize_fn = normalize_fn or self._default_normalize

        # Primary lookup: normalized_title -> [(code, original_title), ...]
        self._code_lookup: dict[str, list[tuple[str, str]]] = {}

        # See references: normalized_title -> see_target_text
        self._see_refs: dict[str, str] = {}

        # Hierarchical structure for resolving comma-separated see refs
        self._main_terms: dict[str, TermNode] = {}

        self._parse()
        self._resolve_see_refs()
        self._build_tokensort_index()
        self._build_code_names()

    # -------------------------------------------------------------------
    # Default normalizer (matches _normalize_for_fuzzy in condition2icd)
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
    # XML parsing
    # -------------------------------------------------------------------

    @staticmethod
    def _extract_title(elem) -> str:
        """Extract full title text from an element, including nemod content."""
        # The title element may contain <nemod> children with parenthetical text.
        # We want the full text including nemod for the "original" title,
        # and _normalize will strip the parentheticals.
        title_elem = elem.find("title")
        if title_elem is None:
            return ""

        parts = []
        if title_elem.text:
            parts.append(title_elem.text)
        for child in title_elem:
            # <nemod> text already contains parentheses in the XML,
            # e.g. "(acute) (chronic)".  Append as-is with a space separator
            # so normalize can strip them cleanly.
            if child.text:
                parts.append(f" {child.text}")
            if child.tail:
                parts.append(child.tail)

        return "".join(parts).strip()

    def _parse_term_node(self, elem) -> TermNode:
        """Recursively parse a <mainTerm> or <term> element into a TermNode."""
        title = self._extract_title(elem)
        code_elem = elem.find("code")
        see_elem = elem.find("see")
        see_also_elem = elem.find("seeAlso")

        node = TermNode(
            title=title,
            code=code_elem.text.strip() if code_elem is not None and code_elem.text else None,
            see=see_elem.text.strip() if see_elem is not None and see_elem.text else None,
            see_also=see_also_elem.text.strip() if see_also_elem is not None and see_also_elem.text else None,
        )

        # Recursively parse child <term> elements
        for child in elem:
            if child.tag == "term":
                node.children.append(self._parse_term_node(child))

        return node

    def _register_codes(
        self, node: TermNode, normalize_fn: Callable,
        parent_titles: list[str] | None = None,
    ) -> None:
        """Register all codes from a TermNode tree into _code_lookup.

        Builds full hierarchical terms (e.g. "Headache daily chronic")
        so sub-terms are not registered as standalone keys.
        """
        if parent_titles is None:
            parent_titles = []

        # Build the full term: parent titles + this node's title
        full_parts = parent_titles + [node.title] if node.title else parent_titles
        full_term = " ".join(full_parts)

        if node.code:
            norm = normalize_fn(full_term)
            if norm:
                self._code_lookup.setdefault(norm, []).append(
                    (node.code, full_term)
                )

        if node.see:
            norm = normalize_fn(full_term)
            if norm:
                self._see_refs[norm] = node.see

        for child in node.children:
            self._register_codes(child, normalize_fn, full_parts)

    def _parse(self) -> None:
        """Parse the XML and build lookup dicts."""
        tree = ET.parse(self._xml_path)
        root = tree.getroot()

        normalize = self._default_normalize

        for letter in root.iter("letter"):
            for main_term in letter.findall("mainTerm"):
                node = self._parse_term_node(main_term)
                if node.title:
                    norm_title = normalize(node.title)
                    self._main_terms[norm_title] = node
                self._register_codes(node, normalize)

    # -------------------------------------------------------------------
    # See reference resolution
    # -------------------------------------------------------------------

    def _find_child_by_title(self, node: TermNode, sub_title: str) -> Optional[TermNode]:
        """Find a direct child node whose normalized title matches sub_title."""
        normalize = self._default_normalize
        for child in node.children:
            if normalize(child.title) == normalize(sub_title):
                return child
        return None

    def _collect_codes(
        self, node: TermNode, parent_titles: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Collect all (code, full_name) pairs from a node and its descendants.

        Builds full hierarchical names so resolved see-refs store
        e.g. "Rheumatoid carditis" instead of just "carditis".
        """
        if parent_titles is None:
            parent_titles = []

        full_parts = parent_titles + [node.title] if node.title else parent_titles
        full_term = " ".join(full_parts)

        results = []
        if node.code:
            results.append((node.code, full_term))
        for child in node.children:
            results.extend(self._collect_codes(child, full_parts))
        return results

    def _resolve_see_refs(self) -> None:
        """Resolve <see> references and add resolved codes to _code_lookup."""
        normalize = self._default_normalize

        # Count incoming see-refs per target main term.
        # Targets referenced by many terms are generic navigational hubs
        # (e.g. "condition" ×486, "disorder" ×317, "neoplasm" ×769)
        # and should NOT be resolved — they produce massive false positives.
        from collections import Counter
        target_counts: Counter[str] = Counter()
        for see_target in self._see_refs.values():
            main = normalize(see_target.split(",")[0].strip())
            target_counts[main] += 1

        # Threshold: skip targets referenced by >30 other terms
        _GENERIC_THRESHOLD = 30
        generic_targets = {t for t, c in target_counts.items() if c > _GENERIC_THRESHOLD}

        for norm_title, see_target in self._see_refs.items():
            # Already has direct codes — skip
            if norm_title in self._code_lookup:
                continue

            # Skip single-word generic keys (e.g. "chronic" -> see condition)
            # These produce false matches when fuzzy-matched against conditions
            if len(norm_title.split()) <= 1:
                continue

            # Parse "Enteritis, regional" -> main="enteritis", sub="regional"
            parts = [p.strip() for p in see_target.split(",")]
            main_key = normalize(parts[0])

            # Skip generic navigational targets (condition, disease, disorder, ...)
            if main_key in generic_targets and len(parts) == 1:
                continue

            target_node = self._main_terms.get(main_key)
            if target_node is None:
                continue

            if len(parts) > 1:
                # Walk down to the sub-term, tracking parent titles
                current = target_node
                walked_titles: list[str] = []
                for sub in parts[1:]:
                    walked_titles.append(current.title)
                    child = self._find_child_by_title(current, sub.strip())
                    if child is None:
                        current = None
                        break
                    current = child
                if current is None:
                    continue
                codes = self._collect_codes(current, walked_titles)
            else:
                codes = self._collect_codes(target_node)

            if codes:
                self._code_lookup[norm_title] = codes

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
        """Look up a term in the CDC index (exact normalized match).

        Args:
            term (str): The search term (e.g. "pandas", "cogan's syndrome").
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
    # Mirrors ConditionMapperBase._FUZZY_STOPWORDS.
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
        """Fuzzy-match a term against all CDC index entries.

        Uses keyword pre-filtering (4+ char tokens) to narrow candidates,
        stopword stripping, then rapidfuzz ``token_sort_ratio`` for scoring
        — same approach as the OMOP vocab fuzzy matching.

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
        # _fuzzy_candidates: list of stopword-stripped keys for scoring
        # _fuzzy_candidate_keys: parallel list of original keys for lookup
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
        return f"CDCIndex({self._xml_path.name}, {n_terms} terms, {n_see} see refs)"
