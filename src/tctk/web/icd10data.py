"""
ICD10DataLookup — scrape icd10data.com for ICD-10-CM codes.

Finds codes via "Applicable To" and "Approximate Synonyms" data that
is not available through the OMOP vocabulary database or free APIs.

Usage::

    from tctk.web import ICD10DataLookup

    lookup = ICD10DataLookup()

    # Search for a condition
    df = lookup.search("Cogan's syndrome")

    # Get details for a specific code
    details = lookup.fetch_code("H16.32")

    # Batch lookup with fuzzy matching
    results = lookup.lookup({
        "Cogan's syndrome": ["Cogan syndrome", "interstitial keratitis"],
    })
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import polars as pl
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from tqdm.auto import tqdm

__all__ = ["ICD10DataLookup"]

# Browser-like headers to avoid being blocked
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Stop markers for section parsing on code detail pages
_STOP_MARKERS = {
    "icd-10-cm",
    "drg grouping",
    "convert",
    "code history",
    "diagnosis index",
    "reimbursement",
    "information for patients",
    "present on admission",
    "icd-10-cm codes",
}


class ICD10DataLookup:
    """Look up ICD-10-CM codes from icd10data.com via web scraping."""

    BASE = "https://www.icd10data.com"

    def __init__(self, delay: float = 0.5, max_workers: int = 4):
        """
        Parameters
        ----------
        delay : float
            Minimum seconds between HTTP requests (global, across all
            threads).  Default 0.5 (~2 req/s).
        max_workers : int
            Maximum number of concurrent threads for batch operations.
            Default 4.  Threading overlaps network I/O wait time while
            the global rate limiter ensures total request rate stays at
            ~1/delay req/s.
        """
        self._delay = delay
        self._max_workers = max_workers
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        # Thread-local storage for per-thread sessions
        self._local = threading.local()
        # Global rate limiter: ensures all threads collectively respect delay
        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

    def _get_session(self) -> requests.Session:
        """Return a thread-local requests session."""
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.headers.update(_HEADERS)
        return self._local.session

    def _rate_limit(self):
        """Block until enough time has passed since the last request."""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._delay:
                time.sleep(self._delay - elapsed)
            self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 25) -> pl.DataFrame:
        """Search icd10data.com for ICD-10-CM codes matching *query*.

        Parameters
        ----------
        query : str
            Free-text condition name to search.
        max_results : int
            Maximum number of results to return.  Default 25.

        Returns
        -------
        pl.DataFrame
            Columns: ``code``, ``name``, ``synonyms``, ``url``.
        """
        url = f"{self.BASE}/search"
        params = {"s": query}

        self._rate_limit()
        session = self._get_session()
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for line in soup.select("div.searchLine"):
            if len(results) >= max_results:
                break

            # Extract ICD code
            code_span = line.select_one("span.identifier")
            code = code_span.get_text(strip=True) if code_span else None

            # Extract URL
            link = line.select_one("a[href]")
            href = link["href"] if link else None
            full_url = f"{self.BASE}{href}" if href and href.startswith("/") else href

            # Extract official name (first div inside searchPadded)
            padded = line.select_one("div.searchPadded")
            name = None
            if padded:
                first_div = padded.select_one("div")
                if first_div:
                    name = first_div.get_text(strip=True)

            # Extract approximate synonyms
            synonyms_span = line.select_one("span.searchKeywords")
            synonyms = ""
            if synonyms_span:
                synonyms = synonyms_span.get_text(strip=True)

            if code:
                results.append({
                    "code": code,
                    "name": name or "",
                    "synonyms": synonyms,
                    "url": full_url or "",
                })

        return pl.DataFrame(results) if results else pl.DataFrame(
            schema={"code": pl.Utf8, "name": pl.Utf8, "synonyms": pl.Utf8, "url": pl.Utf8}
        )

    # ------------------------------------------------------------------
    # fetch_code
    # ------------------------------------------------------------------

    def _resolve_code_url(self, code: str) -> Optional[str]:
        """Resolve a bare ICD-10 code to its full icd10data.com URL.

        The site uses hierarchical paths (e.g.
        ``/ICD10CM/Codes/D50-D89/D55-D59/D59-/D59.12``) that cannot be
        constructed from the code alone.  This method searches for the
        code and extracts the canonical URL from the first matching
        result.

        Returns None if the code cannot be found.
        """
        df = self.search(code, max_results=5)
        if len(df) == 0:
            return None
        # Find exact code match in search results
        for row in df.iter_rows(named=True):
            if row["code"] == code:
                return row["url"]
        # Fall back to first result
        return df["url"][0]

    def fetch_code(self, code_or_url: str) -> dict:
        """Fetch detail page for an ICD-10-CM code.

        Parameters
        ----------
        code_or_url : str
            Either a bare ICD-10 code (e.g. ``"H16.32"``) or a full URL.

        Returns
        -------
        dict
            Keys: ``code``, ``name``, ``applicable_to``, ``approximate_synonyms``.
        """
        if code_or_url.startswith("http") or code_or_url.startswith("/"):
            url = code_or_url
            if not url.startswith("http"):
                url = f"{self.BASE}/{url.lstrip('/')}"
        else:
            # Resolve bare code via search to get the correct hierarchical URL
            resolved = self._resolve_code_url(code_or_url)
            if resolved is None:
                raise ValueError(f"Code {code_or_url!r} not found on icd10data.com")
            url = resolved
            if not url.startswith("http"):
                url = f"{self.BASE}/{url.lstrip('/')}"

        self._rate_limit()
        session = self._get_session()
        resp = session.get(url, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Code and name from page header.
        # h1 format: "2026 ICD-10-CM Diagnosis CodeH16.32"
        # Actual name is in the first h2.codeDescription: "Diffuse interstitial keratitis"
        code = code_or_url if not code_or_url.startswith("http") else ""
        name = ""

        h1 = soup.select_one("h1")
        if h1:
            h1_text = h1.get_text(strip=True)
            # Try legacy "CODE - Name" format first
            m = re.match(r"([A-Z]\d[\dA-Z.]+)\s*[-–—]\s*(.+)", h1_text)
            if m:
                code = m.group(1).strip()
                name = m.group(2).strip()
            else:
                # Current format: extract code from end of h1
                m2 = re.search(r"([A-Z]\d[\dA-Z.]+)\s*$", h1_text)
                if m2:
                    code = m2.group(1).strip()

        # Name from h2.codeDescription (more reliable than h1)
        h2_name = soup.select_one("h2.codeDescription")
        if h2_name:
            name = h2_name.get_text(strip=True)

        if not name:
            title = soup.select_one("title")
            if title:
                name = title.get_text(strip=True)

        # Parse "Applicable To" and "Approximate Synonyms" sections.
        # DOM structure: <span>Section Label</span> followed by sibling <ul><li>items</li></ul>
        # or wrapped in a parent <div>: <div><span>Label</span><ul><li>...</li></ul></div>
        body = soup.select_one("div.body-content") or soup.body or soup
        applicable_to, approximate_synonyms = self._parse_sections(body)

        return {
            "code": code,
            "name": name,
            "applicable_to": applicable_to,
            "approximate_synonyms": approximate_synonyms,
        }

    @staticmethod
    def _parse_sections(container) -> tuple[list[str], list[str]]:
        """Parse 'Applicable To' and 'Approximate Synonyms' from a page container.

        The page uses ``<span>Label</span>`` followed by a sibling ``<ul>``
        with ``<li>`` items.  We only collect from the *first* occurrence of
        each label that directly belongs to the current code (not ancestor
        code descriptions embedded in the page hierarchy).
        """
        applicable_to: list[str] = []
        approximate_synonyms: list[str] = []

        sections = {
            "applicable to": applicable_to,
            "approximate synonyms": approximate_synonyms,
        }

        for label, target_list in sections.items():
            for span in container.find_all(
                "span", string=lambda t: t and t.strip().lower() == label
            ):
                # Look for the <ul> sibling immediately after this span
                ul = span.find_next_sibling("ul")
                if ul is None:
                    # Span may be inside a wrapper <div>; check parent's children
                    parent = span.parent
                    if parent and parent.name == "div":
                        ul = parent.find("ul")
                if ul is None:
                    continue

                for li in ul.find_all("li", recursive=False):
                    text = li.get_text(strip=True)
                    if text and len(text) > 1 and text not in target_list:
                        target_list.append(text)

                # Only use the first valid occurrence (current code level)
                if target_list:
                    break

        return applicable_to, approximate_synonyms

    # ------------------------------------------------------------------
    # validate_codes (batch fetch detail pages)
    # ------------------------------------------------------------------

    def validate_codes(
        self,
        codes: list[str],
    ) -> dict[str, dict]:
        """Fetch detail pages for a list of ICD-10 codes (threaded).

        Returns a dict mapping code → {applicable_to: [...], approximate_synonyms: [...]}.
        Caches results internally to avoid refetching the same code twice.

        Parameters
        ----------
        codes : list[str]
            ICD-10-CM codes to look up (e.g. ``["H16.32", "M35.0"]``).

        Returns
        -------
        dict[str, dict]
            Mapping of code → ``{"applicable_to": [...], "approximate_synonyms": [...]}``.
        """
        if not hasattr(self, "_code_cache"):
            self._code_cache: dict[str, dict] = {}

        result = {}
        to_fetch = []
        for code in codes:
            if code in self._code_cache:
                result[code] = self._code_cache[code]
            else:
                to_fetch.append(code)

        if not to_fetch:
            return result

        def _fetch_one(code: str) -> tuple[str, dict]:
            try:
                details = self.fetch_code(code)
                return code, {
                    "applicable_to": details.get("applicable_to", []),
                    "approximate_synonyms": details.get("approximate_synonyms", []),
                }
            except Exception:
                return code, {"applicable_to": [], "approximate_synonyms": []}

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_fetch_one, code): code for code in to_fetch}
            for future in tqdm(
                as_completed(futures), total=len(to_fetch),
                desc="Fetching ICD-10 codes",
            ):
                code, entry = future.result()
                self._code_cache[code] = entry
                result[code] = entry

        return result

    # ------------------------------------------------------------------
    # lookup (batch)
    # ------------------------------------------------------------------

    def lookup(
        self,
        conditions: dict[str, list[str]],
        fuzzy_threshold: int = 85,
        fetch_details: bool = False,
        export_tsv: bool = False,
        export_prefix: str = "icd10data_lookup",
    ) -> dict:
        """Batch lookup: search icd10data.com for each condition and match results.

        Parameters
        ----------
        conditions : dict[str, list[str]]
            Keys are condition names; values are lists of synonyms.
        fuzzy_threshold : int
            Minimum rapidfuzz score (0-100) for matching.  Default 85.
        fetch_details : bool
            If True, also fetch code detail pages for "Applicable To" data.
            Slower due to extra HTTP requests.  Default False.
        export_tsv : bool
            If True, write results to ``{export_prefix}_results.tsv``.
        export_prefix : str
            Filename prefix for exported files.  Default ``"icd10data_lookup"``.

        Returns
        -------
        dict
            Keys: ``df_results`` (pl.DataFrame with columns: condition_name,
            search_term, icd_code, icd_name, match_source, match_score).
        """
        all_rows = []

        for condition_name, synonyms in conditions.items():
            search_terms = [condition_name] + [s for s in synonyms if s.strip()]

            for term in search_terms:
                time.sleep(self._delay)
                print(f"  Searching: {term!r}")

                try:
                    df_search = self.search(term)
                except Exception as e:
                    print(f"    Error searching {term!r}: {e}")
                    continue

                if len(df_search) == 0:
                    continue

                # Match search term against returned names + synonyms
                for row in df_search.iter_rows(named=True):
                    code = row["code"]
                    name = row["name"]
                    syns_text = row["synonyms"]
                    code_url = row["url"]

                    # Check name match
                    score_name = fuzz.token_sort_ratio(
                        term.lower(), name.lower()
                    )
                    if score_name >= fuzzy_threshold:
                        all_rows.append({
                            "condition_name": condition_name,
                            "search_term": term,
                            "icd_code": code,
                            "icd_name": name,
                            "match_source": "name",
                            "match_score": int(score_name),
                        })

                    # Check synonym matches
                    if syns_text:
                        for syn in re.split(r"[;,]", syns_text):
                            syn = syn.strip()
                            if not syn:
                                continue
                            score_syn = fuzz.token_sort_ratio(
                                term.lower(), syn.lower()
                            )
                            if score_syn >= fuzzy_threshold:
                                all_rows.append({
                                    "condition_name": condition_name,
                                    "search_term": term,
                                    "icd_code": code,
                                    "icd_name": name,
                                    "match_source": "synonym",
                                    "match_score": int(score_syn),
                                })

                    # Optionally fetch detail page for "Applicable To"
                    if fetch_details and code_url:
                        try:
                            details = self.fetch_code(code_url)
                            for app_text in details.get("applicable_to", []):
                                score_app = fuzz.token_sort_ratio(
                                    term.lower(), app_text.lower()
                                )
                                if score_app >= fuzzy_threshold:
                                    all_rows.append({
                                        "condition_name": condition_name,
                                        "search_term": term,
                                        "icd_code": code,
                                        "icd_name": name,
                                        "match_source": "applicable_to",
                                        "match_score": int(score_app),
                                    })
                        except Exception as e:
                            print(f"    Error fetching {code}: {e}")

        # Build results DataFrame
        if all_rows:
            df_results = (
                pl.DataFrame(all_rows)
                .unique(subset=["condition_name", "search_term", "icd_code", "match_source"])
                .sort(["condition_name", "search_term", "match_score"], descending=[False, False, True])
            )
        else:
            df_results = pl.DataFrame(
                schema={
                    "condition_name": pl.Utf8,
                    "search_term": pl.Utf8,
                    "icd_code": pl.Utf8,
                    "icd_name": pl.Utf8,
                    "match_source": pl.Utf8,
                    "match_score": pl.Int64,
                }
            )

        print(f"\n  Results: {len(df_results)} matches for "
              f"{df_results['condition_name'].n_unique() if len(df_results) > 0 else 0} conditions")

        results = {"df_results": df_results}

        if export_tsv and len(df_results) > 0:
            from tctk._utils import write_tsv_bom
            path = f"{export_prefix}_results.tsv"
            write_tsv_bom(df_results, path)
            print(f"  Exported: {path} ({len(df_results)} rows)")

        return results
