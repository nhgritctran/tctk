"""
Internal utilities shared across tctk modules.

Not intended for direct import by users.
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import polars as pl

__all__ = []  # internal module, no public exports

# -------------------------------------------------------------------
# Config file search paths (in priority order)
# -------------------------------------------------------------------

CONFIG_PATHS = [
    Path(".tctk_config.json"),
    Path.home() / ".config" / "tctk" / "credentials.json",
]

# Gemini tier priority (higher = better)
_TIER_PRIORITY = {"pro": 3, "flash": 2, "flash-lite": 1}


# -------------------------------------------------------------------
# SQL helpers
# -------------------------------------------------------------------

def sql_escape(s: str) -> str:
    """Escape backslashes and single quotes for SQL string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


# -------------------------------------------------------------------
# File I/O
# -------------------------------------------------------------------

def write_tsv_bom(df: pl.DataFrame, path: str) -> None:
    """Write a Polars DataFrame as TSV with UTF-8 BOM for Excel/Mac compatibility."""
    with open(path, "wb") as f:
        f.write(b"\xef\xbb\xbf")
        f.write(df.write_csv(separator="\t").encode("utf-8"))


# -------------------------------------------------------------------
# Config / credentials
# -------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load JSON config from a local file.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to config JSON file.
        If None, searches default config paths.

    Returns
    -------
    dict
        Parsed config, or empty dict if no config found.
    """
    search_paths = (
        [Path(config_path)] if config_path else CONFIG_PATHS
    )

    for path in search_paths:
        try:
            if path.is_file():
                return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

    return {}


def load_api_key(
    api_key: Optional[str] = None,
    config_path: Optional[str] = None,
    env_var: str = "GEMINI_API_KEY",
) -> Optional[str]:
    """Load API key with priority chain.

    Priority:
        1. Explicit api_key parameter
        2. Environment variable
        3. Config file (explicit path, then default search paths)

    Parameters
    ----------
    api_key : str, optional
        Directly provided API key.
    config_path : str, optional
        Path to config JSON file.
    env_var : str
        Environment variable name. Default "GEMINI_API_KEY".

    Returns
    -------
    str or None
        API key if found, None otherwise.
    """
    if api_key:
        return api_key

    env_key = os.getenv(env_var)
    if env_key:
        return env_key

    config = load_config(config_path)
    key = config.get("gemini_api_key")
    if key:
        return key

    return None


def check_api_key(api_key: Optional[str]) -> str:
    """Validate API key is available; raise ValueError with instructions if not.

    Parameters
    ----------
    api_key : str or None
        The API key to check.

    Returns
    -------
    str
        The validated API key.

    Raises
    ------
    ValueError
        If api_key is None or empty.
    """
    if not api_key:
        raise ValueError(
            "AI review requires a Gemini API key.\n\n"
            "Option 1 (recommended): Create a config file\n"
            "  Save to .tctk_config.json or ~/.config/tctk/credentials.json:\n"
            '  {"gemini_api_key": "your-key-here"}\n\n'
            "Option 2: Set environment variable\n"
            "  os.environ['GEMINI_API_KEY'] = 'your-key'\n\n"
            "Option 3: Pass directly\n"
            "  Condition2ConceptID(gemini_api_key='your-key')\n\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )
    return api_key


def setup_credentials(path: Optional[str] = None) -> None:
    """Interactive helper to create a credentials file.

    Creates a JSON file with the Gemini API key. Uses getpass to hide
    input. Sets file permissions to owner-only on Unix systems.

    Parameters
    ----------
    path : str, optional
        Path for the credentials file.
        Defaults to ~/.config/tctk/credentials.json.
    """
    import getpass

    key = getpass.getpass("Paste your Gemini API key (hidden): ")
    if not key.strip():
        print("No key provided. Aborted.")
        return

    config_path = Path(path) if path else Path.home() / ".config" / "tctk" / "credentials.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if config_path.is_file():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    existing["gemini_api_key"] = key.strip()

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)

    try:
        config_path.chmod(0o600)
    except OSError:
        pass

    # Auto-add project-level config to .gitignore
    gitignore = Path(".gitignore")
    entry = ".tctk_config.json"
    try:
        if gitignore.exists():
            content = gitignore.read_text()
            if entry not in content:
                with open(gitignore, "a") as f:
                    f.write(f"\n{entry}\n")
        else:
            gitignore.write_text(f"{entry}\n")
    except OSError:
        pass

    print(f"Credentials saved to {config_path}")


# -------------------------------------------------------------------
# Gemini model detection
# -------------------------------------------------------------------

def _parse_model_tier(model_name: str) -> Optional[str]:
    """Extract tier from a Gemini model name.

    Examples:
        "gemini-2.5-pro"       → "pro"
        "gemini-2.0-flash"     → "flash"
        "gemini-2.0-flash-lite" → "flash-lite"
    """
    name = model_name.lower()
    if "flash-lite" in name:
        return "flash-lite"
    elif "flash" in name:
        return "flash"
    elif "pro" in name:
        return "pro"
    return None


def _parse_model_version(model_name: str) -> float:
    """Extract numeric version from a Gemini model name.

    Examples:
        "gemini-2.5-pro"   → 2.5
        "gemini-2.0-flash" → 2.0
    """
    match = re.search(r"(\d+\.\d+)", model_name)
    return float(match.group(1)) if match else 0.0


def detect_best_model(
    api_key: str,
    ai_tier: Optional[str] = None,
) -> str:
    """Query Gemini API and select the best available model.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    ai_tier : str, optional
        Preferred tier: "pro", "flash", or "flash-lite".
        Default None → picks "flash" tier (cost-effective default),
        then best version within that tier.

    Returns
    -------
    str
        Full model name (e.g., "gemini-2.5-flash").

    Raises
    ------
    RuntimeError
        If no suitable models are found.
    """
    import requests

    # Default to flash tier
    if ai_tier is None:
        ai_tier = "flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        models_data = resp.json().get("models", [])
    except Exception as e:
        raise RuntimeError(f"Failed to query Gemini models: {e}") from e

    # Filter to models that support generateContent
    candidates = []
    for m in models_data:
        name = m.get("name", "").replace("models/", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue

        tier = _parse_model_tier(name)
        if tier is None:
            continue

        version = _parse_model_version(name)
        candidates.append({
            "name": name,
            "tier": tier,
            "version": version,
            "tier_priority": _TIER_PRIORITY.get(tier, 0),
        })

    if not candidates:
        raise RuntimeError(
            "No suitable Gemini models found for this API key. "
            "Ensure the Generative Language API is enabled."
        )

    # Filter by preferred tier
    ai_tier = ai_tier.lower().strip()
    tier_candidates = [c for c in candidates if c["tier"] == ai_tier]
    if tier_candidates:
        candidates = tier_candidates
    else:
        available_tiers = sorted(set(c["tier"] for c in candidates))
        print(
            f"  Warning: tier '{ai_tier}' not available. "
            f"Available: {', '.join(available_tiers)}. "
            f"Falling back to best available."
        )

    # Sort: best tier first, then highest version
    candidates.sort(key=lambda c: (c["tier_priority"], c["version"]), reverse=True)

    return candidates[0]["name"]


# -------------------------------------------------------------------
# Gemini API call
# -------------------------------------------------------------------

def call_gemini(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    timeout: int = 30,
) -> str:
    """Call Gemini API directly via REST (no SDK dependency).

    Parameters
    ----------
    prompt : str
        The prompt text.
    api_key : str
        Gemini API key.
    model : str
        Full model name (e.g., "gemini-2.5-flash").
    temperature : float
        Sampling temperature. Default 0.0 (deterministic).
    max_output_tokens : int
        Max tokens in response. Default 1024.
    timeout : int
        Request timeout in seconds. Default 30.

    Returns
    -------
    str
        Model response text.

    Raises
    ------
    RuntimeError
        If the API call fails.
    """
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Gemini API error: {e.response.status_code} - {e.response.text}"
        ) from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini API response format: {e}") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Gemini API request failed: {e}") from e


# -------------------------------------------------------------------
# Rate limiter
# -------------------------------------------------------------------

class RateLimiter:
    """Per-user rate limiter with daily quota and per-minute burst control.

    Parameters
    ----------
    daily_limit : int
        Maximum API calls per user per day. Default 50.
    rpm_limit : int
        Maximum API calls per user per minute. Default 5.
    """

    def __init__(self, daily_limit: int = 50, rpm_limit: int = 5):
        self.daily_limit = daily_limit
        self.rpm_limit = rpm_limit
        self._daily_counts: dict = defaultdict(lambda: {"count": 0, "reset": 0.0})
        self._minute_log: dict = defaultdict(list)

    def check(self, user_id: str = "default") -> tuple[bool, str]:
        """Check if a request is allowed for the given user."""
        now = time.time()

        daily = self._daily_counts[user_id]
        if now - daily["reset"] > 86400:
            daily["count"] = 0
            daily["reset"] = now

        if daily["count"] >= self.daily_limit:
            return False, "Daily AI review limit reached"

        minute_calls = [t for t in self._minute_log[user_id] if now - t < 60]
        self._minute_log[user_id] = minute_calls
        if len(minute_calls) >= self.rpm_limit:
            return False, "Too many requests per minute, please wait"

        daily["count"] += 1
        self._minute_log[user_id].append(now)
        return True, "ok"
