"""
Internal utilities shared across tctk modules.

Not intended for direct import by users.
"""

import json
import os
import re
import time
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

# Model name substrings to exclude (not suitable for text-only tasks)
# Note: "preview" intentionally NOT excluded — newer models (e.g. gemini-3.1-pro)
# may only be available as preview initially. "image" and "vision" already catch
# non-text models like gemini-3.1-flash-image-preview.
_MODEL_EXCLUSIONS = ["image", "vision", "embedding", "aqa", "bison"]


# -------------------------------------------------------------------
# SQL helpers
# -------------------------------------------------------------------

# DuckDB / standard SQL style
def sql_escape(s: str) -> str:
    """Escape single quotes for DuckDB SQL string literals."""
    return s.replace("'", "''")


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
            "  Condition2SNOMED(gemini_api_key='your-key')\n\n"
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

    Excludes image, vision, preview, embedding, and other
    non-text-generation models.

    Examples:
        "gemini-2.5-pro"             → "pro"
        "gemini-2.5-flash"           → "flash"
        "gemini-2.0-flash-lite"      → "flash-lite"
        "gemini-3.1-flash-image-preview" → None (excluded)
    """
    name = model_name.lower()

    # Exclude non-text models
    if any(excl in name for excl in _MODEL_EXCLUSIONS):
        return None

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
    min_version: float = 3.0,
) -> str:
    """Query Gemini API and select the best available text model.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    ai_tier : str, optional
        Preferred tier: "pro", "flash", or "flash-lite".
        Default None → picks "flash" tier (cost-effective default),
        then best version within that tier.
    min_version : float
        Minimum model version. Default 3.0 (prefer Gemini 3.x+).
        Set to 2.5 to allow older models (e.g. gemini-2.5-flash).

    Returns
    -------
    str
        Full model name (e.g., "gemini-3.0-flash").

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

    # Filter to text-generation models only
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
            "No suitable Gemini text models found for this API key. "
            "Ensure the Generative Language API is enabled."
        )

    # Selection priority:
    #   1. Preferred tier + min_version  (e.g. pro >= 3.0)
    #   2. Any tier    + min_version     (e.g. flash >= 3.0)
    #   3. Preferred tier + any version  (e.g. pro 2.5)
    #   4. All candidates                (fallback)
    ai_tier = ai_tier.lower().strip()

    tier_version = [
        c for c in candidates
        if c["tier"] == ai_tier and c["version"] >= min_version
    ]
    if tier_version:
        candidates = tier_version
    else:
        any_version = [c for c in candidates if c["version"] >= min_version]
        if any_version:
            tiers_at_version = sorted(set(c["tier"] for c in any_version))
            print(
                f"  Warning: no '{ai_tier}' models >= {min_version}. "
                f"Using best available tier at >= {min_version} "
                f"({', '.join(tiers_at_version)})."
            )
            candidates = any_version
        else:
            tier_any = [c for c in candidates if c["tier"] == ai_tier]
            if tier_any:
                best_v = max(c["version"] for c in tier_any)
                print(
                    f"  Warning: no models >= {min_version}. "
                    f"Best '{ai_tier}': {best_v}."
                )
                candidates = tier_any
            else:
                available_tiers = sorted(set(c["tier"] for c in candidates))
                print(
                    f"  Warning: tier '{ai_tier}' not available. "
                    f"Available: {', '.join(available_tiers)}."
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
    max_output_tokens: int = 65536,
    timeout: int = 300,
    max_retries: int = 3,
    response_schema: Optional[dict] = None,
) -> str:
    """Call Gemini API directly via REST with automatic retry on rate limits.

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
    max_retries : int
        Maximum retries on 429 rate limit errors. Default 3.

    Returns
    -------
    str
        Model response text.

    Raises
    ------
    RuntimeError
        If the API call fails after all retries.
    """
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
    }
    if response_schema:
        generation_config["responseSchema"] = response_schema

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)

            # Handle rate limiting (429) and server overload (503) with retry
            if resp.status_code in (429, 503) and attempt < max_retries:
                wait = 40  # default wait
                error_msg = ""
                quota_info = ""
                try:
                    error_body = resp.json().get("error", {})
                    error_msg = error_body.get("message", "")
                    details = error_body.get("details", [])
                    for d in details:
                        if d.get("@type", "").endswith("RetryInfo"):
                            delay_str = d.get("retryDelay", "40s")
                            wait = int(float(delay_str.rstrip("s"))) + 2
                        if d.get("@type", "").endswith("QuotaFailure"):
                            violations = d.get("violations", [])
                            parts = [f"{v.get('subject', '')}: {v.get('description', '')}" for v in violations]
                            quota_info = "; ".join(parts)
                except Exception:
                    pass
                label = "Rate limited" if resp.status_code == 429 else "Server overloaded"
                print(f"    {label} (HTTP {resp.status_code}): {error_msg}")
                if quota_info:
                    print(f"    Quota detail: {quota_info}")
                print(f"    Waiting {wait}s (retry {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            candidate = data["candidates"][0]
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                raise RuntimeError(
                    f"Gemini response truncated (MAX_TOKENS). "
                    f"Increase max_output_tokens or reduce batch_size."
                )
            return candidate["content"]["parts"][0]["text"]

        except requests.exceptions.HTTPError as e:
            if resp.status_code in (429, 503):
                raise RuntimeError(
                    f"Gemini API (HTTP {resp.status_code}) failed after {max_retries} retries. "
                    f"Wait a few minutes and try again."
                ) from e
            raise RuntimeError(
                f"Gemini API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected Gemini API response format: {e}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini API request failed: {e}") from e

    raise RuntimeError(
        f"Gemini API failed after {max_retries} retries"
    )
