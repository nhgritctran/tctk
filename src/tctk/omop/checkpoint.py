"""Save / load / replay OMOP mapping results.

Persists the results dict returned by ``Condition2ICD.map()`` (or
``Condition2SNOMED.map()``) as a directory of Parquet files + a JSON
summary.  This avoids pickle's portability and security issues while
keeping file sizes small (Parquet compresses DataFrames well).

Usage::

    from tctk.omop import Condition2ICD, save_results, load_results, print_summary

    results = Condition2ICD().map(conditions, ...)

    # Save checkpoint
    save_results(results, "checkpoints/run_01")

    # Later: reload and inspect
    results = load_results("checkpoints/run_01")
    print_summary(results)          # replay full run log
    results["df_accepted"]          # Polars DataFrame, ready to use
"""

import json
from pathlib import Path

import polars as pl

__all__ = ["save_results", "load_results", "print_summary"]

_META_FILE = "_meta.json"


def save_results(results: dict, path: str | Path) -> Path:
    """Save mapping results to a directory.

    Parameters
    ----------
    results : dict
        The dict returned by ``Condition2ICD.map()`` or
        ``Condition2SNOMED.map()``.
    path : str or Path
        Directory to create.  Parent directories are created as needed.

    Returns
    -------
    Path
        The directory path (for chaining / display).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    meta: dict = {}

    for key, value in results.items():
        if isinstance(value, pl.DataFrame):
            value.write_parquet(path / f"{key}.parquet")
            meta[key] = {"type": "dataframe", "file": f"{key}.parquet",
                         "rows": len(value), "cols": value.width}
        elif key == "_run_log":
            # Store the captured print log as plain text
            (path / "_run_log.txt").write_text(value, encoding="utf-8")
            meta[key] = {"type": "text", "file": "_run_log.txt"}
        else:
            # Scalars / small objects — store inline in JSON
            try:
                json.dumps(value)  # check serializable
                meta[key] = {"type": "value", "value": value}
            except (TypeError, ValueError):
                # Skip non-serializable values (rare)
                pass

    (path / _META_FILE).write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    n_parquet = sum(1 for v in meta.values() if v.get("type") == "dataframe")
    print(f"Saved {n_parquet} DataFrames to {path}/")
    return path


def load_results(path: str | Path) -> dict:
    """Load mapping results from a checkpoint directory.

    Parameters
    ----------
    path : str or Path
        Directory previously created by :func:`save_results`.

    Returns
    -------
    dict
        Same structure as the original ``map()`` return value.
    """
    path = Path(path)
    meta_path = path / _META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {path} (missing {_META_FILE})"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    results: dict = {}

    for key, info in meta.items():
        if info["type"] == "dataframe":
            results[key] = pl.read_parquet(path / info["file"])
        elif info["type"] == "text":
            results[key] = (path / info["file"]).read_text(encoding="utf-8")
        elif info["type"] == "value":
            results[key] = info["value"]

    n_parquet = sum(1 for v in meta.values() if v.get("type") == "dataframe")
    print(f"Loaded {n_parquet} DataFrames from {path}/")
    return results


def print_summary(results: dict) -> None:
    """Replay the captured run log from a results dict.

    Works with both live results (from ``map()``) and loaded checkpoints
    (from ``load_results()``).

    Parameters
    ----------
    results : dict
        The results dict (must contain ``_run_log`` key).
    """
    log = results.get("_run_log")
    if log is None:
        print("No run log found in results. "
              "Run map() first or load a checkpoint that has one.")
        return
    print(log, end="")
