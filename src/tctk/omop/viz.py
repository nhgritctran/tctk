"""Visualizations for the OMOP condition-mapping pipeline.

- ``plot_condition_coverage`` — horizontal stacked bar chart per condition

Accepts either the ``results`` dict from ``mapper.map()`` **or** a file
path to the exported ``_full.tsv`` / ``.csv`` (the flat review table).

matplotlib is imported lazily so the package stays importable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

__all__ = ["plot_condition_coverage"]


# ── colours (Okabe-Ito colorblind-safe palette) ──────────────────────
_C_EXACT = "#0072B2"       # blue
_C_AI_ACCEPT = "#56B4E9"   # sky blue
_C_FUZZY_PASS = "#CC79A7"  # reddish purple
_C_REJECT = "#D55E00"      # vermillion
_C_HUMAN = "#E69F00"       # orange
_C_NOMATCH = "#999999"     # gray


# =====================================================================
# plot_condition_coverage
# =====================================================================

def plot_condition_coverage(
    source: Union[dict, str, Path],
    figsize=(12, None),
    title=None,
    top_n=None,
):
    """Horizontal stacked bar chart of per-condition mapping coverage.

    Args:
        source (dict | str | Path): Either the ``results`` dict returned by
            ``mapper.map()``, or a file path to the exported full review
            table (TSV / CSV).
        figsize (tuple): ``(width, height)``.  Height auto-calculated if *None*.
        title (str, optional): Figure title.  Defaults to "Condition Coverage Summary".
        top_n (int, optional): Show only the top *n* conditions by total accepted codes.

    Returns:
        matplotlib.figure.Figure:
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df_review = _load_review_df(source)

    # ── build per-condition breakdown ────────────────────────────────
    rows = _build_condition_breakdown(df_review)

    if not rows:
        fig, ax = plt.subplots(figsize=(figsize[0], 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=14, color="gray")
        ax.axis("off")
        return fig

    # sort by total accepted descending
    rows.sort(key=lambda r: r["accepted"], reverse=True)

    if top_n is not None:
        rows = rows[:top_n]

    conditions = [r["condition"] for r in rows]
    n = len(conditions)

    # auto height: per-bar height + fixed margin for title/xlabel/legend
    bar_inch = 0.2
    margin_inch = 1.5
    height = figsize[1] if figsize[1] is not None else n * bar_inch + margin_inch
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    y_pos = list(range(n))

    # draw bars left-to-right: exact, fuzzy_accepted, fuzzy_pass, human, reject, nomatch
    categories = [
        ("exact",         _C_EXACT,      "Exact"),
        ("fuzzy_accepted",_C_AI_ACCEPT,  "Fuzzy (AI accept)"),
        ("fuzzy_pass",    _C_FUZZY_PASS, "Fuzzy (unreviewed)"),
        ("human",         _C_HUMAN,      "Human review"),
        ("reject",        _C_REJECT,     "Rejected"),
        ("nomatch",       _C_NOMATCH,    "No match"),
    ]

    lefts = [0.0] * n
    for key, colour, _label in categories:
        widths = [r[key] for r in rows]
        ax.barh(y_pos, widths, left=lefts, height=0.7,
                color=colour, edgecolor="white", linewidth=0.5)
        lefts = [l + w for l, w in zip(lefts, widths)]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5 + 0.4, -0.5 - 0.4)  # breathing room bottom/top
    ax.set_xlabel("Number of ICD code matches", fontsize=10)
    ax.set_title(title or "Condition Coverage Summary",
                 fontsize=13, fontweight="bold")

    # remove spines except bottom x-axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)  # hide y-axis tick marks

    # legend
    patches = [mpatches.Patch(color=c, label=l) for _, c, l in categories]
    ax.legend(handles=patches, loc="lower right", fontsize=7,
              framealpha=0.9)

    fig.tight_layout()
    return fig


# =====================================================================
# helpers (private)
# =====================================================================

def _load_review_df(source: Union[dict, str, Path]):
    """Return a ``df_review`` polars DataFrame from *source*.

    *source* may be:
    - A ``results`` dict (uses ``results["df_review"]``).
    - A file path (str / Path) to a TSV or CSV file.
    - A ``polars.DataFrame`` directly.
    """
    import polars as pl

    if isinstance(source, pl.DataFrame):
        return source

    if isinstance(source, dict):
        return source["df_review"]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    return pl.read_csv(path, separator=sep, infer_schema_length=0)


def _build_condition_breakdown(df_review) -> list[dict]:
    """Build per-condition match-type breakdown from *df_review*."""
    import polars as pl

    cols = df_review.columns
    conditions = df_review["condition_name"].unique().sort().to_list()

    # helper: check for empty-string-as-null (CSV round-trip)
    def _eq(col_name, value):
        return pl.col(col_name) == value

    def _is_null_or_empty(col_name):
        return pl.col(col_name).is_null() | (pl.col(col_name) == "")

    def _has_icd():
        return pl.col("icd_code").is_not_null() & (pl.col("icd_code") != "")

    rows = []
    for cond in conditions:
        df_c = df_review.filter(pl.col("condition_name") == cond)

        # exact
        if "match_type" in cols:
            n_exact = len(df_c.filter(_eq("match_type", "exact")))
        else:
            n_exact = 0

        # fuzzy accepted by AI
        n_fuzzy_accepted = 0
        if "ai_verdict" in cols and "match_type" in cols:
            n_fuzzy_accepted = len(df_c.filter(
                _eq("match_type", "fuzzy") & _eq("ai_verdict", "accept")
            ))

        # unreviewed fuzzy pass-through
        if "match_type" in cols:
            fuzzy_pass_filter = _has_icd() & _eq("match_type", "fuzzy")
        else:
            fuzzy_pass_filter = _has_icd()
        if "ai_verdict" in cols:
            fuzzy_pass_filter = fuzzy_pass_filter & _is_null_or_empty("ai_verdict")
        n_fuzzy_pass = len(df_c.filter(fuzzy_pass_filter))

        # human review
        n_human = 0
        if "ai_verdict" in cols:
            n_human = len(df_c.filter(_eq("ai_verdict", "human review")))

        # rejected
        n_reject = 0
        if "ai_verdict" in cols:
            n_reject = len(df_c.filter(_eq("ai_verdict", "reject")))

        # no match
        n_nomatch = 0
        if "ai_verdict" in cols:
            n_nomatch = len(df_c.filter(_eq("ai_verdict", "no match")))
        else:
            n_nomatch = len(df_c.filter(~_has_icd()))

        accepted = n_exact + n_fuzzy_accepted + n_fuzzy_pass

        rows.append({
            "condition": cond,
            "exact": n_exact,
            "fuzzy_accepted": n_fuzzy_accepted,
            "fuzzy_pass": n_fuzzy_pass,
            "human": n_human,
            "reject": n_reject,
            "nomatch": n_nomatch,
            "accepted": accepted,
        })

    return rows
