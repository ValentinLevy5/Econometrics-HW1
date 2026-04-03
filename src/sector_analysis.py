"""
sector_analysis.py
------------------
Sector-level correlation analysis (Q7).

Assigns each stock a GICS sector (from yfinance metadata), extracts within-
and between-sector pairwise correlations, and tests whether they differ
significantly.

Public API
----------
build_sector_map(sector_info)              → dict[ticker, sector]
extract_within_between_corr(corr, smap)   → (pd.Series, pd.Series)
test_sector_correlation_diff(within, btw) → dict
sector_avg_corr(corr, smap)               → pd.DataFrame
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from src.config import SECTOR_COLORS
from src.utils import logger


# ── Sector map helper ─────────────────────────────────────────────────────────

def build_sector_map(sector_info: pd.DataFrame) -> Dict[str, str]:
    """
    Convert sector_info DataFrame (index = ticker) to a plain dict
    { ticker: sector_name }.

    Unknown or missing sectors are replaced with "Unknown".
    """
    smap: Dict[str, str] = {}
    for ticker, row in sector_info.iterrows():
        sector = str(row.get("sector", "Unknown") or "Unknown")
        smap[str(ticker)] = sector
    return smap


# ── Within / between sector correlations ─────────────────────────────────────

def extract_within_between_corr(
    corr: pd.DataFrame,
    sector_map: Dict[str, str],
) -> Tuple[pd.Series, pd.Series]:
    """
    From the full N×N pairwise correlation matrix, extract:
      - within_corr  : upper-triangle entries where both tickers share a sector
      - between_corr : upper-triangle entries where tickers are in different sectors

    Diagonal entries and "Unknown" sectors are excluded.

    Parameters
    ----------
    corr       : symmetric correlation matrix (tickers × tickers)
    sector_map : dict { ticker: sector }

    Returns
    -------
    (within_corr, between_corr)  each a pd.Series of correlation values
    """
    tickers = [t for t in corr.index if t in sector_map and sector_map[t] != "Unknown"]
    corr_sub = corr.loc[tickers, tickers]

    within_vals:  list[float] = []
    between_vals: list[float] = []

    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_sub.iloc[i, j]
            if np.isnan(val):
                continue
            si = sector_map[tickers[i]]
            sj = sector_map[tickers[j]]
            if si == sj:
                within_vals.append(val)
            else:
                between_vals.append(val)

    logger.info(
        "Extracted %d within-sector and %d between-sector correlation pairs.",
        len(within_vals), len(between_vals),
    )
    return (
        pd.Series(within_vals,  name="within_sector"),
        pd.Series(between_vals, name="between_sector"),
    )


# ── Statistical test ──────────────────────────────────────────────────────────

def test_sector_correlation_diff(
    within_corr: pd.Series,
    between_corr: pd.Series,
) -> Dict[str, object]:
    """
    Test whether within-sector correlations are significantly higher than
    between-sector correlations using:
      1. Welch two-sample t-test  (H0: means equal, H1: within > between)
      2. Mann–Whitney U test      (H0: distributions equal, H1: within > between)

    Returns
    -------
    dict with descriptive stats and test results:
        mean_within, mean_between, median_within, median_between,
        welch_t, welch_p, mw_stat, mw_p, conclusion
    """
    w = within_corr.dropna().values
    b = between_corr.dropna().values

    # Welch t-test (one-sided: within > between)
    t_stat, t_pval_two = ttest_ind(w, b, equal_var=False, alternative="greater")

    # Mann–Whitney U (one-sided: within > between)
    mw_stat, mw_pval = mannwhitneyu(w, b, alternative="greater", use_continuity=True)

    alpha = 0.05
    welch_sig = t_pval_two < alpha
    mw_sig    = mw_pval    < alpha

    if welch_sig and mw_sig:
        conclusion = (
            "Both tests confirm that within-sector correlations are "
            "significantly higher than between-sector correlations at the 5% level."
        )
    elif welch_sig or mw_sig:
        conclusion = (
            "Mixed evidence: one test rejects H₀ at 5%, the other does not. "
            "Within-sector correlations appear moderately higher."
        )
    else:
        conclusion = (
            "Neither test rejects H₀ at 5%. No statistically significant "
            "difference between within- and between-sector correlations detected."
        )

    return {
        "mean_within":   float(np.mean(w)),
        "mean_between":  float(np.mean(b)),
        "median_within": float(np.median(w)),
        "median_between":float(np.median(b)),
        "n_within":      int(len(w)),
        "n_between":     int(len(b)),
        "welch_t":       float(t_stat),
        "welch_p":       float(t_pval_two),
        "welch_sig":     bool(welch_sig),
        "mw_stat":       float(mw_stat),
        "mw_p":          float(mw_pval),
        "mw_sig":        bool(mw_sig),
        "conclusion":    conclusion,
    }


# ── Sector-level summary statistics ──────────────────────────────────────────

def sector_avg_corr(
    corr: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute average within-sector correlation for each GICS sector.

    Returns
    -------
    pd.DataFrame  columns: sector, n_stocks, mean_within_corr, median_within_corr
    """
    sectors = sorted(set(v for v in sector_map.values() if v != "Unknown"))
    rows = []
    for sec in sectors:
        members = [t for t, s in sector_map.items() if s == sec and t in corr.index]
        if len(members) < 2:
            continue
        sub = corr.loc[members, members]
        n   = len(members)
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                v = sub.iloc[i, j]
                if not np.isnan(v):
                    vals.append(v)
        if not vals:
            continue
        rows.append(
            {
                "sector":           sec,
                "n_stocks":         n,
                "mean_within_corr": float(np.mean(vals)),
                "median_within_corr": float(np.median(vals)),
                "color":            SECTOR_COLORS.get(sec, "#8b949e"),
            }
        )
    df = pd.DataFrame(rows).sort_values("mean_within_corr", ascending=False)
    return df


# ── Sector-level moment averages (for display) ────────────────────────────────

def sector_moment_summary(
    moments: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute mean of each moment metric grouped by sector.

    Parameters
    ----------
    moments    : stock-level moments DataFrame (index = ticker)
    sector_map : dict { ticker: sector }

    Returns
    -------
    pd.DataFrame  index = sector, columns = moment metrics
    """
    df = moments.copy()
    df["sector"] = df.index.map(lambda t: sector_map.get(t, "Unknown"))
    grouped = df.groupby("sector").mean(numeric_only=True)
    grouped["n_stocks"] = df.groupby("sector").size()
    return grouped.sort_values("mean", ascending=False)
