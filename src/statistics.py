"""
statistics.py
-------------
Statistical computations: correlations, pairwise correlation matrices,
and KS tests for distributional equality.

Public API
----------
compute_stock_index_corr(stock_returns, index_returns)  → pd.Series
compute_pairwise_corr(returns)                          → pd.DataFrame (square)
ks_test_two_samples(a, b)                               → dict
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr

from src.config import CORR_CACHE, INDEX_TICKER
from src.utils import logger


# ── Stock–index correlations (Q6) ─────────────────────────────────────────────

def compute_stock_index_corr(
    stock_returns: pd.DataFrame,
    index_returns: pd.Series,
) -> pd.Series:
    """
    Compute the Pearson correlation between each stock's return series and
    the S&P 500 index return series.

    Only overlapping non-NaN dates are used for each stock.

    Parameters
    ----------
    stock_returns  : pd.DataFrame  % log returns (dates × stocks)
    index_returns  : pd.Series     % log returns for ^GSPC

    Returns
    -------
    pd.Series  index = ticker, values = correlation coefficient ∈ [−1, 1]
    """
    corrs = {}
    for ticker in stock_returns.columns:
        s = stock_returns[ticker].dropna()
        idx_aligned = index_returns.reindex(s.index).dropna()
        s_aligned   = s.reindex(idx_aligned.index)

        if len(s_aligned) < 30:
            corrs[ticker] = np.nan
            continue

        try:
            r, _ = pearsonr(s_aligned.values, idx_aligned.values)
            corrs[ticker] = float(r)
        except Exception:
            corrs[ticker] = np.nan

    result = pd.Series(corrs, name="corr_to_index")
    result.index.name = "ticker"
    logger.info(
        "Computed stock-index correlations: %d tickers  median=%.3f",
        result.notna().sum(), result.median(),
    )
    return result


# ── Pairwise correlations (Q7) ────────────────────────────────────────────────

def compute_pairwise_corr(
    returns: pd.DataFrame,
    force_refresh: bool = False,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute the full N×N pairwise return correlation matrix.

    For large universes this is an O(N²) operation; the result is cached.

    Parameters
    ----------
    returns       : pd.DataFrame  % log returns (dates × stocks)
    force_refresh : bypass Parquet cache
    method        : 'pearson' or 'spearman'

    Returns
    -------
    pd.DataFrame  shape (N, N), symmetric, diagonal = 1.0
    """
    if not force_refresh and CORR_CACHE.exists():
        corr = pd.read_parquet(CORR_CACHE)
        logger.info("Pairwise correlation matrix loaded from cache: shape=%s", corr.shape)
        return corr

    logger.info("Computing %s pairwise correlations (%d × %d)…",
                method, returns.shape[1], returns.shape[1])

    corr = returns.corr(method=method, min_periods=50)
    corr.to_parquet(CORR_CACHE)
    logger.info("Pairwise correlation matrix saved to cache.")
    return corr


# ── KS test ───────────────────────────────────────────────────────────────────

def ks_test_two_samples(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Two-sample Kolmogorov–Smirnov test.

    Parameters
    ----------
    a, b        : 1-D arrays of observations (NaNs dropped internally)
    alternative : 'two-sided', 'less', or 'greater'

    Returns
    -------
    dict with keys: statistic, p_value, n_a, n_b
    """
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]

    if len(a) < 5 or len(b) < 5:
        return {"statistic": np.nan, "p_value": np.nan, "n_a": len(a), "n_b": len(b)}

    stat, pval = ks_2samp(a, b, alternative=alternative)
    return {
        "statistic": float(stat),
        "p_value":   float(pval),
        "n_a":       int(len(a)),
        "n_b":       int(len(b)),
    }


# ── Batch KS tests for stability analysis (Q9) ───────────────────────────────

def ks_batch_stability(
    returns_full: pd.DataFrame,
    returns_pre:  pd.DataFrame,
    returns_post: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each stock present in all three periods, run KS tests comparing:
      (1) pre-2020 vs. post-2021 return distributions

    Returns
    -------
    pd.DataFrame  columns: ticker, ks_stat, ks_pval, n_pre, n_post
    """
    common = (
        set(returns_full.columns)
        & set(returns_pre.columns)
        & set(returns_post.columns)
    )
    common.discard(INDEX_TICKER)
    common = sorted(common)

    logger.info("Running KS stability tests for %d tickers…", len(common))
    records = []
    for ticker in common:
        a = returns_pre[ticker].dropna().values
        b = returns_post[ticker].dropna().values
        res = ks_test_two_samples(a, b)
        records.append(
            {
                "ticker":  ticker,
                "ks_stat": res["statistic"],
                "ks_pval": res["p_value"],
                "n_pre":   res["n_a"],
                "n_post":  res["n_b"],
            }
        )

    df = pd.DataFrame(records)
    df["reject_H0_5pct"] = df["ks_pval"] < 0.05
    df = df.sort_values("ks_stat", ascending=False).reset_index(drop=True)
    return df


# ── Portfolio-level KS test (index returns) ───────────────────────────────────

def ks_index_stability(
    index_pre: pd.Series,
    index_post: pd.Series,
) -> Dict[str, float]:
    """
    KS test comparing the distribution of ^GSPC daily returns across
    the pre-2020 and post-2021 periods.
    """
    return ks_test_two_samples(
        index_pre.dropna().values,
        index_post.dropna().values,
    )


# ── Variance ratio test (are post-2021 returns more volatile?) ────────────────

def variance_ratio_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
) -> Dict[str, float]:
    """
    Levene-type test of equality of variances between two return series.

    Returns
    -------
    dict: var_a, var_b, ratio (var_b / var_a), f_stat, p_value
    """
    from scipy.stats import levene

    a = returns_a.dropna().values
    b = returns_b.dropna().values

    if len(a) < 10 or len(b) < 10:
        return {"var_a": np.nan, "var_b": np.nan, "ratio": np.nan,
                "f_stat": np.nan, "p_value": np.nan}

    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    stat, pval = levene(a, b)

    return {
        "var_a":   var_a,
        "var_b":   var_b,
        "ratio":   var_b / var_a if var_a > 0 else np.nan,
        "f_stat":  float(stat),
        "p_value": float(pval),
    }
