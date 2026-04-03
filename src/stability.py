"""
stability.py
------------
Subperiod comparison and KS-based stability analysis (Q8, Q9).

Public API
----------
build_subperiod_returns(returns)           → dict[period, DataFrame]
compare_moments_across_periods(...)        → pd.DataFrame
risk_evolution_summary(returns_dict)       → pd.DataFrame
ks_stability_report(returns_dict)          → pd.DataFrame
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.config import PERIODS, MIN_OBS_SUBPERIOD
from src.analytics import compute_stock_moments
from src.statistics import ks_test_two_samples, variance_ratio_test
from src.preprocessing import filter_period, get_index_returns, get_stock_returns
from src.utils import logger, safe_skew, safe_kurtosis


# ── Build subperiod return slices ─────────────────────────────────────────────

def build_subperiod_returns(
    returns: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Slice the full returns DataFrame into the three canonical subperiods.

    Returns
    -------
    dict { period_name: returns_DataFrame }
    """
    result = {}
    for period_name in PERIODS:
        try:
            sub = filter_period(returns, period=period_name)
            result[period_name] = sub
            logger.info(
                "Subperiod '%s': %d days × %d tickers",
                period_name, len(sub), sub.shape[1],
            )
        except Exception as exc:
            logger.warning("Could not build subperiod '%s': %s", period_name, exc)
    return result


# ── Cross-period moment comparison ───────────────────────────────────────────

def compare_moments_across_periods(
    returns_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    For each subperiod, compute cross-sectional averages of the six stock-level
    moment statistics (mean of means, mean of variances, etc.).

    Returns
    -------
    pd.DataFrame  index = period_name, columns = [mean, variance, skewness, kurtosis, p1, p99]
    """
    rows = {}
    for period, ret in returns_dict.items():
        stocks = get_stock_returns(ret)
        arr = stocks.values.ravel()
        arr = arr[~np.isnan(arr)]
        if len(arr) < 100:
            continue
        rows[period] = {
            "mean":     float(np.mean(arr)),
            "variance": float(np.var(arr, ddof=1)),
            "skewness": safe_skew(arr),
            "kurtosis": safe_kurtosis(arr),
            "p1":       float(np.percentile(arr, 1)),
            "p99":      float(np.percentile(arr, 99)),
        }
    return pd.DataFrame(rows).T


# ── Risk evolution summary ────────────────────────────────────────────────────

def risk_evolution_summary(
    returns_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize key risk indicators across subperiods for the index and the
    average stock.

    Indicators: annualized volatility, max drawdown of cumulative return,
    mean excess kurtosis, mean skewness, 1% tail (average across stocks).

    Returns
    -------
    pd.DataFrame  one row per period, columns = risk indicators
    """
    rows = []
    for period, ret in returns_dict.items():
        # Index series
        try:
            idx = get_index_returns(ret)
        except KeyError:
            idx = pd.Series(dtype=float)

        # Annualized vol of index
        ann_vol_idx = float(idx.std(ddof=1) * np.sqrt(252)) if len(idx) > 20 else np.nan

        # Max drawdown of index cumulative return
        if len(idx) > 0:
            cum = (idx / 100 + 1).cumprod()
            roll_max = cum.cummax()
            drawdown = (cum - roll_max) / roll_max
            max_dd = float(drawdown.min()) * 100
        else:
            max_dd = np.nan

        # Cross-sectional average volatility of individual stocks
        stocks = get_stock_returns(ret)
        avg_stock_vol = float(stocks.std(ddof=1).mean() * np.sqrt(252))

        # Average stock skewness and kurtosis
        stock_arr = stocks.values.ravel()
        stock_arr = stock_arr[~np.isnan(stock_arr)]
        avg_skew = safe_skew(stock_arr)
        avg_kurt = safe_kurtosis(stock_arr)

        # Avg 1st-percentile across stocks (tail risk)
        p1_avg = float(np.nanpercentile(stocks.values, 1))

        rows.append(
            {
                "Period":               period,
                "Index Ann. Vol (%)":   round(ann_vol_idx, 2) if not np.isnan(ann_vol_idx) else None,
                "Index Max Drawdown (%)": round(max_dd, 2) if not np.isnan(max_dd) else None,
                "Avg Stock Ann. Vol (%)": round(avg_stock_vol, 2),
                "Avg Skewness":         round(avg_skew, 3) if not np.isnan(avg_skew) else None,
                "Avg Exc. Kurtosis":    round(avg_kurt, 3) if not np.isnan(avg_kurt) else None,
                "Avg 1st Pctile (%)":   round(p1_avg, 3),
                "N Trading Days":       len(ret),
                "N Stocks":             stocks.shape[1],
            }
        )
    return pd.DataFrame(rows).set_index("Period")


# ── KS stability report ───────────────────────────────────────────────────────

def ks_stability_report(
    returns_dict: Dict[str, pd.DataFrame],
    metric: str = "returns",
) -> pd.DataFrame:
    """
    Run two-sample KS tests between every pair of subperiods for either
    raw returns or a specified moment metric.

    Parameters
    ----------
    returns_dict : dict { period: DataFrame }
    metric       : 'returns' tests the pooled return distribution;
                   'variance', 'skewness', etc. test the distribution of
                   per-stock moment values across periods.

    Returns
    -------
    pd.DataFrame  one row per period pair, with KS statistic and p-value
    """
    period_names = list(returns_dict.keys())
    rows = []

    for i, p1 in enumerate(period_names):
        for p2 in period_names[i + 1:]:
            ret1 = returns_dict[p1]
            ret2 = returns_dict[p2]

            if metric == "returns":
                # Pool all stock returns for each period
                s1 = get_stock_returns(ret1).values.ravel()
                s2 = get_stock_returns(ret2).values.ravel()
                s1 = s1[~np.isnan(s1)]
                s2 = s2[~np.isnan(s2)]
                # Subsample to avoid memory issues (100k pts each)
                if len(s1) > 100_000:
                    s1 = np.random.choice(s1, 100_000, replace=False)
                if len(s2) > 100_000:
                    s2 = np.random.choice(s2, 100_000, replace=False)
                label = "Pooled stock returns"
            else:
                # Use per-stock moment values
                from src.analytics import compute_stock_moments
                m1 = compute_stock_moments(get_stock_returns(ret1), force_refresh=True)
                m2 = compute_stock_moments(get_stock_returns(ret2), force_refresh=True)
                if metric not in m1.columns:
                    continue
                s1 = m1[metric].dropna().values
                s2 = m2[metric].dropna().values
                label = f"Per-stock {metric}"

            res = ks_test_two_samples(s1, s2)
            rows.append(
                {
                    "Period A":   p1,
                    "Period B":   p2,
                    "Metric":     label,
                    "KS Stat":    round(res["statistic"], 4),
                    "p-value":    round(res["p_value"], 4),
                    "n_A":        res["n_a"],
                    "n_B":        res["n_b"],
                    "Reject H₀ (5%)": res["p_value"] < 0.05,
                }
            )

    return pd.DataFrame(rows)


# ── Index KS test (full report) ───────────────────────────────────────────────

def ks_index_pairwise(
    returns_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    KS tests on ^GSPC index returns between every pair of subperiods.
    Also reports Levene variance test.
    """
    from src.config import INDEX_TICKER
    period_names = list(returns_dict.keys())
    rows = []

    for i, p1 in enumerate(period_names):
        for p2 in period_names[i + 1:]:
            try:
                idx1 = get_index_returns(returns_dict[p1])
                idx2 = get_index_returns(returns_dict[p2])
            except KeyError:
                continue

            ks  = ks_test_two_samples(idx1.values, idx2.values)
            var = variance_ratio_test(idx1, idx2)

            rows.append(
                {
                    "Period A":        p1,
                    "Period B":        p2,
                    "KS Stat":         round(ks["statistic"], 4),
                    "KS p-value":      round(ks["p_value"], 4),
                    "Variance A (%²)": round(var["var_a"], 4),
                    "Variance B (%²)": round(var["var_b"], 4),
                    "Var Ratio B/A":   round(var["ratio"], 3) if var["ratio"] else None,
                    "Levene p-value":  round(var["p_value"], 4),
                    "KS Reject H₀":   ks["p_value"] < 0.05,
                    "Levene Reject H₀": var["p_value"] < 0.05,
                }
            )

    return pd.DataFrame(rows)
