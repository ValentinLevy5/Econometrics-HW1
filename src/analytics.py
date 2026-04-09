"""
analytics.py
------------
Core statistical computations:
  - Stock-level moment estimation (Q2)
  - 30-day rolling moments of the index (Q3)
  - Daily cross-sectional moments and KDE surface (Q4)
  - Kernel density estimation helpers

Public API
----------
compute_stock_moments(returns)              → pd.DataFrame (6 metrics × N stocks)
compute_rolling_moments(series, window=30)  → pd.DataFrame (4 moments × T dates)
compute_cross_sectional_moments(returns)    → pd.DataFrame (3 moments × T dates)
build_kde_surface(returns, stride)          → (dates, grid, Z_array)
estimate_kde(data, n_points, x_range)       → (x_grid, density)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis as sp_kurtosis, skew as sp_skew

from src.config import (
    ROLLING_WINDOW, KDE_GRID_POINTS, XSECTION_STRIDE,
    MOMENTS_CACHE, XSEC_MOMENTS_CACHE,
    MOMENT_COLS,
)
from src.utils import logger, safe_skew, safe_kurtosis, safe_percentile


# ── KDE helper ────────────────────────────────────────────────────────────────

def estimate_kde(
    data: np.ndarray,
    n_points: int = KDE_GRID_POINTS,
    x_range: Optional[Tuple[float, float]] = None,
    bandwidth: str | float = "silverman",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a Gaussian KDE to *data* and evaluate it on a uniform grid.

    Parameters
    ----------
    data      : 1-D array of observations (NaNs are dropped)
    n_points  : number of evaluation points on the x-grid
    x_range   : (lo, hi) tuple; if None, use data ± 3 std
    bandwidth : scalar or 'silverman' / 'scott'

    Returns
    -------
    (x_grid, density) both shape (n_points,)
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 5:
        x = np.linspace(-1, 1, n_points)
        return x, np.zeros(n_points)

    if x_range is None:
        mu, sig = arr.mean(), arr.std()
        lo = mu - 4 * sig
        hi = mu + 4 * sig
    else:
        lo, hi = x_range

    x_grid = np.linspace(lo, hi, n_points)

    try:
        kde = gaussian_kde(arr, bw_method=bandwidth)
        density = kde(x_grid)
    except Exception:
        density = np.zeros(n_points)

    return x_grid, density


# ── Stock-level moments (Q2) ──────────────────────────────────────────────────

def compute_stock_moments(
    returns: pd.DataFrame,
    force_refresh: bool = False,
    save_cache: bool = True,
) -> pd.DataFrame:
    """
    Compute six descriptive statistics for each stock's full return history.

    Statistics
    ----------
    mean      : sample mean (%)
    variance  : sample variance (%²)  [ddof=1]
    skewness  : Fisher's skewness (bias-corrected)
    kurtosis  : excess kurtosis (Fisher, so Gaussian = 0, bias-corrected)
    p1        : 1st percentile (%)
    p99       : 99th percentile (%)

    Parameters
    ----------
    returns       : pd.DataFrame  % log returns  (dates × tickers)
    force_refresh : bypass Parquet cache
    save_cache    : whether to persist result to the Parquet cache (set False
                    for subperiod computations to avoid corrupting the
                    full-sample cache file)

    Returns
    -------
    pd.DataFrame  shape (N_stocks, 6)  index = ticker
    """
    if not force_refresh and MOMENTS_CACHE.exists():
        df = pd.read_parquet(MOMENTS_CACHE)
        logger.info("Stock moments loaded from cache: shape=%s", df.shape)
        return df

    logger.info("Computing stock-level moments for %d tickers…", returns.shape[1])

    records = {}
    for ticker in returns.columns:
        arr = returns[ticker].dropna().values
        if len(arr) < 30:
            continue
        records[ticker] = {
            "mean":     float(np.mean(arr)),
            "variance": float(np.var(arr, ddof=1)),
            "skewness": safe_skew(arr),
            "kurtosis": safe_kurtosis(arr),
            "p1":       safe_percentile(arr, 1),
            "p99":      safe_percentile(arr, 99),
        }

    df = pd.DataFrame.from_dict(records, orient="index")[MOMENT_COLS]
    df.index.name = "ticker"

    if save_cache:
        df.to_parquet(MOMENTS_CACHE)
        logger.info("Stock moments saved: shape=%s", df.shape)
    else:
        logger.info("Stock moments computed (not cached): shape=%s", df.shape)

    return df


# ── Rolling index moments (Q3) ────────────────────────────────────────────────

def compute_rolling_moments(
    series: pd.Series,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling descriptive statistics for a return series using a
    *window*-day expanding window (min_periods = window // 2).

    Statistics: mean, variance, skewness, excess kurtosis.

    Parameters
    ----------
    series : pd.Series  % log returns, indexed by date
    window : rolling window size in trading days (default 30)

    Returns
    -------
    pd.DataFrame  columns = [mean, variance, skewness, kurtosis]
                  index   = DatetimeIndex (same as input, NaN before min_periods)
    """
    min_periods = max(window // 2, 10)

    roll = series.rolling(window=window, min_periods=min_periods)

    df = pd.DataFrame(index=series.index)
    df["mean"]     = roll.mean()
    df["variance"] = roll.var(ddof=1)
    df["skewness"] = roll.apply(
        lambda x: safe_skew(x), raw=True
    )
    df["kurtosis"] = roll.apply(
        lambda x: safe_kurtosis(x), raw=True
    )

    logger.info("Rolling moments computed: window=%d  shape=%s", window, df.shape)
    return df


# ── Cross-sectional moments (Q4) ─────────────────────────────────────────────

def compute_cross_sectional_moments(
    returns: pd.DataFrame,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    For each trading day, compute moments of the cross-section of stock
    returns (i.e., across all stocks on that day).

    Statistics: variance, skewness, excess kurtosis.
    (Mean is also computed but usually near zero by construction.)

    Parameters
    ----------
    returns       : pd.DataFrame  % log returns  (dates × stocks)
    force_refresh : bypass Parquet cache

    Returns
    -------
    pd.DataFrame  columns = [n_stocks, mean, variance, skewness, kurtosis]
                  index   = DatetimeIndex
    """
    if not force_refresh and XSEC_MOMENTS_CACHE.exists():
        df = pd.read_parquet(XSEC_MOMENTS_CACHE)
        df.index = pd.to_datetime(df.index)
        logger.info("Cross-sectional moments loaded from cache: shape=%s", df.shape)
        return df

    logger.info("Computing cross-sectional moments for %d dates…", len(returns))

    rows = []
    for date, row in returns.iterrows():
        arr = row.dropna().values
        n   = len(arr)
        if n < 20:
            continue
        rows.append(
            {
                "date":     date,
                "n_stocks": n,
                "mean":     float(np.mean(arr)),
                "variance": float(np.var(arr, ddof=1)),
                "skewness": safe_skew(arr),
                "kurtosis": safe_kurtosis(arr),
            }
        )

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)

    df.to_parquet(XSEC_MOMENTS_CACHE)
    logger.info("Cross-sectional moments saved: shape=%s", df.shape)
    return df


# ── 3-D KDE density surface (Q4) ─────────────────────────────────────────────

def build_kde_surface(
    returns: pd.DataFrame,
    stride: int = XSECTION_STRIDE,
    n_points: int = KDE_GRID_POINTS,
    x_range: Optional[Tuple[float, float]] = None,
    percentile_clip: float = 0.5,
) -> Tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
    """
    Build a 2-D density surface  Z[i, j] = KDE(cross-section at date[i], x_grid[j])
    for use in the 3-D surface plot.

    Every *stride*-th trading day is sampled to keep computation manageable.

    Parameters
    ----------
    returns          : pd.DataFrame  % log returns (dates × stocks)
    stride           : sample every N trading days
    n_points         : KDE grid resolution
    x_range          : fixed (lo, hi) range for all KDE evaluations;
                       if None, determined from global percentiles
    percentile_clip  : drop top/bottom *percentile_clip* % of return grid
                       (removes extreme daily cross-sections that distort the surface)

    Returns
    -------
    dates  : list[pd.Timestamp]   length = T
    x_grid : np.ndarray           shape (n_points,)
    Z      : np.ndarray           shape (T, n_points)  — density surface
    """
    sampled = returns.iloc[::stride]

    if x_range is None:
        # Use global 0.5th–99.5th percentile of ALL returns for a stable grid
        flat = returns.values.ravel()
        flat = flat[~np.isnan(flat)]
        lo = np.percentile(flat, percentile_clip)
        hi = np.percentile(flat, 100 - percentile_clip)
        x_range = (lo, hi)

    x_grid  = np.linspace(x_range[0], x_range[1], n_points)
    dates   = []
    density_rows: list[np.ndarray] = []

    for date, row in sampled.iterrows():
        arr = row.dropna().values
        if len(arr) < 20:
            continue
        try:
            kde = gaussian_kde(arr, bw_method="silverman")
            d   = kde(x_grid)
        except Exception:
            d = np.zeros(n_points)

        dates.append(date)
        density_rows.append(d)

    Z = np.array(density_rows)
    logger.info(
        "KDE surface built: %d time points × %d grid points  (stride=%d)",
        len(dates), n_points, stride,
    )
    return dates, x_grid, Z


# ── KDE distributions of the 6 moment statistics ─────────────────────────────

def compute_moments_kde(
    moments: pd.DataFrame,
    n_points: int = KDE_GRID_POINTS,
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate a KDE for each of the six stock-level moment columns.

    Returns a dict: { col_name: (x_grid, density) }
    """
    result = {}
    for col in MOMENT_COLS:
        if col not in moments.columns:
            continue
        arr = moments[col].dropna().values
        x, d = estimate_kde(arr, n_points=n_points)
        result[col] = (x, d)
    return result


# ── Multi-period moments comparison ──────────────────────────────────────────

def compute_period_moments(
    returns: pd.DataFrame,
    periods: dict[str, tuple],
) -> dict[str, pd.DataFrame]:
    """
    Compute stock-level moments for multiple subperiods.

    Parameters
    ----------
    returns : pd.DataFrame  full % log returns
    periods : dict mapping period name → (start, end) date strings

    Returns
    -------
    dict { period_name: moments_DataFrame }
    """
    from src.preprocessing import filter_period
    result = {}
    for name in periods:
        try:
            sub = filter_period(returns, period=name)
            mom = compute_stock_moments(sub, force_refresh=True)
            result[name] = mom
        except Exception as exc:
            logger.warning("Could not compute moments for period '%s': %s", name, exc)
    return result
