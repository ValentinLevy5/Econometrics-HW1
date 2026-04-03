"""
preprocessing.py
----------------
Price cleaning, percentage log-return computation, and subperiod filtering.

Public API
----------
compute_returns(prices)          → pd.DataFrame   % log returns
clean_returns(returns, min_obs)  → pd.DataFrame   drop low-coverage tickers
filter_period(returns, period)   → pd.DataFrame   subperiod slice
get_index_returns(returns)       → pd.Series       ^GSPC series
get_stock_returns(returns)       → pd.DataFrame    non-index columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    INDEX_TICKER, RETURNS_CACHE,
    MIN_OBS_FULL, MIN_OBS_SUBPERIOD, PERIODS,
)
from src.utils import logger, filter_by_date, drop_low_coverage


# ── Return computation ────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage log returns from adjusted close prices.

    Formula: r_t = 100 × (log P_t − log P_{t-1})

    Any price ≤ 0 is replaced with NaN before taking logs to avoid domain errors.

    Parameters
    ----------
    prices : pd.DataFrame  (dates × tickers), adjusted close prices

    Returns
    -------
    pd.DataFrame  same shape minus one row (first row dropped as NaN)
    """
    # Guard against non-positive prices
    prices = prices.where(prices > 0, other=np.nan)

    log_prices = np.log(prices)
    returns    = 100.0 * log_prices.diff()

    # Drop the very first row (always NaN from diff)
    returns = returns.iloc[1:]

    logger.info("Computed returns: shape=%s", returns.shape)
    return returns


def clean_returns(
    returns: pd.DataFrame,
    min_obs: int = MIN_OBS_FULL,
) -> pd.DataFrame:
    """
    Drop tickers with too few valid observations and forward-fill isolated
    NaN gaps (≤ 2 consecutive missing days, e.g., stale quotes).

    Parameters
    ----------
    returns : pd.DataFrame  % log returns
    min_obs : minimum required non-NaN observations per column

    Returns
    -------
    pd.DataFrame  cleaned returns
    """
    # Drop tickers with insufficient history
    returns = drop_low_coverage(returns, min_obs)

    # Fill very short gaps (e.g., stale-quote artefacts in Yahoo Finance)
    # We deliberately do NOT fill longer gaps to preserve data integrity.
    returns = returns.ffill(limit=2)

    # Extreme outlier guard: winsorize at ±50% daily return
    # (genuine extreme moves are preserved; only data errors beyond ±50% are clipped)
    returns = returns.clip(lower=-50.0, upper=50.0)

    logger.info("Cleaned returns: shape=%s  (min_obs=%d)", returns.shape, min_obs)
    return returns


# ── Subperiod filtering ───────────────────────────────────────────────────────

def filter_period(
    returns: pd.DataFrame,
    period: str = "Full Sample",
) -> pd.DataFrame:
    """
    Slice returns to one of the named subperiods defined in config.PERIODS.

    Also drops tickers that fall below MIN_OBS_SUBPERIOD observations
    within the chosen subperiod.

    Parameters
    ----------
    returns : pd.DataFrame  cleaned % log returns
    period  : one of "Full Sample", "Pre-2020", "Post-2021"

    Returns
    -------
    pd.DataFrame  returns for the chosen subperiod, low-coverage tickers dropped
    """
    if period not in PERIODS:
        raise ValueError(f"Unknown period '{period}'. Choose from {list(PERIODS)}")

    start, end = PERIODS[period]
    sliced = filter_by_date(returns, start, end)

    min_obs = MIN_OBS_FULL if period == "Full Sample" else MIN_OBS_SUBPERIOD
    sliced  = drop_low_coverage(sliced, min_obs)

    logger.info(
        "Period '%s': %d days × %d tickers", period, len(sliced), sliced.shape[1]
    )
    return sliced


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_index_returns(returns: pd.DataFrame) -> pd.Series:
    """Extract the S&P 500 index (^GSPC) return series."""
    if INDEX_TICKER in returns.columns:
        return returns[INDEX_TICKER].dropna()
    raise KeyError(
        f"Index ticker {INDEX_TICKER!r} not found in returns. "
        "Check that prices were downloaded correctly."
    )


def get_stock_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Return all columns except the S&P 500 index ticker."""
    cols = [c for c in returns.columns if c != INDEX_TICKER]
    return returns[cols]


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_returns(returns: pd.DataFrame) -> None:
    """Persist returns to the Parquet cache."""
    returns.to_parquet(RETURNS_CACHE)
    logger.info("Returns saved to %s", RETURNS_CACHE)


def load_cached_returns() -> pd.DataFrame | None:
    """Load returns from the Parquet cache if it exists, else return None."""
    if RETURNS_CACHE.exists():
        df = pd.read_parquet(RETURNS_CACHE)
        df.index = pd.to_datetime(df.index)
        logger.info("Returns loaded from cache: shape=%s", df.shape)
        return df
    return None


# ── Return statistics summary ─────────────────────────────────────────────────

def return_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Quick descriptive summary of the returns DataFrame (useful for Data Pipeline page).

    Columns: n_obs, start_date, end_date, n_stocks, n_missing_pct
    """
    stock_ret = get_stock_returns(returns)
    n_obs     = len(returns)
    n_stocks  = stock_ret.shape[1]
    n_missing = stock_ret.isna().sum().sum()
    total_cells = n_obs * n_stocks
    missing_pct = 100.0 * n_missing / total_cells if total_cells > 0 else 0.0

    # All values cast to str — avoids pyarrow mixed-type serialization errors
    return pd.DataFrame(
        {
            "Metric": [
                "Observations (trading days)",
                "Start date",
                "End date",
                "Number of stocks",
                "Number of series (stocks + index)",
                "Missing values (%)",
            ],
            "Value": [
                str(n_obs),
                str(returns.index.min().date()),
                str(returns.index.max().date()),
                str(n_stocks),
                str(n_stocks + 1),
                f"{missing_pct:.2f}%",
            ],
        }
    )
