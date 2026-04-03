"""
utils.py
--------
Shared helper utilities used across the project.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, Any

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finec")


# ── Retry decorator ───────────────────────────────────────────────────────────
def retry(max_attempts: int = 3, wait: float = 2.0, backoff: float = 2.0):
    """Exponential-backoff retry decorator for flaky network calls."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            delay = wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                        attempt, max_attempts, fn.__name__, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


# ── Ticker utilities ──────────────────────────────────────────────────────────
def normalize_ticker(ticker: str) -> str:
    """
    Convert Wikipedia-style tickers to yfinance-compatible format.

    Wikipedia lists BRK.B and BF.B with dots; yfinance expects BRK-B and BF-B.
    Also strips whitespace.
    """
    return ticker.strip().replace(".", "-")


def denormalize_ticker(ticker: str) -> str:
    """
    Convert yfinance tickers back to a display-friendly form (dots restored).
    """
    return ticker.replace("-", ".")


# ── Date helpers ─────────────────────────────────────────────────────────────
def parse_date_bound(bound: str | None) -> pd.Timestamp | None:
    """Return pd.Timestamp or None (meaning 'today')."""
    if bound is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(bound)


def filter_by_date(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """
    Slice a DataFrame with a DatetimeIndex to [start, end] (both inclusive).
    None values are treated as unbounded.
    """
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


# ── Statistical helpers ───────────────────────────────────────────────────────
def safe_skew(x: np.ndarray) -> float:
    """Fisher skewness; returns NaN if fewer than 3 valid observations."""
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    from scipy.stats import skew
    return float(skew(x, bias=False))


def safe_kurtosis(x: np.ndarray) -> float:
    """
    Excess kurtosis (Fisher definition, so normal = 0); returns NaN if
    fewer than 4 valid observations.
    """
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return np.nan
    from scipy.stats import kurtosis
    return float(kurtosis(x, fisher=True, bias=False))


def safe_percentile(x: np.ndarray, q: float) -> float:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, q))


# ── DataFrame helpers ─────────────────────────────────────────────────────────
def drop_low_coverage(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Drop columns that have fewer than *min_obs* non-NaN values."""
    return df.loc[:, df.notna().sum() >= min_obs]


def winsorize_series(s: pd.Series, lower: float = 0.001, upper: float = 0.999) -> pd.Series:
    """Winsorize a Series at the given quantile bounds (not used in returns, only for display)."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def pct_change_to_log_return(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage log returns: 100 * diff(log(P)).

    This is the canonical formula used throughout the project.
    """
    return 100.0 * np.log(prices).diff().iloc[1:]


# ── Display helpers ───────────────────────────────────────────────────────────
def fmt_pct(val: float, decimals: int = 2) -> str:
    return f"{val:+.{decimals}f}%"


def fmt_float(val: float, decimals: int = 3) -> str:
    if np.isnan(val):
        return "—"
    return f"{val:.{decimals}f}"


def color_return(val: float) -> str:
    """Return a CSS color string for a return value."""
    from src.config import THEME
    if val > 0:
        return THEME["green_bright"]
    elif val < 0:
        return THEME["red_bright"]
    return THEME["text_dim"]


# ── Streamlit helpers ─────────────────────────────────────────────────────────
def metric_card_html(label: str, value: str, delta: str = "", color: str = "#388bfd") -> str:
    """Generate a styled HTML metric card string for st.markdown()."""
    delta_html = f'<div style="font-size:0.8rem;color:{color};margin-top:2px">{delta}</div>' if delta else ""
    return f"""
    <div style="background:#1c2128;border:1px solid #30363d;border-radius:8px;
                padding:14px 18px;margin:4px 0;">
        <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
                    letter-spacing:0.06em;">{label}</div>
        <div style="font-size:1.5rem;font-weight:700;color:#e6edf3;margin-top:4px">{value}</div>
        {delta_html}
    </div>"""


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<p style="color:#8b949e;font-size:0.9rem;margin:0 0 16px 0">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin:24px 0 8px 0;border-left:3px solid #388bfd;padding-left:12px">
        <h3 style="color:#e6edf3;margin:0 0 4px 0">{title}</h3>
        {sub}
    </div>"""


def interpretation_box(text: str) -> str:
    """Render a styled interpretation / commentary block."""
    return f"""
    <div style="background:#161b22;border:1px solid #30363d;border-left:3px solid #2ea043;
                border-radius:6px;padding:14px 18px;margin:12px 0;
                font-size:0.9rem;color:#c9d1d9;line-height:1.6">
        <strong style="color:#39d353">📊 Interpretation</strong><br>{text}
    </div>"""
