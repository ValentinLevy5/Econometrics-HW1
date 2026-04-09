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
    sub = (
        f'<p style="color:#8b949e;font-size:0.88rem;margin:5px 0 0 0;'
        f'font-weight:400;letter-spacing:0.01em">{subtitle}</p>'
        if subtitle else ""
    )
    return f"""
    <div style="margin:32px 0 14px 0;padding:12px 16px;
                background:linear-gradient(90deg,rgba(56,139,253,0.08) 0%,transparent 100%);
                border-left:3px solid #388bfd;border-radius:0 6px 6px 0">
        <h3 style="color:#e6edf3;margin:0;font-size:1.1rem;font-weight:700;
                   letter-spacing:0.01em;line-height:1.3">{title}</h3>
        {sub}
    </div>"""


def interpretation_box(text: str) -> str:
    """Render a styled interpretation / commentary block."""
    return f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1c2128 100%);
                border:1px solid #30363d;border-left:3px solid #2ea043;
                border-radius:10px;padding:18px 22px;margin:18px 0;
                font-size:0.9rem;color:#c9d1d9;line-height:1.75;
                box-shadow:0 2px 8px rgba(0,0,0,0.25)">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <span style="font-size:1rem">📊</span>
            <span style="color:#39d353;font-weight:700;font-size:0.82rem;
                         text-transform:uppercase;letter-spacing:0.1em">Interpretation</span>
        </div>
        <div style="border-top:1px solid rgba(46,160,67,0.25);padding-top:10px">{text}</div>
    </div>"""


def page_css() -> str:
    """
    Return a complete CSS block for consistent dark-theme styling across all pages.
    Inject with: st.markdown(page_css(), unsafe_allow_html=True)
    """
    from src.config import THEME
    return f"""
    <style>
    /* ── Base ─────────────────────────────────────────────────── */
    .stApp {{ background-color:{THEME['bg']}; color:{THEME['text']}; font-family:'Inter',system-ui,sans-serif; }}
    [data-testid="stSidebar"] {{
        background-color:{THEME['bg_secondary']};
        border-right:1px solid {THEME['border']};
    }}
    [data-testid="stSidebar"] h3 {{
        font-size:0.9rem !important;
        color:{THEME['text_dim']} !important;
        text-transform:uppercase;
        letter-spacing:0.07em;
        margin-bottom:8px;
    }}
    /* ── Metrics ───────────────────────────────────────────────── */
    div[data-testid="metric-container"] {{
        background:linear-gradient(135deg,{THEME['bg_card']} 0%,{THEME['bg_secondary']} 100%);
        border:1px solid {THEME['border']};
        border-radius:10px;
        padding:14px 18px;
        transition:border-color 0.2s;
    }}
    div[data-testid="metric-container"]:hover {{
        border-color:{THEME['blue']};
    }}
    div[data-testid="metric-container"] label {{
        color:{THEME['text_dim']} !important;
        font-size:0.75rem !important;
        text-transform:uppercase;
        letter-spacing:0.07em;
        font-weight:500;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size:1.45rem !important;
        font-weight:700 !important;
        color:{THEME['text']} !important;
    }}
    /* ── Tabs ───────────────────────────────────────────────────── */
    div[data-baseweb="tab-list"] {{
        background-color:{THEME['bg_secondary']};
        border-bottom:2px solid {THEME['border']};
        padding:0 4px;
        gap:0;
        flex-wrap:nowrap;
        overflow-x:auto;
    }}
    /* Hide the default BaseWeb animated slide-indicator (causes visual overlap) */
    div[data-baseweb="tab-highlight"] {{
        display:none !important;
    }}
    div[data-baseweb="tab-border"] {{
        display:none !important;
    }}
    button[data-baseweb="tab"] {{
        color:{THEME['text_dim']};
        padding:10px 18px;
        white-space:nowrap;
        border-bottom:2px solid transparent !important;
        border-radius:0 !important;
        background:transparent !important;
        font-size:0.88rem;
        flex-shrink:0;
    }}
    button[data-baseweb="tab"]:hover {{
        color:{THEME['text']};
        background:rgba(56,139,253,0.06) !important;
    }}
    button[aria-selected="true"][data-baseweb="tab"] {{
        color:{THEME['text']} !important;
        border-bottom:2px solid {THEME['blue']} !important;
        font-weight:600;
    }}
    /* ── Dataframes ─────────────────────────────────────────────── */
    .dvn-scroller {{ scrollbar-width:thin; scrollbar-color:{THEME['border']} transparent; }}
    thead tr th {{ background-color:{THEME['bg_secondary']} !important; color:{THEME['text_dim']} !important; }}
    tbody tr:hover td {{ background-color:{THEME['bg_card']} !important; }}
    /* ── Expander ────────────────────────────────────────────────── */
    details[data-testid="stExpander"] {{
        background-color:{THEME['bg_secondary']};
        border:1px solid {THEME['border']};
        border-radius:8px;
        margin:6px 0;
    }}
    /* ── Scrollbar ───────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width:6px; height:6px; }}
    ::-webkit-scrollbar-track {{ background:transparent; }}
    ::-webkit-scrollbar-thumb {{ background:{THEME['border']}; border-radius:3px; }}
    /* ── Sidebar label ───────────────────────────────────────────── */
    [data-testid="stSidebarContent"] .st-emotion-cache-dvne4q {{
        font-size:0.85rem;
    }}
    /* ── Hide Streamlit footer ────────────────────────────────────── */
    footer {{ visibility:hidden; }}
    </style>"""


def page_header_html(title: str, subtitle: str = "", icon: str = "📈") -> str:
    """
    Premium gradient page banner with subtle mesh pattern and accent divider.
    """
    from src.config import THEME
    sub_html = (
        f'<p style="color:{THEME["text_dim"]};margin:8px 0 0 0;font-size:0.92rem;'
        f'font-weight:400;max-width:800px;line-height:1.5">{subtitle}</p>'
        if subtitle else ""
    )
    return f"""
    <div style="background:linear-gradient(135deg,#0a0e14 0%,#0d1117 40%,#161b22 70%,#0d1117 100%);
                border:1px solid {THEME['border']};border-radius:12px;
                padding:26px 32px 24px 32px;margin-bottom:24px;
                box-shadow:0 4px 20px rgba(0,0,0,0.4),inset 0 1px 0 rgba(56,139,253,0.1);
                position:relative;overflow:hidden">
        <div style="position:absolute;top:0;right:0;width:300px;height:100%;
                    background:radial-gradient(ellipse at top right,rgba(56,139,253,0.05) 0%,transparent 70%);
                    pointer-events:none"></div>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
            <span style="font-size:1.6rem;line-height:1">{icon}</span>
            <h1 style="margin:0;font-size:1.7rem;font-weight:800;
                       background:linear-gradient(90deg,#79c0ff 0%,#388bfd 40%,#2ea043 100%);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       letter-spacing:-0.02em">{title}</h1>
        </div>
        <div style="height:2px;background:linear-gradient(90deg,rgba(56,139,253,0.6),rgba(46,160,67,0.4),transparent);
                    border-radius:1px;margin:10px 0 0 0"></div>
        {sub_html}
    </div>"""
