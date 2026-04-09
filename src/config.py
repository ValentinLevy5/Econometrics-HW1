"""
config.py
---------
App-wide constants, paths, color schemes, and configuration values.
"""

import os
from pathlib import Path

# ── Directory layout ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = BASE_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Yahoo Finance settings ───────────────────────────────────────────────────
INDEX_TICKER  = "^GSPC"
START_DATE    = "2016-01-01"

# Subperiod definitions
PERIODS = {
    "Full Sample":  ("2016-01-01", None),          # None → today
    "Pre-2020":     ("2016-01-01", "2019-12-31"),
    "Post-2021":    ("2021-01-01", None),
}

# Minimum number of valid trading-day observations required to include a
# stock in a given subperiod analysis.
MIN_OBS_FULL    = 500
MIN_OBS_SUBPERIOD = 200

# ── Rolling window ───────────────────────────────────────────────────────────
ROLLING_WINDOW = 30          # trading days

# ── KDE settings ────────────────────────────────────────────────────────────
KDE_GRID_POINTS  = 200       # evaluation grid size for each KDE
KDE_BANDWIDTH    = "silverman"  # passed to scipy.stats.gaussian_kde

# Cross-sectional density surface: evaluate every N trading days
XSECTION_STRIDE  = 5

# ── Cache file names ─────────────────────────────────────────────────────────
PRICES_CACHE      = DATA_DIR / "prices.parquet"
RETURNS_CACHE     = DATA_DIR / "returns.parquet"
SECTOR_CACHE      = CACHE_DIR / "sector_info.parquet"
MOMENTS_CACHE     = CACHE_DIR / "stock_moments.parquet"
CORR_CACHE        = CACHE_DIR / "pairwise_corr.parquet"
XSEC_MOMENTS_CACHE = CACHE_DIR / "xsec_moments.parquet"

# ── Dark theme colors ────────────────────────────────────────────────────────
THEME = {
    "bg":          "#0e1117",
    "bg_secondary":"#161b22",
    "bg_card":     "#1c2128",
    "border":      "#30363d",
    "text":        "#e6edf3",
    "text_dim":    "#8b949e",
    "green":       "#26a641",
    "green_bright":"#39d353",
    "red":         "#da3633",
    "red_bright":  "#f85149",
    "yellow":      "#d29922",
    "blue":        "#388bfd",
    "purple":      "#bc8cff",
    "orange":      "#ffa657",
    "teal":        "#2ea043",
    "accent":      "#1f6feb",
}

# Plotly base layout defaults
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT_DEFAULTS = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=THEME["bg"],
    plot_bgcolor=THEME["bg_secondary"],
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", color=THEME["text"], size=12),
    margin=dict(l=55, r=30, t=65, b=55),
    colorway=[
        THEME["blue"], THEME["green_bright"], THEME["orange"],
        THEME["purple"], THEME["teal"], THEME["yellow"],
        THEME["red_bright"], "#79c0ff", "#a5d6ff",
    ],
    xaxis=dict(
        showgrid=False,
        linecolor=THEME["border"],
        tickfont=dict(size=11, color=THEME["text_dim"]),
        title_font=dict(size=12, color=THEME["text_dim"]),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor=THEME["border"],
        gridwidth=0.5,
        linecolor=THEME["border"],
        tickfont=dict(size=11, color=THEME["text_dim"]),
        title_font=dict(size=12, color=THEME["text_dim"]),
    ),
    hoverlabel=dict(
        bgcolor=THEME["bg_card"],
        bordercolor=THEME["border"],
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=THEME["text"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=THEME["border"],
        borderwidth=1,
        font=dict(size=11, color=THEME["text_dim"]),
    ),
    # Explicitly set title text to "" so Plotly.js never renders "undefined"
    title=dict(text="", font=dict(size=14, color=THEME["text"])),
)

# ── Sector color palette ─────────────────────────────────────────────────────
SECTOR_COLORS = {
    "Technology":             "#388bfd",
    "Health Care":            "#2ea043",
    "Financials":             "#ffa657",
    "Consumer Discretionary": "#d29922",
    "Communication Services": "#bc8cff",
    "Industrials":            "#79c0ff",
    "Consumer Staples":       "#39d353",
    "Energy":                 "#f85149",
    "Utilities":              "#a5d6ff",
    "Real Estate":            "#ff7b72",
    "Materials":              "#e3b341",
    "Unknown":                "#8b949e",
}

# ── Return / color scale for heatmaps ───────────────────────────────────────
# Diverging red-green scale mimicking Finviz
FINVIZ_COLORSCALE = [
    [0.0,  "#8b1a1a"],
    [0.1,  "#c0392b"],
    [0.2,  "#e74c3c"],
    [0.35, "#a93226"],
    [0.45, "#2c2c2c"],
    [0.5,  "#2c2c2c"],
    [0.55, "#1e4d2b"],
    [0.65, "#27ae60"],
    [0.8,  "#2ecc71"],
    [0.9,  "#1a8a3c"],
    [1.0,  "#0d5c26"],
]

# Six stock-level moment names (used across multiple modules)
MOMENT_COLS = ["mean", "variance", "skewness", "kurtosis", "p1", "p99"]
MOMENT_LABELS = {
    "mean":     "Mean Return (%)",
    "variance": "Variance (%²)",
    "skewness": "Skewness",
    "kurtosis": "Excess Kurtosis",
    "p1":       "1st Percentile (%)",
    "p99":      "99th Percentile (%)",
}
