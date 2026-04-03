"""
visualizations.py
-----------------
All Plotly chart builders used across the Streamlit pages.

Every function returns a plotly.graph_objects.Figure ready to be passed
to st.plotly_chart(..., use_container_width=True).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    PLOTLY_LAYOUT_DEFAULTS,
    SECTOR_COLORS,
    MOMENT_LABELS,
    MOMENT_COLS,
    THEME,
)
from src.analytics import estimate_kde


# ── Layout helper ─────────────────────────────────────────────────────────────
def _base_layout(**kwargs) -> dict:
    d = dict(PLOTLY_LAYOUT_DEFAULTS)
    d.update(kwargs)
    return d


# ── Color helper: hex → rgba (Plotly 6.x dropped 8-digit hex alpha) ──────────
def _rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,100,100,{alpha})"


# ── DataFrame safety helpers ──────────────────────────────────────────────────
def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first occurrence of duplicated column names."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _ensure_single_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee exactly one 'ticker' column.
    """
    df = df.copy()
    df = _drop_duplicate_columns(df)

    if "ticker" in df.columns:
        return df

    idx_name = df.index.name

    if idx_name == "ticker":
        df = df.reset_index()
        df = _drop_duplicate_columns(df)
        return df

    try:
        df = df.reset_index(names="ticker")
    except TypeError:
        df = df.reset_index()
        first_col = df.columns[0]
        if first_col != "ticker":
            df = df.rename(columns={first_col: "ticker"})

    df = _drop_duplicate_columns(df)
    return df


def _safe_numeric_series(s: pd.Series, fill_value: float = np.nan) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if pd.notna(fill_value):
        out = out.fillna(fill_value)
    return out


def _metric_suffix(metric_name: str) -> str:
    metric_name = metric_name.lower()
    if any(x in metric_name for x in ["return", "mean", "p01", "p99", "ytd"]):
        return "%"
    return ""


def _metric_to_hex(value: float, vmax: float) -> str:
    """
    Explicit numeric-to-color mapping for leaf treemap tiles.
    This avoids Plotly treemap parent/branch color quirks.

    Negative -> red
    Near zero -> dark neutral
    Positive -> green
    """
    if pd.isna(value):
        return "#3b404c"

    if vmax <= 0:
        vmax = 1.0

    x = float(np.clip(value / vmax, -1.0, 1.0))

    # anchor colors
    neg_hi = np.array([122, 23, 23])    # deep red
    neg_lo = np.array([201, 60, 45])    # red-orange
    neutral = np.array([59, 64, 76])    # dark neutral
    pos_lo = np.array([53, 95, 61])     # dark green
    pos_hi = np.array([123, 214, 126])  # bright green

    if x < 0:
        t = abs(x)
        if t < 0.5:
            # neutral -> red-orange
            w = t / 0.5
            rgb = neutral * (1 - w) + neg_lo * w
        else:
            # red-orange -> deep red
            w = (t - 0.5) / 0.5
            rgb = neg_lo * (1 - w) + neg_hi * w
    else:
        t = x
        if t < 0.5:
            # neutral -> dark green
            w = t / 0.5
            rgb = neutral * (1 - w) + pos_lo * w
        else:
            # dark green -> bright green
            w = (t - 0.5) / 0.5
            rgb = pos_lo * (1 - w) + pos_hi * w

    rgb = np.clip(rgb.round().astype(int), 0, 255)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Finviz-inspired Market Treemap
# ══════════════════════════════════════════════════════════════════════════════

def treemap_market_map(
    stock_data: pd.DataFrame,
    color_metric: str = "daily_return",
    sector_map: Optional[Dict[str, str]] = None,
    market_caps: Optional[pd.Series] = None,
    title: str = "S&P 500 Market Map",
    size_mode: str = "market_cap",
) -> go.Figure:
    """
    Finviz-style interactive treemap.

    Default:
    - area = market cap
    - color = selected metric

    Optional:
    - area = absolute magnitude of selected metric

    Important:
    - Sector/header nodes are forced neutral
    - Only stock leaf nodes get signed red/green coloring
    - This avoids Plotly's parent-node averaging problem
    """
    df = stock_data.copy()
    df = _ensure_single_ticker_column(df)
    df = _drop_duplicate_columns(df)

    if "ticker" not in df.columns:
        raise ValueError("treemap_market_map requires a 'ticker' column or ticker index.")

    if color_metric not in df.columns:
        raise ValueError(f"Column '{color_metric}' not found in stock_data.")

    df["ticker"] = df["ticker"].astype(str)

    if sector_map:
        df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
    else:
        if "sector" not in df.columns:
            df["sector"] = "Unknown"
        df["sector"] = df["sector"].fillna("Unknown")

    df["sector"] = df["sector"].astype(str)
    df[color_metric] = pd.to_numeric(df[color_metric], errors="coerce")
    df = df.dropna(subset=["ticker", color_metric]).copy()

    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title=title, height=500))
        return fig

    # ---------- SIZE ----------
    if size_mode == "market_cap":
        if market_caps is not None:
            size_series = df["ticker"].map(market_caps)
        else:
            size_series = pd.to_numeric(df.get("mktcap", 1.0), errors="coerce")
        df["tile_size"] = pd.to_numeric(size_series, errors="coerce").fillna(1.0).clip(lower=1e-9)

    elif size_mode == "metric_abs":
        abs_metric = df[color_metric].abs()
        floor = max(float(abs_metric.quantile(0.15)), 0.05)
        df["tile_size"] = abs_metric.clip(lower=floor)
        df["tile_size"] = np.sqrt(df["tile_size"])

        # Normalize within sector so one stock does not swallow everything
        sector_totals = df.groupby("sector")["tile_size"].transform("sum")
        df["tile_size"] = (df["tile_size"] / sector_totals).clip(lower=1e-6)

    else:
        raise ValueError("size_mode must be 'market_cap' or 'metric_abs'")

    # ---------- COLOR ----------
    col = df[color_metric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(col) == 0:
        vmax = 1.0
    else:
        q05 = float(col.quantile(0.05))
        q95 = float(col.quantile(0.95))
        vmax = max(abs(q05), abs(q95), 0.25)

    metric_label = color_metric.replace("_", " ").title()
    suffix = _metric_suffix(color_metric)

    # ---------- Build manual treemap nodes ----------
    labels = []
    ids = []
    parents = []
    values = []
    texts = []
    colors = []
    customdata = []

    root_id = "root"
    root_label = "S&P 500"
    root_value = float(df["tile_size"].sum())

    labels.append(root_label)
    ids.append(root_id)
    parents.append("")
    values.append(root_value)
    texts.append(root_label)
    colors.append(THEME["bg_secondary"])
    customdata.append(["", "Root", np.nan, root_value, root_label, False])

    for sector in sorted(df["sector"].unique()):
        sub = df[df["sector"] == sector].copy()
        sector_id = f"sector::{sector}"
        sector_value = float(sub["tile_size"].sum())

        labels.append(sector)
        ids.append(sector_id)
        parents.append(root_id)
        values.append(sector_value)
        texts.append(sector)
        colors.append(THEME["bg_secondary"])   # keep sector headers neutral
        customdata.append(["", sector, np.nan, sector_value, sector, False])

        for _, r in sub.iterrows():
            ticker = str(r["ticker"])
            metric_val = float(r[color_metric])
            tile_size = float(r["tile_size"])
            display_text = f"{ticker}<br>{metric_val:+.2f}{suffix}"

            labels.append(ticker)
            ids.append(f"stock::{ticker}")
            parents.append(sector_id)
            values.append(tile_size)
            texts.append(display_text)
            colors.append(_metric_to_hex(metric_val, vmax))
            customdata.append([ticker, sector, metric_val, tile_size, display_text, True])

    customdata_arr = np.array(customdata, dtype=object)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            ids=ids,
            parents=parents,
            values=values,
            branchvalues="total",
            text=texts,
            texttemplate="%{text}",
            textfont=dict(color="white", size=12),
            marker=dict(
                colors=colors,
                line=dict(color="rgba(10,10,10,0.95)", width=1.2),
            ),
            root_color=THEME["bg"],
            tiling=dict(pad=2),
            pathbar=dict(visible=False),
            customdata=customdata_arr,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Sector: %{customdata[1]}<br>"
                f"{metric_label}: %{{customdata[2]:+.3f}}{suffix}<br>"
                "Tile size: %{customdata[3]:.4f}<extra></extra>"
            ),
        )
    )

    # colorbar proxy so the user still sees a scale legend
    # uses an invisible scatter trace
    scale_vals = np.linspace(-vmax, vmax, 101)
    fig.add_trace(
        go.Scatter(
            x=[None] * len(scale_vals),
            y=[None] * len(scale_vals),
            mode="markers",
            marker=dict(
                colorscale=[
                    [0.00, "#7a1717"],
                    [0.20, "#c93c2d"],
                    [0.45, "#2f3440"],
                    [0.50, "#3b404c"],
                    [0.55, "#355f3d"],
                    [0.80, "#49a35d"],
                    [1.00, "#7bd67e"],
                ],
                cmin=-vmax,
                cmax=vmax,
                color=scale_vals,
                size=0.01,
                showscale=True,
                colorbar=dict(
                    title=metric_label,
                    tickformat=".2f",
                    len=0.75,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=16)),
            margin=dict(l=8, r=8, t=40, b=8),
            height=780,
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["bg"],
            clickmode="event",
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. KDE plots for 6 stock-level moment statistics
# ══════════════════════════════════════════════════════════════════════════════

def kde_moments_grid(
    moments: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
    highlight_sector: Optional[str] = None,
) -> go.Figure:
    """
    2×3 grid of KDE density plots, one for each of the six moment statistics.
    Optionally overlays a highlighted sector's distribution.
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[MOMENT_LABELS[c] for c in MOMENT_COLS],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    palette = [
        THEME["blue"],
        THEME["green_bright"],
        THEME["orange"],
        THEME["purple"],
        THEME["yellow"],
        THEME["teal"],
    ]

    for idx, col in enumerate(MOMENT_COLS):
        r, c = positions[idx]
        arr = moments[col].dropna().values
        if len(arr) < 5:
            continue

        x, d = estimate_kde(arr)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=d,
                mode="lines",
                name=MOMENT_LABELS[col],
                line=dict(color=palette[idx], width=2),
                fill="tozeroy",
                fillcolor=_rgba(palette[idx], 0.15),
                showlegend=False,
                hovertemplate=f"{MOMENT_LABELS[col]}: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        if highlight_sector and sector_map:
            tickers_in_sector = [t for t, s in sector_map.items() if s == highlight_sector]
            arr_sec = moments.loc[moments.index.isin(tickers_in_sector), col].dropna().values
            if len(arr_sec) >= 5:
                xs, ds = estimate_kde(arr_sec, x_range=(x.min(), x.max()))
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ds,
                        mode="lines",
                        name=highlight_sector,
                        line=dict(color=THEME["red_bright"], width=1.5, dash="dash"),
                        showlegend=(idx == 0),
                    ),
                    row=r,
                    col=c,
                )

        med = float(np.median(arr))
        fig.add_vline(
            x=med,
            line_dash="dot",
            line_color=THEME["text_dim"],
            line_width=1,
            row=r,
            col=c,
            annotation_text=f"med={med:.2f}",
            annotation_font_size=9,
            annotation_font_color=THEME["text_dim"],
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Nonparametric Density Estimates — Stock-Level Moments",
                font=dict(size=15),
            ),
            height=520,
            showlegend=bool(highlight_sector),
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["border"])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rolling index moments (Q3)
# ══════════════════════════════════════════════════════════════════════════════

def rolling_moments_plot(
    rolling_df: pd.DataFrame,
    title: str = "30-Day Rolling Moments — S&P 500 Index Returns",
) -> go.Figure:
    """
    Four-panel time series of rolling mean, variance, skewness, kurtosis.
    """
    moment_config = [
        ("mean", "Mean (%)", THEME["blue"], (1, 1)),
        ("variance", "Variance (%²)", THEME["orange"], (1, 2)),
        ("skewness", "Skewness", THEME["purple"], (2, 1)),
        ("kurtosis", "Excess Kurtosis", THEME["green_bright"], (2, 2)),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Rolling Mean",
            "Rolling Variance",
            "Rolling Skewness",
            "Rolling Excess Kurtosis",
        ],
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    for col, label, color, (r, c) in moment_config:
        if col not in rolling_df.columns:
            continue
        s = rolling_df[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=label,
                line=dict(color=color, width=1.5),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.4f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=THEME["text_dim"],
            line_width=0.8,
            row=r,
            col=c,
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=500,
            showlegend=False,
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["border"])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. 3D Cross-sectional KDE surface
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_3d_surface(
    dates: list,
    x_grid: np.ndarray,
    Z: np.ndarray,
    title: str = "Cross-Sectional Return Density Over Time",
) -> go.Figure:
    """
    3D surface: time × return × density.
    """
    date_nums = [(d - dates[0]).days for d in dates]

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_grid,
                y=date_nums,
                z=Z,
                colorscale="Viridis",
                opacity=0.88,
                showscale=True,
                colorbar=dict(title="Density", len=0.5),
                hovertemplate="Return: %{x:.2f}%<br>Density: %{z:.4f}<extra></extra>",
            )
        ]
    )

    n_ticks = min(10, len(dates))
    step = max(1, len(dates) // n_ticks)
    tick_vals = [date_nums[i] for i in range(0, len(dates), step)]
    tick_texts = [str(dates[i].date()) for i in range(0, len(dates), step)]

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=600,
            scene=dict(
                xaxis=dict(
                    title="Daily Return (%)",
                    gridcolor=THEME["border"],
                    backgroundcolor=THEME["bg_secondary"],
                ),
                yaxis=dict(
                    title="Date",
                    tickvals=tick_vals,
                    ticktext=tick_texts,
                    gridcolor=THEME["border"],
                    backgroundcolor=THEME["bg_secondary"],
                ),
                zaxis=dict(
                    title="Density",
                    gridcolor=THEME["border"],
                    backgroundcolor=THEME["bg_secondary"],
                ),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
                bgcolor=THEME["bg"],
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cross-sectional moment time series (Q4)
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_moments_plot(
    xsec_df: pd.DataFrame,
    title: str = "Daily Cross-Sectional Moments of Stock Returns",
) -> go.Figure:
    """
    Three time-series subplots: cross-sectional variance, skewness, kurtosis.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Cross-Sectional Variance (%²)",
            "Cross-Sectional Skewness",
            "Cross-Sectional Excess Kurtosis",
        ],
        shared_xaxes=True,
        vertical_spacing=0.07,
    )

    config = [
        ("variance", THEME["orange"], 1),
        ("skewness", THEME["purple"], 2),
        ("kurtosis", THEME["green_bright"], 3),
    ]

    for col, color, row in config:
        if col not in xsec_df.columns:
            continue
        s = xsec_df[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line=dict(color=color, width=1.2),
                name=col.title(),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{col}: %{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color=THEME["text_dim"],
            line_width=0.8,
            row=row,
            col=1,
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=600,
            showlegend=False,
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["border"])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pairwise scatterplots of 6 moment metrics (Q5)
# ══════════════════════════════════════════════════════════════════════════════

def pairwise_scatterplot(
    moments: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    sector_map: Optional[Dict[str, str]] = None,
    color_by_sector: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Scatterplot of two moment metrics with optional sector coloring.
    """
    df = moments[[x_metric, y_metric]].dropna().copy()
    df.index.name = "ticker"
    df = df.reset_index()

    if sector_map and color_by_sector:
        df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
        color_col = "sector"
        color_map = SECTOR_COLORS
    else:
        df["sector"] = "All Stocks"
        color_col = "sector"
        color_map = {"All Stocks": THEME["blue"]}

    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color=color_col,
        color_discrete_map=color_map,
        hover_name="ticker",
        hover_data={"sector": True},
        labels={
            x_metric: MOMENT_LABELS.get(x_metric, x_metric),
            y_metric: MOMENT_LABELS.get(y_metric, y_metric),
        },
        title=title or f"{MOMENT_LABELS.get(x_metric)} vs {MOMENT_LABELS.get(y_metric)}",
        opacity=0.7,
    )

    x_arr = df[x_metric].values
    y_arr = df[y_metric].values
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if valid.sum() > 10:
        coef = np.polyfit(x_arr[valid], y_arr[valid], 1)
        x_line = np.linspace(x_arr[valid].min(), x_arr[valid].max(), 200)
        y_line = np.polyval(coef, x_line)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=THEME["yellow"], width=1.5, dash="dash"),
                name=f"OLS (slope={coef[0]:.3f})",
                showlegend=True,
            )
        )

    fig.update_layout(
        **_base_layout(
            height=480,
            legend=dict(
                orientation="v",
                bgcolor=THEME["bg_card"],
                bordercolor=THEME["border"],
            ),
        )
    )
    fig.update_traces(marker=dict(size=5))
    return fig


def all_pairs_subplot(
    moments: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    15-panel grid of all pairwise scatterplots (C(6,2) = 15 pairs).
    """
    from itertools import combinations

    pairs = list(combinations(MOMENT_COLS, 2))
    n_cols = 3
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=0.07,
        horizontal_spacing=0.06,
    )

    if sector_map:
        sectors_in_data = [sector_map.get(t, "Unknown") for t in moments.index]
        color_map_vals = [SECTOR_COLORS.get(s, "#8b949e") for s in sectors_in_data]
    else:
        color_map_vals = [THEME["blue"]] * len(moments)

    for k, (xc, yc) in enumerate(pairs):
        row = k // n_cols + 1
        col = k % n_cols + 1
        sub = moments[[xc, yc]].dropna()

        colors = (
            [color_map_vals[moments.index.get_loc(t)] for t in sub.index if t in moments.index]
            if sector_map
            else [THEME["blue"]] * len(sub)
        )

        fig.add_trace(
            go.Scatter(
                x=sub[xc].values,
                y=sub[yc].values,
                mode="markers",
                marker=dict(size=3, color=colors, opacity=0.6),
                text=sub.index.tolist(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + f"{MOMENT_LABELS.get(xc, 'X')}: %{{x:.3f}}<br>"
                    + f"{MOMENT_LABELS.get(yc, 'Y')}: %{{y:.3f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=xc[:4], row=row, col=col, title_font=dict(size=9))
        fig.update_yaxes(title_text=yc[:4], row=row, col=col, title_font=dict(size=9))

    fig.update_layout(
        **_base_layout(
            title=dict(text="All Pairwise Metric Scatterplots", font=dict(size=15)),
            height=max(400, n_rows * 200),
            showlegend=False,
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. Stock–index correlation KDE (Q6)
# ══════════════════════════════════════════════════════════════════════════════

def kde_corr_to_index(
    corr_series: pd.Series,
    sector_map: Optional[Dict[str, str]] = None,
    title: str = "Distribution of Stock–Index (^GSPC) Correlations",
) -> go.Figure:
    """
    KDE + histogram of stock–index correlations, optionally broken out by sector.
    """
    fig = go.Figure()

    arr = corr_series.dropna().values
    if len(arr) == 0:
        fig.update_layout(**_base_layout(title=title, height=300))
        return fig

    x, d = estimate_kde(arr, x_range=(-1, 1))

    fig.add_trace(
        go.Histogram(
            x=arr,
            nbinsx=50,
            name="All stocks",
            histnorm="probability density",
            marker_color=THEME["blue"],
            opacity=0.4,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=d,
            mode="lines",
            name="KDE (all)",
            line=dict(color=THEME["blue"], width=2.5),
        )
    )

    if sector_map:
        sector_set = sorted({s for s in sector_map.values() if s != "Unknown"})
        for sec in sector_set[:8]:
            tickers = [t for t, s in sector_map.items() if s == sec and t in corr_series.index]
            arr_s = corr_series.loc[corr_series.index.isin(tickers)].dropna().values
            if len(arr_s) < 5:
                continue
            xs, ds = estimate_kde(arr_s, x_range=(-1, 1))
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ds,
                    mode="lines",
                    name=sec,
                    line=dict(color=SECTOR_COLORS.get(sec, "#888"), width=1.5, dash="dot"),
                )
            )

    median_corr = float(np.median(arr))
    fig.add_vline(
        x=median_corr,
        line_dash="dash",
        line_color=THEME["yellow"],
        annotation_text=f"Median={median_corr:.3f}",
        annotation_font_color=THEME["yellow"],
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            xaxis_title="Pearson Correlation with ^GSPC",
            yaxis_title="Density",
            height=450,
            barmode="overlay",
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. Sector within vs. between correlation (Q7)
# ══════════════════════════════════════════════════════════════════════════════

def sector_within_between_plot(
    within_corr: pd.Series,
    between_corr: pd.Series,
    sector_avg_df: pd.DataFrame,
    test_results: dict,
) -> go.Figure:
    """
    Two-panel figure:
      Left  — KDE of within vs. between sector correlations
      Right — Average within-sector correlation bar chart by sector
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Within- vs. Between-Sector Correlation Distribution",
            "Mean Within-Sector Correlation by Sector",
        ],
        column_widths=[0.55, 0.45],
    )

    for arr, name, color in [
        (within_corr.values, "Within-sector", THEME["green_bright"]),
        (between_corr.values, "Between-sector", THEME["red_bright"]),
    ]:
        arr = arr[~np.isnan(arr)]
        if len(arr) < 5:
            continue
        x, d = estimate_kde(arr, x_range=(-0.2, 1.0))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=d,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.15),
                hovertemplate="Corr: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if not sector_avg_df.empty:
        fig.add_trace(
            go.Bar(
                x=sector_avg_df["mean_within_corr"].values,
                y=sector_avg_df["sector"].values,
                orientation="h",
                marker_color=sector_avg_df["color"].tolist(),
                text=[f"{v:.3f}" for v in sector_avg_df["mean_within_corr"]],
                textposition="outside",
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Mean corr: %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    p_welch = test_results.get("welch_p", 1.0)
    p_mw = test_results.get("mw_p", 1.0)
    annot = (
        f"Welch t-test: p={p_welch:.4f} {'✓' if p_welch < 0.05 else '✗'}<br>"
        f"Mann–Whitney: p={p_mw:.4f} {'✓' if p_mw < 0.05 else '✗'}"
    )
    fig.add_annotation(
        text=annot,
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.97,
        showarrow=False,
        font=dict(size=10, color=THEME["text_dim"]),
        align="left",
        bgcolor=THEME["bg_card"],
        bordercolor=THEME["border"],
        borderpad=6,
        row=1,
        col=1,
    )

    fig.update_layout(
        **_base_layout(
            height=450,
            legend=dict(orientation="h", y=1.05),
        )
    )
    fig.update_xaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Mean Within-Sector Corr", row=1, col=2)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9. Stability / Subperiod comparison (Q8, Q9)
# ══════════════════════════════════════════════════════════════════════════════

def stability_kde_overlay(
    returns_dict: Dict[str, pd.DataFrame],
    metric: str = "returns",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Overlay KDE plots for each subperiod — for either raw returns or a
    specified stock-level moment metric.
    """
    from src.preprocessing import get_stock_returns

    colors = [THEME["blue"], THEME["green_bright"], THEME["orange"]]
    fig = go.Figure()
    xlabel = "Daily Return (%)" if metric == "returns" else MOMENT_LABELS.get(metric, metric)

    for i, (period, ret) in enumerate(returns_dict.items()):
        color = colors[i % len(colors)]
        if metric == "returns":
            arr = get_stock_returns(ret).values.ravel()
            arr = arr[~np.isnan(arr)]
            if len(arr) > 50_000:
                arr = np.random.choice(arr, 50_000, replace=False)
            x_range = (-10, 10)
        else:
            from src.analytics import compute_stock_moments
            mom = compute_stock_moments(get_stock_returns(ret), force_refresh=True)
            if metric not in mom.columns:
                continue
            arr = mom[metric].dropna().values
            x_range = None

        if len(arr) < 10:
            continue

        x, d = estimate_kde(arr, x_range=x_range)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=d,
                mode="lines",
                name=period,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.15),
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=title or f"Stability Comparison — {metric.title()}",
                font=dict(size=15),
            ),
            xaxis_title=xlabel,
            yaxis_title="Density",
            height=420,
        )
    )
    return fig


def ks_results_heatmap(ks_df: pd.DataFrame, title: str = "KS Test Statistics") -> go.Figure:
    """
    Heatmap of KS test statistics across period pairs × metrics.
    """
    if ks_df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title=title, height=200))
        return fig

    pivot = ks_df.pivot_table(
        index="Metric",
        columns=["Period A", "Period B"],
        values="KS Stat",
        aggfunc="first",
    )
    pivot.columns = [f"{a} vs {b}" for a, b in pivot.columns]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=0.3,
            text=[[f"{v:.3f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="Periods: %{x}<br>Metric: %{y}<br>KS Stat: %{z:.4f}<extra></extra>",
            colorbar=dict(title="KS Stat"),
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=14)),
            height=max(250, 60 * len(pivot)),
        )
    )
    return fig


def risk_evolution_bar(risk_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing key risk indicators across subperiods.
    """
    metric_cols = [
        c for c in risk_df.columns
        if "Ann. Vol" in c or "Kurtosis" in c or "Skewness" in c
    ]
    colors = [THEME["blue"], THEME["orange"], THEME["green_bright"]]

    fig = go.Figure()
    for i, period in enumerate(risk_df.index):
        vals = [risk_df.loc[period, m] for m in metric_cols]
        fig.add_trace(
            go.Bar(
                name=period,
                x=metric_cols,
                y=vals,
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Risk Indicators Across Subperiods", font=dict(size=15)),
            barmode="group",
            xaxis_title="Risk Metric",
            yaxis_title="Value",
            height=420,
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 10. Correlation heatmap for a sector (Q7)
# ══════════════════════════════════════════════════════════════════════════════

def sector_corr_heatmap(
    corr: pd.DataFrame,
    sector_map: Dict[str, str],
    sector: str,
    max_tickers: int = 40,
) -> go.Figure:
    """
    Correlation heatmap for all stocks within a specified sector.
    """
    members = [t for t, s in sector_map.items() if s == sector and t in corr.index]
    members = members[:max_tickers]

    if len(members) < 2:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title=f"Not enough data for {sector}"))
        return fig

    sub = corr.loc[members, members].values

    fig = go.Figure(
        go.Heatmap(
            z=sub,
            x=members,
            y=members,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corr"),
            hovertemplate="<b>%{y} × %{x}</b><br>Corr: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=f"{sector} — Pairwise Correlation Heatmap", font=dict(size=14)),
            height=max(350, 14 * len(members)),
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 11. 2025 "Winning stocks" ranked table (bonus)
# ══════════════════════════════════════════════════════════════════════════════

def ytd_performance_bar(
    ytd_series: pd.Series,
    sector_map: Optional[Dict[str, str]] = None,
    top_n: int = 30,
    title: str = "Top Performers — YTD Return (%)",
    ascending: bool = False,
) -> go.Figure:
    """
    Horizontal bar chart of top/bottom N stocks by YTD return.
    """
    s = ytd_series.dropna().sort_values(ascending=ascending).head(top_n)

    colors = []
    for t in s.index:
        sec = sector_map.get(t, "Unknown") if sector_map else "Unknown"
        colors.append(SECTOR_COLORS.get(sec, THEME["blue"]))

    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=s.index.tolist(),
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in s.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>YTD: %{x:+.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis_title="YTD Return (%)",
            height=max(300, top_n * 18),
            margin=dict(l=80, r=60, t=50, b=40),
        )
    )
    return fig