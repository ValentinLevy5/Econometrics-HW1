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
    Dark-Terminal Red-Green colorscale for treemap tiles.

    Philosophy: keep the universal red=bad / green=good convention but
    make every stop richer and darker than the standard Finviz palette:

      Negative extreme → deep wine / maroon    (#5a0000)
      Negative mid     → vivid crimson          (#c0392b) — no orange tint
      Slight negative  → muted rose             (#7f2020)
      Neutral          → deep navy-charcoal     (#0f172a) — Bloomberg-style, NOT flat grey
      Slight positive  → dark forest green      (#1a4731)
      Positive mid     → vivid emerald          (#27ae60)
      Positive extreme → electric money-green   (#52e882)

    Key differences vs Finviz:
    · Neutral zone = dark navy (not grey/black)
    · Red  = pure wine-crimson (no orange contamination)
    · Green = pure emerald (no yellow/lime contamination)
    · Richer, darker tones → looks premium on a dark dashboard
    """
    if pd.isna(value):
        return "#131c2e"  # neutral tile: deep navy

    if vmax <= 0:
        vmax = 1.0

    x = float(np.clip(value / vmax, -1.0, 1.0))

    # Terminal Red-Green anchor colors
    neg_hi  = np.array([90,   0,   0])   # deep maroon / wine        (max negative)
    neg_mid = np.array([192,  57,  43])  # vivid crimson             (mid negative)
    neg_lo  = np.array([127,  32,  32])  # dark rose                 (slight negative)
    neutral = np.array([15,   23,  42])  # deep navy-charcoal        (zero)
    pos_lo  = np.array([26,   71,  49])  # dark forest green         (slight positive)
    pos_mid = np.array([39,  174,  96])  # vivid emerald             (mid positive)
    pos_hi  = np.array([82,  232, 130])  # electric money-green      (max positive)

    def _lerp(a, b, t):
        return a * (1 - t) + b * t

    if x < 0:
        t = abs(x)
        if t < 0.35:
            w = t / 0.35
            rgb = _lerp(neutral, neg_lo, w)
        elif t < 0.70:
            w = (t - 0.35) / 0.35
            rgb = _lerp(neg_lo, neg_mid, w)
        else:
            w = (t - 0.70) / 0.30
            rgb = _lerp(neg_mid, neg_hi, w)
    else:
        t = x
        if t < 0.35:
            w = t / 0.35
            rgb = _lerp(neutral, pos_lo, w)
        elif t < 0.70:
            w = (t - 0.35) / 0.35
            rgb = _lerp(pos_lo, pos_mid, w)
        else:
            w = (t - 0.70) / 0.30
            rgb = _lerp(pos_mid, pos_hi, w)

    rgb = np.clip(rgb.round().astype(int), 0, 255)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _metric_to_rgba(value: float, vmax: float, opacity: float = 1.0) -> str:
    """Like _metric_to_hex but returns rgba string with explicit opacity."""
    h = _metric_to_hex(value, vmax).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity:.3f})"


def _breadth_bar(pct_pos: float, width: int = 8) -> str:
    """ASCII progress bar showing fraction of stocks positive in a sector."""
    filled = int(round(pct_pos / 100 * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


# ══════════════════════════════════════════════════════════════════════════════
# 1. S&P 500 Interactive Market Map
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
    Advanced S&P 500 market map with dual visual encoding.

    Unique features beyond standard market maps:
    ─ Dark-Terminal Red-Green scale  (deep maroon → navy → electric emerald)
      Keeps the universal red=bad / green=good language but with far richer,
      darker tones than Finviz — designed for a dark dashboard.
    ─ DUAL ENCODING: hue = metric direction, opacity = volatility (variance)
      High-variance stocks glow vivid; stable stocks are muted.
      The eye is drawn to "hot spots": volatile + moving stocks.
    ─ Sector headers: ASCII breadth bar + avg metric + sector rank badge
    ─ Sector tile color: subtle tint of the sector's directional color
    ─ Rich hover: all 6 statistical moments + volatility rank
    ─ Live market breadth banner with segment bar (red|green)
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

    df["sector"]     = df["sector"].astype(str)
    df[color_metric] = pd.to_numeric(df[color_metric], errors="coerce")
    df = df.dropna(subset=["ticker", color_metric]).copy()

    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=16)), height=500))
        return fig

    # ── Tile sizing ───────────────────────────────────────────────────────────
    if size_mode == "market_cap":
        if market_caps is not None:
            size_series = df["ticker"].map(market_caps)
        else:
            size_series = pd.to_numeric(df.get("mktcap", 1.0), errors="coerce")
        df["tile_size"] = pd.to_numeric(size_series, errors="coerce").fillna(1.0).clip(lower=1e-9)
    elif size_mode == "metric_abs":
        abs_metric = df[color_metric].abs()
        floor = max(float(abs_metric.quantile(0.15)), 0.05)
        df["tile_size"] = np.sqrt(abs_metric.clip(lower=floor))
        sector_totals = df.groupby("sector")["tile_size"].transform("sum")
        df["tile_size"] = (df["tile_size"] / sector_totals).clip(lower=1e-6)
    else:
        raise ValueError("size_mode must be 'market_cap' or 'metric_abs'")

    # ── Color range ───────────────────────────────────────────────────────────
    col_clean = df[color_metric].replace([np.inf, -np.inf], np.nan).dropna()
    vmax = max(abs(col_clean.quantile(0.05)), abs(col_clean.quantile(0.95)), 0.25) if len(col_clean) else 1.0
    metric_label = color_metric.replace("_", " ").title()
    suffix = _metric_suffix(color_metric)

    # ── Opacity from variance (2nd encoding dimension) ─────────────────────
    # High variance → full opacity (vivid color = draws the eye)
    # Low variance  → reduced opacity (muted = recedes into background)
    if "variance" in df.columns:
        var_vals = pd.to_numeric(df["variance"], errors="coerce").fillna(0.0)
        var_min, var_max = var_vals.min(), var_vals.max()
        if var_max > var_min:
            df["opacity"] = 0.55 + 0.45 * (var_vals - var_min) / (var_max - var_min)
        else:
            df["opacity"] = 1.0
        # Also compute variance rank (1=most volatile) for hover
        df["var_rank"] = var_vals.rank(ascending=False).astype(int)
    else:
        df["opacity"]  = 1.0
        df["var_rank"] = np.nan

    # ── Market breadth stats ──────────────────────────────────────────────────
    n_total       = len(df)
    n_pos         = int((df[color_metric] > 0).sum())
    n_neg         = int((df[color_metric] < 0).sum())
    n_flat        = n_total - n_pos - n_neg
    pct_pos       = 100 * n_pos / max(n_total, 1)
    pct_neg       = 100 * n_neg / max(n_total, 1)
    sec_avgs      = df.groupby("sector")[color_metric].mean()
    best_sec      = sec_avgs.idxmax() if len(sec_avgs) else "—"
    worst_sec     = sec_avgs.idxmin() if len(sec_avgs) else "—"
    breadth_bar_w = 16
    bull_bars     = int(round(pct_pos / 100 * breadth_bar_w))
    bull_bars     = max(0, min(breadth_bar_w, bull_bars))
    bear_bars     = breadth_bar_w - bull_bars
    # Terminal Red-Green accent: crimson bears, emerald bulls
    BEAR_COL      = "#c0392b"   # vivid crimson
    BULL_COL      = "#27ae60"   # vivid emerald
    breadth_color = BULL_COL if pct_pos >= 50 else BEAR_COL

    # Pre-compute sector ranks (1 = best performing) for the sector header badge
    sec_ranks = sec_avgs.rank(ascending=False).astype(int).to_dict()

    # ── Build tree nodes ──────────────────────────────────────────────────────
    labels     = []
    ids        = []
    parents    = []
    values     = []
    texts      = []
    colors     = []
    customdata = []

    root_id  = "root"
    root_val = float(df["tile_size"].sum())
    # customdata layout (8 cols): [ticker, sector, metric_val, mean, variance, skewness, kurtosis, var_rank]
    labels.append("S&P 500"); ids.append(root_id); parents.append("")
    values.append(root_val);  texts.append("S&P 500")
    colors.append("#0f172a")  # deep navy-charcoal — matches the new neutral
    customdata.append(["", "Market", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    n_sectors = len(df["sector"].unique())
    for sector in sorted(df["sector"].unique()):
        sub       = df[df["sector"] == sector].copy()
        sec_id    = f"sector::{sector}"
        sec_val   = float(sub["tile_size"].sum())
        sec_avg   = float(sub[color_metric].mean())
        sec_n     = len(sub)
        sec_ppos  = float((sub[color_metric] > 0).mean() * 100)
        sec_bbar  = _breadth_bar(sec_ppos, width=7)
        sec_arrow = "▲" if sec_avg > 0 else "▼"
        sec_rank  = sec_ranks.get(sector, "?")
        # Sector header: breadth bar + avg + rank badge (#1 of 11)
        sec_text  = (
            f"<b>{sector}</b>   #{sec_rank}/{n_sectors}<br>"
            f"{sec_bbar} {sec_ppos:.0f}% advancing<br>"
            f"{sec_arrow} avg {sec_avg:+.3f}{suffix}  ·  {sec_n} stocks"
        )

        # Sector tile: very subtle tint (low opacity) using the new red-green scale
        sec_tint = _metric_to_rgba(sec_avg, vmax, opacity=0.28)

        labels.append(sector);  ids.append(sec_id);    parents.append(root_id)
        values.append(sec_val); texts.append(sec_text); colors.append(sec_tint)
        customdata.append(["", sector, sec_avg, np.nan, np.nan, np.nan, np.nan, np.nan])

        for _, r in sub.iterrows():
            ticker     = str(r["ticker"])
            metric_val = float(r[color_metric])
            tile_sz    = float(r["tile_size"])
            opacity    = float(r["opacity"])
            var_rank   = r["var_rank"]
            arrow      = "▲" if metric_val > 0 else "▼"
            disp_text  = f"<b>{ticker}</b><br>{arrow}{metric_val:+.2f}{suffix}"

            def _g(c): return float(r[c]) if c in r.index and pd.notna(r[c]) else np.nan
            m_mean = _g("mean"); m_var = _g("variance")
            m_skew = _g("skewness"); m_kurt = _g("kurtosis")

            labels.append(ticker);  ids.append(f"stock::{ticker}"); parents.append(sec_id)
            values.append(tile_sz); texts.append(disp_text)
            colors.append(_metric_to_rgba(metric_val, vmax, opacity))
            customdata.append([ticker, sector, metric_val, m_mean, m_var, m_skew, m_kurt, var_rank])

    customdata_arr = np.array(customdata, dtype=object)

    hover_tmpl = (
        "<b>%{customdata[0]}</b>"
        "  <span style='color:#8b949e;font-size:0.85em'>%{customdata[1]}</span><br>"
        f"<b>{metric_label}:</b> %{{customdata[2]:+.4f}}{suffix}<br>"
        "<span style='color:#8b949e'>──────────────────────</span><br>"
        "Mean return:  %{customdata[3]:.4f}<br>"
        "Variance:     %{customdata[4]:.4f}  "
        "<span style='color:#8b949e'>(volatility rank #%{customdata[7]})</span><br>"
        "Skewness:     %{customdata[5]:.4f}<br>"
        "Kurtosis:     %{customdata[6]:.4f}<extra></extra>"
    )

    # ── Dark-Terminal Red-Green colorscale (colorbar proxy) ──────────────────
    # Mirrors the _metric_to_hex function — same hue logic, continuous gradient
    terminal_rg_cs = [
        [0.00, "#5a0000"],   # deep maroon / wine       (max negative)
        [0.14, "#8b1a1a"],   # dark crimson
        [0.28, "#c0392b"],   # vivid crimson            (mid negative)
        [0.40, "#7f2020"],   # dark rose                (slight negative)
        [0.47, "#1e1a28"],   # dark transition
        [0.50, "#0f172a"],   # deep navy-charcoal       (zero)
        [0.53, "#131c2e"],   # dark transition
        [0.60, "#1a4731"],   # dark forest green        (slight positive)
        [0.72, "#27ae60"],   # vivid emerald            (mid positive)
        [0.86, "#2ecc71"],   # bright emerald
        [1.00, "#52e882"],   # electric money-green     (max positive)
    ]

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            ids=ids,
            parents=parents,
            values=values,
            branchvalues="total",
            text=texts,
            texttemplate="%{text}",
            textfont=dict(
                color="rgba(241,245,249,0.95)",
                size=11,
                family="Inter, system-ui, -apple-system, sans-serif",
            ),
            marker=dict(
                colors=colors,
                line=dict(color="rgba(5,5,8,0.90)", width=0.7),
                pad=dict(t=20, l=4, r=4, b=4),
            ),
            root_color=THEME["bg"],
            tiling=dict(pad=3, squarifyratio=1.618),
            pathbar=dict(
                visible=True, side="top",
                textfont=dict(size=11, color=THEME["text_dim"]),
            ),
            customdata=customdata_arr,
            hovertemplate=hover_tmpl,
        )
    )

    # ── Colorbar proxy (dark-terminal red-green scale) ────────────────────────
    sv = np.linspace(-vmax, vmax, 40)
    fig.add_trace(go.Scatter(
        x=[None] * 40, y=[None] * 40, mode="markers",
        marker=dict(
            colorscale=terminal_rg_cs,
            cmin=-vmax, cmax=vmax, color=sv, size=0.01,
            showscale=True,
            colorbar=dict(
                title=dict(text=metric_label, side="right",
                           font=dict(size=11, color=THEME["text_dim"])),
                tickformat="+.2f", len=0.65, thickness=14,
                tickfont=dict(size=9, color=THEME["text_dim"]),
                tickvals=[-vmax, -vmax/2, 0, vmax/2, vmax],
                ticktext=[
                    f"▼ {-vmax:+.2f}",
                    f"▼ {-vmax/2:+.2f}",
                    "  0",
                    f"▲ {vmax/2:+.2f}",
                    f"▲ {vmax:+.2f}",
                ],
            ),
        ),
        hoverinfo="skip", showlegend=False,
    ))

    # ── Market breadth banner ─────────────────────────────────────────────────
    # Segment bar: crimson blocks = bearish stocks, emerald = bullish stocks
    breadth_bar_str = (
        f"<span style='color:{BEAR_COL}'>{'▰' * bear_bars}</span>"
        f"<span style='color:{BULL_COL}'>{'▰' * bull_bars}</span>"
        f"  <b>{pct_pos:.0f}%</b> advancing  ·  "
        f"<span style='color:{BULL_COL}'>▲ {n_pos}</span>  "
        f"<span style='color:{BEAR_COL}'>▼ {n_neg}</span>  "
        f"<span style='color:#8b949e'>— {n_flat}</span>  "
        f"·  Best: <b>{best_sec}</b>  ·  Laggard: <b>{worst_sec}</b>"
    )
    fig.add_annotation(
        text=breadth_bar_str,
        xref="paper", yref="paper",
        x=0.5, y=1.038,
        xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=10.5, color=breadth_color,
                  family="Inter, system-ui, sans-serif"),
        align="center",
        bgcolor=_rgba(breadth_color, 0.07),
        bordercolor=_rgba(breadth_color, 0.28),
        borderpad=7,
        borderwidth=1,
    )

    # ── Opacity legend note ───────────────────────────────────────────────────
    fig.add_annotation(
        text="Tile brightness = volatility rank  (vivid = high variance)",
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=9, color=_rgba(THEME["text_dim"], 0.7),
                  family="Inter, system-ui, sans-serif"),
        align="right",
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=16, color=THEME["text"]), x=0.01),
            margin=dict(l=6, r=85, t=68, b=6),
            height=840,
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["bg"],
            clickmode="event+select",
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 1b. Sector bubble chart — risk/return scatter for a single sector
# ══════════════════════════════════════════════════════════════════════════════

def sector_bubble_chart(
    moments_df: pd.DataFrame,
    sector: str,
    color_metric: str = "daily_return",
    market_caps: Optional[pd.Series] = None,
    selected_ticker: Optional[str] = None,
) -> go.Figure:
    """
    Risk-return bubble chart for all stocks in a sector.
    x = mean return, y = variance, size = market cap, color = selected metric.
    The selected stock gets a large highlighted ring.
    """
    df = moments_df.copy()
    df.index.name = "ticker"
    df = df.reset_index()

    needed = ["mean", "variance"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    if color_metric not in df.columns:
        df[color_metric] = 0.0

    df["mean"]     = pd.to_numeric(df["mean"],     errors="coerce")
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce")
    df[color_metric] = pd.to_numeric(df[color_metric], errors="coerce")
    df = df.dropna(subset=["mean", "variance"]).copy()

    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout(
            title=dict(text=f"{sector} — No data", font=dict(size=14))
        ))
        return fig

    # Bubble size from market cap (if available)
    if market_caps is not None:
        df["mktcap"] = df["ticker"].map(market_caps)
        df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce").fillna(1e9)
    else:
        df["mktcap"] = 1e9

    # Normalize bubble size: sqrt(mktcap), scaled to [8, 45]
    raw_sz = np.sqrt(df["mktcap"].clip(lower=1e6))
    sz_min, sz_max = raw_sz.min(), raw_sz.max()
    if sz_max > sz_min:
        df["bubble_sz"] = 8 + 37 * (raw_sz - sz_min) / (sz_max - sz_min)
    else:
        df["bubble_sz"] = 20.0

    # Color range
    col_clean = df[color_metric].replace([np.inf, -np.inf], np.nan).dropna()
    vmax = max(abs(col_clean.quantile(0.05)), abs(col_clean.quantile(0.95)), 0.1) if len(col_clean) else 1.0
    metric_label = color_metric.replace("_", " ").title()
    suffix = _metric_suffix(color_metric)

    fig = go.Figure()

    # Non-selected stocks
    others = df[df["ticker"] != selected_ticker] if selected_ticker else df
    fig.add_trace(
        go.Scatter(
            x=others["mean"], y=others["variance"],
            mode="markers+text",
            text=others["ticker"],
            textposition="top center",
            textfont=dict(size=9, color=THEME["text_dim"]),
            marker=dict(
                size=others["bubble_sz"],
                color=others[color_metric],
                colorscale=[
                    [0.00, "#6b0f0f"], [0.25, "#e04040"],
                    [0.48, "#2a2f3b"], [0.52, "#2a2f3b"],
                    [0.75, "#1e7a3c"], [1.00, "#39d353"],
                ],
                cmin=-vmax, cmax=vmax,
                showscale=True,
                colorbar=dict(
                    title=dict(text=metric_label, side="right"),
                    thickness=12, len=0.7,
                    tickformat="+.2f",
                    tickfont=dict(size=9, color=THEME["text_dim"]),
                ),
                line=dict(color="rgba(255,255,255,0.12)", width=0.8),
                opacity=0.82,
            ),
            customdata=np.column_stack([
                others["ticker"],
                others[color_metric].round(4),
                others["mean"].round(4),
                others["variance"].round(4),
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{metric_label}: %{{customdata[1]:+.4f}}{suffix}<br>"
                "Mean return: %{customdata[2]:+.4f}<br>"
                "Variance: %{customdata[3]:.4f}<extra></extra>"
            ),
            name="Sector stocks",
            showlegend=False,
        )
    )

    # Highlighted selected stock (big ring)
    if selected_ticker and selected_ticker in df["ticker"].values:
        sel = df[df["ticker"] == selected_ticker].iloc[0]
        sel_color = _metric_to_hex(float(sel[color_metric]), vmax)
        fig.add_trace(
            go.Scatter(
                x=[sel["mean"]], y=[sel["variance"]],
                mode="markers+text",
                text=[selected_ticker],
                textposition="top center",
                textfont=dict(size=13, color=THEME["text"], family="Inter, system-ui, sans-serif"),
                marker=dict(
                    size=sel["bubble_sz"] * 1.5,
                    color=sel_color,
                    line=dict(color=THEME["blue"], width=3),
                    opacity=1.0,
                    symbol="circle",
                ),
                name=selected_ticker,
                showlegend=False,
                hovertemplate=(
                    f"<b>{selected_ticker}</b> ← selected<br>"
                    f"{metric_label}: {sel[color_metric]:+.4f}{suffix}<br>"
                    f"Mean: {sel['mean']:+.4f}<br>"
                    f"Variance: {sel['variance']:.4f}<extra></extra>"
                ),
            )
        )

    # Mean & variance reference lines
    x_med = float(df["mean"].median())
    y_med = float(df["variance"].median())
    fig.add_vline(x=x_med, line_dash="dot", line_color=_rgba(THEME["text_dim"], 0.5), line_width=1)
    fig.add_hline(y=y_med, line_dash="dot", line_color=_rgba(THEME["text_dim"], 0.5), line_width=1)

    # Quadrant labels
    x_rng = df["mean"].max() - df["mean"].min()
    y_rng = df["variance"].max() - df["variance"].min()
    for qtext, qx, qy in [
        ("High return · High risk", x_med + x_rng*0.35, y_med + y_rng*0.35),
        ("High return · Low risk",  x_med + x_rng*0.35, y_med - y_rng*0.30),
        ("Low return · High risk",  x_med - x_rng*0.40, y_med + y_rng*0.35),
        ("Low return · Low risk",   x_med - x_rng*0.40, y_med - y_rng*0.30),
    ]:
        fig.add_annotation(
            x=qx, y=qy, text=qtext,
            showarrow=False,
            font=dict(size=8, color=_rgba(THEME["text_dim"], 0.45)),
            xanchor="center",
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"📍 {sector} — Risk–Return Landscape  ({len(df)} stocks)",
                font=dict(size=14, color=THEME["text"]),
            ),
            xaxis=dict(title="Mean Daily Return (%)", showgrid=True,
                       gridcolor=_rgba(THEME["border"], 0.5), zeroline=True,
                       zerolinecolor=_rgba(THEME["border"], 0.8)),
            yaxis=dict(title="Return Variance (%²)", showgrid=True,
                       gridcolor=_rgba(THEME["border"], 0.5)),
            height=420,
            margin=dict(l=60, r=80, t=55, b=50),
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
    2×3 grid of premium KDE density plots, one for each of the six moment statistics.
    Features: gradient fills, IQR shading, percentile vlines, sector overlay.
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[MOMENT_LABELS[c] for c in MOMENT_COLS],
        vertical_spacing=0.16,
        horizontal_spacing=0.09,
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
        color = palette[idx]

        # Outer gradient fill (wide, low opacity)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([d, np.zeros(len(d))]),
                fill="toself",
                fillcolor=_rgba(color, 0.08),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )

        # Main KDE line with inner fill
        fig.add_trace(
            go.Scatter(
                x=x,
                y=d,
                mode="lines",
                name=MOMENT_LABELS[col],
                line=dict(color=color, width=2.5),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.18),
                showlegend=False,
                hovertemplate=(
                    f"<b>{MOMENT_LABELS[col]}</b><br>"
                    "Value: %{x:.3f}<br>"
                    "Density: %{y:.4f}<extra></extra>"
                ),
            ),
            row=r,
            col=c,
        )

        # IQR shading (25th – 75th percentile band)
        q25, q50, q75 = float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75))
        iqr_mask = (x >= q25) & (x <= q75)
        if iqr_mask.sum() > 1:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x[iqr_mask], x[iqr_mask][::-1]]),
                    y=np.concatenate([d[iqr_mask], np.zeros(iqr_mask.sum())]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.30),
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name="IQR",
                ),
                row=r,
                col=c,
            )

        # Median vline (solid, bright)
        fig.add_vline(
            x=q50, line_dash="solid", line_color=color, line_width=1.5,
            row=r, col=c,
            annotation_text=f"μ½={q50:.2f}",
            annotation_font_size=9,
            annotation_font_color=color,
            annotation_position="top right",
        )

        # Sector highlight overlay
        if highlight_sector and sector_map:
            tickers_in_sector = [t for t, s in sector_map.items() if s == highlight_sector]
            arr_sec = moments.loc[moments.index.isin(tickers_in_sector), col].dropna().values
            if len(arr_sec) >= 5:
                xs, ds = estimate_kde(arr_sec, x_range=(float(x.min()), float(x.max())))
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ds,
                        mode="lines",
                        name=highlight_sector if idx == 0 else highlight_sector,
                        line=dict(color=THEME["red_bright"], width=2, dash="dash"),
                        showlegend=(idx == 0),
                        hovertemplate=(
                            f"<b>{highlight_sector}</b><br>"
                            "Value: %{x:.3f}<br>"
                            "Density: %{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )

    # Style subplot title annotations (they are figure annotations[0..5])
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = THEME["text_dim"]

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="Nonparametric Density Estimates — Stock-Level Moments",
                font=dict(size=15, color=THEME["text"]),
            ),
            height=560,
            showlegend=bool(highlight_sector),
            legend=dict(
                orientation="h", y=1.04, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)", bordercolor=THEME["border"],
            ),
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=_rgba(THEME["border"], 0.5), zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rolling index moments (Q3)
# ══════════════════════════════════════════════════════════════════════════════

def rolling_moments_plot(
    rolling_df: pd.DataFrame,
    title: str = "30-Day Rolling Moments — S&P 500 Index Returns",
) -> go.Figure:
    """
    Premium four-panel time series.
    Features: COVID shading, area fills, zero-lines, range selector.
    """
    moment_config = [
        ("mean",     "Rolling Mean (%)",          THEME["blue"],         (1, 1)),
        ("variance", "Rolling Variance (%²)",     THEME["orange"],       (1, 2)),
        ("skewness", "Rolling Skewness",           THEME["purple"],       (2, 1)),
        ("kurtosis", "Rolling Excess Kurtosis",    THEME["green_bright"], (2, 2)),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m[1] for m in moment_config],
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.09,
    )

    # COVID crash window (approx)
    covid_start = "2020-02-19"
    covid_end   = "2020-04-23"

    for col, label, color, (r, c) in moment_config:
        if col not in rolling_df.columns:
            continue
        s = rolling_df[col].dropna()
        if len(s) == 0:
            continue

        # Soft fill between line and zero
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values,
                mode="lines",
                name=label,
                line=dict(color=color, width=1.8),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.10),
                hovertemplate=f"%{{x|%b %Y}}<br>{label}: %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
            row=r, col=c,
        )

        # Zero reference line
        fig.add_hline(
            y=0, line_dash="dot", line_color=_rgba(THEME["text_dim"], 0.6),
            line_width=1, row=r, col=c,
        )

        # COVID shading rectangle
        try:
            idx = s.index
            if hasattr(idx, "min") and pd.Timestamp(covid_start) >= idx.min():
                fig.add_vrect(
                    x0=covid_start, x1=covid_end,
                    fillcolor=_rgba(THEME["red_bright"], 0.10),
                    line_width=0,
                    row=r, col=c,
                )
        except Exception:
            pass

    # COVID label annotation (only once, in top-left panel, paper coords to avoid yref issues)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.15, y=0.98,
        text="🦠 COVID crash",
        showarrow=False,
        font=dict(size=9, color=_rgba(THEME["red_bright"], 0.85)),
        align="left",
        bgcolor=_rgba(THEME["red_bright"], 0.08),
        bordercolor=_rgba(THEME["red_bright"], 0.3),
        borderpad=4,
        borderwidth=1,
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15, color=THEME["text"])),
            height=520,
            showlegend=False,
        )
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=THEME["border"],
        tickformat="%Y",
        rangeslider=dict(visible=False),
    )
    fig.update_yaxes(showgrid=True, gridcolor=_rgba(THEME["border"], 0.5), zeroline=False)

    # Style subplot title annotations
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=THEME["text_dim"])

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
    Premium 3D density surface with a custom plasma-dark colorscale,
    wireframe contours, and carefully chosen camera angle.
    """
    date_nums = [(d - dates[0]).days for d in dates]

    # Custom colorscale: dark purple → blue → teal → yellow (like plasma reversed)
    plasma_dark = [
        [0.00, "#0d0221"],
        [0.15, "#1c1464"],
        [0.30, "#388bfd"],
        [0.50, "#2ea043"],
        [0.70, "#d29922"],
        [0.85, "#ffa657"],
        [1.00, "#f85149"],
    ]

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_grid,
                y=date_nums,
                z=Z,
                colorscale=plasma_dark,
                opacity=0.90,
                showscale=True,
                lighting=dict(
                    ambient=0.7, diffuse=0.8,
                    roughness=0.5, specular=0.3, fresnel=0.2,
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="rgba(255,255,255,0.2)",
                           project_z=True, width=1),
                ),
                colorbar=dict(
                    title=dict(text="Density", side="right", font=dict(size=11, color=THEME["text_dim"])),
                    thickness=14, len=0.65,
                    tickfont=dict(size=9, color=THEME["text_dim"]),
                ),
                hovertemplate=(
                    "Return: %{x:.2f}%<br>"
                    "Density: %{z:.5f}<extra></extra>"
                ),
            )
        ]
    )

    n_ticks = min(10, len(dates))
    step = max(1, len(dates) // n_ticks)
    tick_vals  = [date_nums[i] for i in range(0, len(dates), step)]
    tick_texts = [str(dates[i])[:7] for i in range(0, len(dates), step)]  # "YYYY-MM"

    axis_style = dict(
        gridcolor=_rgba(THEME["border"], 0.6),
        backgroundcolor=THEME["bg_secondary"],
        showbackground=True,
        linecolor=THEME["border"],
        tickfont=dict(size=9, color=THEME["text_dim"]),
        title_font=dict(size=11, color=THEME["text_dim"]),
        zeroline=False,
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15, color=THEME["text"])),
            height=640,
            margin=dict(l=0, r=0, t=60, b=0),
            scene=dict(
                xaxis=dict(**axis_style, title="Daily Return (%)"),
                yaxis=dict(
                    **axis_style,
                    title="Date",
                    tickvals=tick_vals,
                    ticktext=tick_texts,
                ),
                zaxis=dict(**axis_style, title="Density"),
                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.9)),
                bgcolor=THEME["bg"],
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=2.0, z=0.6),
            ),
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
    Premium stacked time-series: cross-sectional variance, skewness, kurtosis.
    Features: area fills, COVID shading, rolling-average smoothing line.
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Cross-Sectional Variance (%²)",
            "Cross-Sectional Skewness",
            "Cross-Sectional Excess Kurtosis",
        ],
        shared_xaxes=True,
        vertical_spacing=0.07,
    )

    cfg = [
        ("variance", "Cross-Sectional Variance (%²)", THEME["orange"],       1),
        ("skewness", "Cross-Sectional Skewness",       THEME["purple"],       2),
        ("kurtosis", "Excess Kurtosis",                THEME["green_bright"], 3),
    ]

    covid_start, covid_end = "2020-02-19", "2020-04-23"

    for col, label, color, row in cfg:
        if col not in xsec_df.columns:
            continue
        s = xsec_df[col].dropna()
        if len(s) == 0:
            continue

        # Raw thin line (very transparent)
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values,
                mode="lines",
                line=dict(color=_rgba(color, 0.35), width=0.8),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=1,
        )

        # 10-day rolling average as the bold line
        smooth = s.rolling(10, min_periods=3).mean()
        fig.add_trace(
            go.Scatter(
                x=smooth.index, y=smooth.values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.10),
                hovertemplate=f"%{{x|%b %Y}}<br>{label}: %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
            row=row, col=1,
        )

        # Zero reference
        fig.add_hline(y=0, line_dash="dot",
                      line_color=_rgba(THEME["text_dim"], 0.5), line_width=0.8,
                      row=row, col=1)

        # COVID shading
        try:
            if pd.Timestamp(covid_start) >= s.index.min():
                fig.add_vrect(
                    x0=covid_start, x1=covid_end,
                    fillcolor=_rgba(THEME["red_bright"], 0.08),
                    line_width=0, row=row, col=1,
                )
        except Exception:
            pass

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15, color=THEME["text"])),
            height=620,
            showlegend=False,
        )
    )
    fig.update_xaxes(showgrid=False, tickformat="%Y")
    fig.update_yaxes(showgrid=True, gridcolor=_rgba(THEME["border"], 0.5), zeroline=False)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=THEME["text_dim"])
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
    Premium scatterplot with OLS line, R² annotation, and confidence band.
    """
    df = moments[[x_metric, y_metric]].dropna().copy()
    df.index.name = "ticker"
    df = df.reset_index()

    if sector_map and color_by_sector:
        df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
    else:
        df["sector"] = "All Stocks"

    x_arr = df[x_metric].values
    y_arr = df[y_metric].values
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))

    fig = go.Figure()

    # Scatter per sector
    for sec in sorted(df["sector"].unique()):
        mask = df["sector"] == sec
        sc = df[mask]
        col = SECTOR_COLORS.get(sec, THEME["blue"]) if sec != "All Stocks" else THEME["blue"]
        fig.add_trace(
            go.Scatter(
                x=sc[x_metric], y=sc[y_metric],
                mode="markers",
                name=sec,
                marker=dict(
                    size=6, color=col, opacity=0.72,
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5),
                ),
                text=sc["ticker"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{MOMENT_LABELS.get(x_metric, x_metric)}: %{{x:.3f}}<br>"
                    f"{MOMENT_LABELS.get(y_metric, y_metric)}: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

    # OLS line + confidence band
    if valid.sum() > 10:
        xv, yv = x_arr[valid], y_arr[valid]
        coef = np.polyfit(xv, yv, 1)
        x_line = np.linspace(xv.min(), xv.max(), 300)
        y_line = np.polyval(coef, x_line)

        # R² calculation
        y_hat = np.polyval(coef, xv)
        ss_res = np.sum((yv - y_hat) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # OLS confidence band (±1 SE of the fit)
        n = len(xv)
        se = np.sqrt(ss_res / max(n - 2, 1))
        x_mean = xv.mean()
        leverage = np.sqrt(1/n + (x_line - x_mean)**2 / max(np.sum((xv - x_mean)**2), 1e-12))
        band = 1.96 * se * leverage

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([y_line + band, (y_line - band)[::-1]]),
                fill="toself",
                fillcolor=_rgba(THEME["yellow"], 0.10),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="95% CI band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color=THEME["yellow"], width=2, dash="dash"),
                name=f"OLS  slope={coef[0]:.3f}  R²={r2:.3f}",
            )
        )

        # R² annotation
        fig.add_annotation(
            text=f"<b>OLS fit</b><br>slope = {coef[0]:.4f}<br>R² = {r2:.4f}",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=10, color=THEME["text_dim"]),
            align="left",
            bgcolor=THEME["bg_card"],
            bordercolor=THEME["border"],
            borderpad=8,
            borderwidth=1,
        )

    xl = MOMENT_LABELS.get(x_metric, x_metric)
    yl = MOMENT_LABELS.get(y_metric, y_metric)
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=title or f"{xl}  vs.  {yl}",
                font=dict(size=14, color=THEME["text"]),
            ),
            xaxis=dict(title=xl, showgrid=True, gridcolor=_rgba(THEME["border"], 0.5)),
            yaxis=dict(title=yl, showgrid=True, gridcolor=_rgba(THEME["border"], 0.5)),
            height=500,
            legend=dict(
                orientation="v", x=1.02, y=1,
                bgcolor=THEME["bg_card"],
                bordercolor=THEME["border"],
                borderwidth=1,
                font=dict(size=10),
            ),
        )
    )
    return fig


def all_pairs_subplot(
    moments: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Premium 15-panel grid — C(6,2)=15 pairwise scatterplots with OLS trend lines.
    """
    from itertools import combinations

    pairs = list(combinations(MOMENT_COLS, 2))
    n_cols = 3
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        vertical_spacing=0.06,
        horizontal_spacing=0.07,
    )

    if sector_map:
        sectors_in_data = [sector_map.get(t, "Unknown") for t in moments.index]
        color_map_vals = [SECTOR_COLORS.get(s, THEME["text_dim"]) for s in sectors_in_data]
    else:
        color_map_vals = [THEME["blue"]] * len(moments)

    for k, (xc, yc) in enumerate(pairs):
        row = k // n_cols + 1
        col = k % n_cols + 1
        sub = moments[[xc, yc]].dropna()
        if len(sub) == 0:
            continue

        # Colors aligned to sub.index
        if sector_map:
            pt_colors = []
            for t in sub.index:
                try:
                    pt_colors.append(color_map_vals[moments.index.get_loc(t)])
                except Exception:
                    pt_colors.append(THEME["text_dim"])
        else:
            pt_colors = [THEME["blue"]] * len(sub)

        fig.add_trace(
            go.Scatter(
                x=sub[xc].values,
                y=sub[yc].values,
                mode="markers",
                marker=dict(
                    size=4, color=pt_colors, opacity=0.65,
                    line=dict(color="rgba(255,255,255,0.08)", width=0.4),
                ),
                text=sub.index.tolist(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + f"{MOMENT_LABELS.get(xc, xc)}: %{{x:.3f}}<br>"
                    + f"{MOMENT_LABELS.get(yc, yc)}: %{{y:.3f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )

        # Lightweight OLS trend line per panel
        xv, yv = sub[xc].values, sub[yc].values
        valid = ~(np.isnan(xv) | np.isnan(yv))
        if valid.sum() > 5:
            coef = np.polyfit(xv[valid], yv[valid], 1)
            xl = np.linspace(xv[valid].min(), xv[valid].max(), 80)
            fig.add_trace(
                go.Scatter(
                    x=xl, y=np.polyval(coef, xl),
                    mode="lines",
                    line=dict(color=_rgba(THEME["yellow"], 0.8), width=1.2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )

        fig.update_xaxes(
            title_text=xc[:4],
            title_font=dict(size=8, color=THEME["text_dim"]),
            tickfont=dict(size=7),
            showgrid=True, gridcolor=_rgba(THEME["border"], 0.4),
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text=yc[:4],
            title_font=dict(size=8, color=THEME["text_dim"]),
            tickfont=dict(size=7),
            showgrid=True, gridcolor=_rgba(THEME["border"], 0.4),
            row=row, col=col,
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="All Pairwise Metric Scatterplots  (dashed = OLS trend)",
                font=dict(size=14, color=THEME["text"]),
            ),
            height=max(480, n_rows * 220),
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
    Premium KDE + histogram of stock-index correlations with sector overlays.
    Features: gradient area fill, percentile bands, sector comparison, rich annotations.
    """
    fig = go.Figure()

    arr = corr_series.dropna().values
    if len(arr) == 0:
        fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=15)), height=300))
        return fig

    x, d = estimate_kde(arr, x_range=(-0.3, 1.0))
    blue = THEME["blue"]

    # Background histogram (very light)
    fig.add_trace(
        go.Histogram(
            x=arr,
            nbinsx=60,
            name="All stocks",
            histnorm="probability density",
            marker=dict(color=_rgba(blue, 0.25), line=dict(color=_rgba(blue, 0.4), width=0.5)),
            hovertemplate="Corr bin: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
        )
    )

    # Outer soft fill
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([d, np.zeros(len(d))]),
            fill="toself",
            fillcolor=_rgba(blue, 0.07),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Main KDE line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=d,
            mode="lines",
            name="KDE — all stocks",
            line=dict(color=blue, width=2.5),
            fill="tozeroy",
            fillcolor=_rgba(blue, 0.13),
            hovertemplate="Corr: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        )
    )

    # Sector KDE overlays
    if sector_map:
        sector_set = sorted({s for s in sector_map.values() if s != "Unknown"})
        for sec in sector_set[:10]:
            tickers = [t for t, s in sector_map.items() if s == sec and t in corr_series.index]
            arr_s = corr_series.loc[corr_series.index.isin(tickers)].dropna().values
            if len(arr_s) < 5:
                continue
            xs, ds = estimate_kde(arr_s, x_range=(-0.3, 1.0))
            sec_color = SECTOR_COLORS.get(sec, "#888888")
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ds,
                    mode="lines",
                    name=sec,
                    line=dict(color=sec_color, width=1.6, dash="dot"),
                    opacity=0.85,
                    hovertemplate=f"<b>{sec}</b><br>Corr: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
                )
            )

    # Percentile vlines
    q10, q25, q50, q75, q90 = [float(np.percentile(arr, q)) for q in (10, 25, 50, 75, 90)]
    for val, label, col, dash in [
        (q50, f"Median\n{q50:.3f}", THEME["yellow"], "dash"),
        (q25, f"P25 {q25:.2f}", THEME["text_dim"], "dot"),
        (q75, f"P75 {q75:.2f}", THEME["text_dim"], "dot"),
    ]:
        fig.add_vline(
            x=val, line_dash=dash, line_color=col, line_width=1.5,
            annotation_text=label.split("\n")[0],
            annotation_font_color=col,
            annotation_font_size=10,
        )

    # IQR shading
    iqr_mask = (x >= q25) & (x <= q75)
    if iqr_mask.sum() > 1:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x[iqr_mask], x[iqr_mask][::-1]]),
                y=np.concatenate([d[iqr_mask], np.zeros(iqr_mask.sum())]),
                fill="toself",
                fillcolor=_rgba(blue, 0.22),
                line=dict(width=0),
                showlegend=False,
                name="IQR (25–75)",
                hoverinfo="skip",
            )
        )

    # Annotation box with stats
    n_positive = (arr > 0.5).sum()
    stats_text = (
        f"n={len(arr)} stocks<br>"
        f"Mean: {arr.mean():.3f} | Std: {arr.std():.3f}<br>"
        f"Corr > 0.5: {n_positive} ({100*n_positive/len(arr):.0f}%)"
    )
    fig.add_annotation(
        text=stats_text, xref="paper", yref="paper",
        x=0.01, y=0.97, showarrow=False,
        font=dict(size=10, color=THEME["text_dim"]),
        align="left",
        bgcolor=THEME["bg_card"],
        bordercolor=THEME["border"],
        borderpad=8,
        borderwidth=1,
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15, color=THEME["text"])),
            xaxis=dict(title="Pearson Correlation with ^GSPC", showgrid=True,
                       gridcolor=_rgba(THEME["border"], 0.5), zeroline=True,
                       zerolinecolor=THEME["border"], zerolinewidth=1),
            yaxis=dict(title="Density", showgrid=False),
            height=480,
            barmode="overlay",
            legend=dict(
                orientation="v", x=1.01, y=1,
                bgcolor=THEME["bg_card"], bordercolor=THEME["border"],
                borderwidth=1, font=dict(size=10),
            ),
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
    Advanced two-panel figure:
      Left  — Dual KDE comparison with IQR bands and statistical annotation
      Right — Ranked horizontal bar chart of within-sector correlations by sector
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Within- vs. Between-Sector Correlation Distribution",
            "Mean Within-Sector Correlation by Sector",
        ],
        column_widths=[0.54, 0.46],
        horizontal_spacing=0.10,
    )

    kde_cfg = [
        (within_corr.values,  "Within-sector",  THEME["green_bright"]),
        (between_corr.values, "Between-sector", THEME["red_bright"]),
    ]

    for arr_raw, name, color in kde_cfg:
        arr = arr_raw[~np.isnan(arr_raw)]
        if len(arr) < 5:
            continue
        x, d = estimate_kde(arr, x_range=(-0.2, 1.0))

        # Soft outer fill
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([d, np.zeros(len(d))]),
                fill="toself",
                fillcolor=_rgba(color, 0.07),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )
        # Main filled KDE
        fig.add_trace(
            go.Scatter(
                x=x, y=d,
                mode="lines",
                name=name,
                line=dict(color=color, width=2.5),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.18),
                hovertemplate=f"<b>{name}</b><br>Corr: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )
        # IQR shading
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr_mask = (x >= q25) & (x <= q75)
        if iqr_mask.sum() > 1:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x[iqr_mask], x[iqr_mask][::-1]]),
                    y=np.concatenate([d[iqr_mask], np.zeros(iqr_mask.sum())]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.30),
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )
        # Median line
        med = float(np.median(arr))
        fig.add_vline(
            x=med, line_dash="dash", line_color=color, line_width=1.8,
            row=1, col=1,
        )

    # Bar chart — sorted by value, colored by sector palette
    if not sector_avg_df.empty:
        sorted_df = sector_avg_df.sort_values("mean_within_corr", ascending=True)
        bar_colors = [
            SECTOR_COLORS.get(s, THEME["blue"])
            for s in sorted_df["sector"].values
        ]
        fig.add_trace(
            go.Bar(
                x=sorted_df["mean_within_corr"].values,
                y=sorted_df["sector"].values,
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    opacity=0.85,
                ),
                text=[f"{v:.3f}" for v in sorted_df["mean_within_corr"]],
                textposition="outside",
                textfont=dict(size=10, color=THEME["text_dim"]),
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Mean within-sector corr: %{x:.4f}<extra></extra>",
            ),
            row=1, col=2,
        )

    # Statistical annotation box (paper coords → never overlaps axes)
    p_welch = test_results.get("welch_p", 1.0)
    p_mw    = test_results.get("mw_p",    1.0)
    mean_w  = test_results.get("mean_within",  float("nan"))
    mean_b  = test_results.get("mean_between", float("nan"))
    tick_w  = "✅" if p_welch < 0.05 else "❌"
    tick_mw = "✅" if p_mw    < 0.05 else "❌"
    annot = (
        f"<b>Statistical Tests (H₀: within = between)</b><br>"
        f"Mean within: {mean_w:.4f}  |  Mean between: {mean_b:.4f}<br>"
        f"Welch t-test : p = {p_welch:.4f}  {tick_w}<br>"
        f"Mann–Whitney : p = {p_mw:.4f}  {tick_mw}"
    )
    fig.add_annotation(
        text=annot,
        xref="paper", yref="paper",
        x=0.01, y=-0.14,
        showarrow=False,
        font=dict(size=10, color=THEME["text_dim"]),
        align="left",
        bgcolor=THEME["bg_card"],
        bordercolor=THEME["blue"],
        borderpad=10,
        borderwidth=1,
    )

    fig.update_layout(
        **_base_layout(
            height=500,
            margin=dict(l=55, r=30, t=70, b=110),
            legend=dict(orientation="h", y=1.06, x=0.27, xanchor="center"),
        )
    )
    fig.update_xaxes(title_text="Pearson Correlation", showgrid=True,
                     gridcolor=_rgba(THEME["border"], 0.5), row=1, col=1)
    fig.update_yaxes(title_text="Density", showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="Mean Within-Sector Corr", showgrid=True,
                     gridcolor=_rgba(THEME["border"], 0.5), row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=2)

    # Style the subplot title annotations
    for ann in fig.layout.annotations:
        if ann.text and ann.xref == "paper":
            continue  # skip the stats box
        ann.font = dict(size=11, color=THEME["text_dim"])

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
    Premium KDE overlay comparing subperiods.
    Each period gets a distinct color with IQR shading and median vline.
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

        # Soft outer glow
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([d, np.zeros(len(d))]),
                fill="toself",
                fillcolor=_rgba(color, 0.06),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Main KDE
        fig.add_trace(
            go.Scatter(
                x=x, y=d,
                mode="lines",
                name=period,
                line=dict(color=color, width=2.5),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.14),
                hovertemplate=f"<b>{period}</b><br>{xlabel}: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
            )
        )
        # IQR shading
        q25, q50, q75 = (float(np.percentile(arr, q)) for q in (25, 50, 75))
        iqr_mask = (x >= q25) & (x <= q75)
        if iqr_mask.sum() > 1:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x[iqr_mask], x[iqr_mask][::-1]]),
                    y=np.concatenate([d[iqr_mask], np.zeros(iqr_mask.sum())]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.28),
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        # Median marker
        fig.add_vline(
            x=q50, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=f"{period[:3]} μ½={q50:.2f}",
            annotation_font_color=color,
            annotation_font_size=9,
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=title or f"Stability Comparison — {metric.replace('_', ' ').title()}",
                font=dict(size=15, color=THEME["text"]),
            ),
            xaxis=dict(title=xlabel, showgrid=True, gridcolor=_rgba(THEME["border"], 0.5)),
            yaxis=dict(title="Density", showgrid=False),
            height=440,
            legend=dict(
                orientation="h", y=1.04, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)", font=dict(size=11),
            ),
        )
    )
    return fig


def ks_results_heatmap(ks_df: pd.DataFrame, title: str = "KS Test Statistics") -> go.Figure:
    """
    Annotated heatmap of KS test statistics: rows = metrics, cols = period pairs.
    Cells with KS > 0.1 are labelled; significant cells (p<0.05) get a border effect.
    """
    if ks_df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(
            title=dict(text=title, font=dict(size=14))
        ))
        return fig

    pivot = ks_df.pivot_table(
        index="Metric",
        columns=["Period A", "Period B"],
        values="KS Stat",
        aggfunc="first",
    )
    pivot.columns = [f"{a} vs {b}" for a, b in pivot.columns]

    z = pivot.values
    text_matrix = [[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in z]

    # Custom sequential colorscale: dark → amber → red (higher KS = more different)
    ks_cs = [
        [0.00, "#1c2128"],
        [0.33, "#3b404c"],
        [0.55, THEME["yellow"]],
        [0.80, THEME["orange"]],
        [1.00, THEME["red_bright"]],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=ks_cs,
            zmin=0,
            zmax=0.30,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, color=THEME["text"]),
            colorbar=dict(
                title=dict(text="KS Stat", side="right", font=dict(size=11)),
                tickvals=[0, 0.10, 0.20, 0.30],
                ticktext=["0", "0.10", "0.20", "≥0.30"],
                thickness=14,
            ),
            hovertemplate="<b>%{y}</b><br>Periods: %{x}<br>KS Stat: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=14, color=THEME["text"])),
            height=max(280, 62 * len(pivot)),
            xaxis=dict(tickfont=dict(size=10, color=THEME["text_dim"]), showgrid=False),
            yaxis=dict(tickfont=dict(size=10, color=THEME["text_dim"]), showgrid=False),
        )
    )
    return fig


def risk_evolution_bar(risk_df: pd.DataFrame) -> go.Figure:
    """
    Premium grouped bar chart comparing key risk indicators across subperiods.
    """
    metric_cols = [
        c for c in risk_df.columns
        if "Ann. Vol" in c or "Kurtosis" in c or "Skewness" in c
    ]
    if not metric_cols:
        metric_cols = list(risk_df.columns)

    colors = [THEME["blue"], THEME["orange"], THEME["green_bright"]]

    fig = go.Figure()
    for i, period in enumerate(risk_df.index):
        color = colors[i % len(colors)]
        vals = [risk_df.loc[period, m] for m in metric_cols]
        fig.add_trace(
            go.Bar(
                name=period,
                x=metric_cols,
                y=vals,
                marker=dict(
                    color=color,
                    opacity=0.82,
                    line=dict(color=_rgba(color, 0.6), width=1),
                ),
                text=[f"{v:.3f}" if not np.isnan(v) else "" for v in vals],
                textposition="outside",
                textfont=dict(size=9, color=_rgba(color, 0.9)),
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Risk Indicators Across Subperiods", font=dict(size=15, color=THEME["text"])),
            barmode="group",
            xaxis=dict(title="Risk Metric", tickangle=-15, showgrid=False),
            yaxis=dict(title="Value", showgrid=True, gridcolor=_rgba(THEME["border"], 0.5)),
            height=440,
            legend=dict(
                orientation="h", y=1.04, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)", font=dict(size=11),
            ),
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
    Premium annotated correlation heatmap for a single sector.
    Strong correlations (|r|>0.65) are labelled; diagonal is hidden.
    """
    members = [t for t, s in sector_map.items() if s == sector and t in corr.index]
    members = members[:max_tickers]

    if len(members) < 2:
        fig = go.Figure()
        fig.update_layout(**_base_layout(
            title=dict(text=f"Not enough data for {sector}", font=dict(size=14))
        ))
        return fig

    sub_df = corr.loc[members, members].copy()
    # Mask diagonal (always 1.0 — not interesting). Use a writable copy.
    z = sub_df.values.copy()
    np.fill_diagonal(z, np.nan)

    # Build text matrix: show value only for strong correlations
    text_matrix = []
    for row in z:
        text_row = []
        for v in row:
            if np.isnan(v):
                text_row.append("")
            elif abs(v) >= 0.65:
                text_row.append(f"{v:.2f}")
            else:
                text_row.append("")
        text_matrix.append(text_row)

    # Custom diverging colorscale optimised for dark background
    custom_cs = [
        [0.00, "#7b1a1a"],   # deep red
        [0.20, "#c0392b"],
        [0.40, "#3b404c"],   # neutral dark
        [0.50, "#2a2f3b"],   # mid neutral
        [0.60, "#3b404c"],
        [0.80, "#1a6b3c"],
        [1.00, "#0d5c26"],   # deep green
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=members,
            y=members,
            colorscale=custom_cs,
            zmin=-1,
            zmax=1,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=8, color="rgba(230,237,243,0.85)"),
            colorbar=dict(
                title=dict(text="Pearson r", side="right", font=dict(size=11)),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0", "+0.5", "+1.0"],
                thickness=14,
                len=0.85,
            ),
            hovertemplate="<b>%{y}  ×  %{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
        )
    )

    n = len(members)
    tick_size = max(7, min(10, int(180 / n)))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"{sector} — Pairwise Correlation Heatmap  ({n} stocks)",
                font=dict(size=14, color=THEME["text"]),
            ),
            height=max(380, 16 * n),
            margin=dict(l=80, r=60, t=60, b=80),
            xaxis=dict(
                tickfont=dict(size=tick_size, color=THEME["text_dim"]),
                tickangle=-45,
                showgrid=False,
            ),
            yaxis=dict(
                tickfont=dict(size=tick_size, color=THEME["text_dim"]),
                showgrid=False,
                autorange="reversed",
            ),
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