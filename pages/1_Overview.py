"""
1_Overview.py
-------------
Advanced S&P 500 market map with interactive stock analytics panel.
Performance: treemap is cached — clicking a stock tile does NOT rebuild the map.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.config import THEME, SECTOR_CACHE, PLOTLY_LAYOUT_DEFAULTS
from src.utils import section_header, interpretation_box, page_css, page_header_html
from src.visualizations import treemap_market_map, sector_bubble_chart

st.set_page_config(page_title="Overview | FinEC", page_icon="🗺️", layout="wide")
st.markdown(page_css(), unsafe_allow_html=True)


# ── Data loaders (all cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    from src.preprocessing import load_cached_returns, get_stock_returns, get_index_returns
    r = load_cached_returns()
    if r is None:
        return None, None, None
    stock_ret = get_stock_returns(r)
    try:
        idx_ret = get_index_returns(r)
    except KeyError:
        idx_ret = pd.Series(dtype=float)
    return r, stock_ret, idx_ret


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def load_sector():
    if not SECTOR_CACHE.exists():
        return None
    return pd.read_parquet(SECTOR_CACHE)


@st.cache_data(show_spinner=False)
def load_moments(_stock_ret):
    from src.analytics import compute_stock_moments
    return compute_stock_moments(_stock_ret)


@st.cache_data(show_spinner=False)
def get_index_corr(_stock_ret, _idx_ret):
    from src.statistics import compute_stock_index_corr
    return compute_stock_index_corr(_stock_ret, _idx_ret)


@st.cache_data(show_spinner=False)
def build_display_df(_stock_ret, _idx_ret):
    from src.config import PRICES_CACHE
    from src.data_loader import compute_ytd_returns

    moments  = load_moments(_stock_ret)
    corr_s   = get_index_corr(_stock_ret, _idx_ret)
    latest_s = _stock_ret.ffill().iloc[-1].rename("daily_return") if len(_stock_ret) > 0 else pd.Series(dtype=float, name="daily_return")

    disp = moments.copy()
    disp["daily_return"] = latest_s.reindex(disp.index)
    disp["corr_sp500"]   = corr_s.reindex(disp.index)

    if PRICES_CACHE.exists():
        prices = pd.read_parquet(PRICES_CACHE)
        prices.index = pd.to_datetime(prices.index)
        ytd = compute_ytd_returns(prices, year=2025)
        disp["ytd_2025"] = ytd.reindex(disp.index)
    else:
        disp["ytd_2025"] = np.nan

    disp.index.name = "ticker"
    return disp


# ── Cached figure builders ────────────────────────────────────────────────────
# Using _ prefix on DataFrames/dicts so Streamlit skips hashing them
# (the explicit scalar keys are what really control cache validity).

@st.cache_data(show_spinner=False, max_entries=20)
def build_treemap_figure(
    color_metric: str,
    size_mode_value: str,
    sector_filter_tuple: tuple,   # hashable version of the multiselect
    _disp_df, _sector_map, _market_caps,
):
    pdf = _disp_df.copy()
    if sector_filter_tuple:
        keep = [t for t in pdf.index if (_sector_map or {}).get(t, "Unknown") in set(sector_filter_tuple)]
        pdf  = pdf.loc[pdf.index.intersection(keep)]
    return treemap_market_map(
        stock_data  = pdf,
        color_metric= color_metric,
        sector_map  = _sector_map or {},
        market_caps = _market_caps,
        title       = "S&P 500 Market Map",
        size_mode   = size_mode_value,
    )


@st.cache_data(show_spinner=False, max_entries=50)
def build_sector_zoom(
    sector: str,
    color_metric: str,
    size_mode_value: str,
    selected_ticker: str,
    _disp_df, _sector_map, _market_caps,
):
    """Sector bubble chart shown below the main map when a stock is selected."""
    keep = [t for t in _disp_df.index if (_sector_map or {}).get(t, "Unknown") == sector]
    pdf  = _disp_df.loc[_disp_df.index.intersection(keep)]
    if len(pdf) < 2:
        return None
    return sector_bubble_chart(
        moments_df      = pdf,
        sector          = sector,
        color_metric    = color_metric,
        market_caps     = _market_caps,
        selected_ticker = selected_ticker,
    )


@st.cache_data(show_spinner=False)
def compute_rolling_for_stock(series: pd.Series, window: int = 30):
    s = series.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["mean", "variance", "skewness", "kurtosis"])
    out = pd.DataFrame(index=s.index)
    out["mean"]     = s.rolling(window).mean()
    out["variance"] = s.rolling(window).var()
    out["skewness"] = s.rolling(window).skew()
    out["kurtosis"] = s.rolling(window).kurt()
    return out


# ── Per-stock chart helpers ───────────────────────────────────────────────────

def make_stock_timeseries_figure(ret_s, idx_ret, ticker):
    cum_stock = ret_s.fillna(0).cumsum()
    cum_idx   = idx_ret.reindex(ret_s.index).fillna(0).cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_stock.index, y=cum_stock.values,
        mode="lines", name=ticker,
        line=dict(color=THEME["blue"], width=2.2),
        hovertemplate="%{x|%Y-%m-%d}<br>Cumulative: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cum_idx.index, y=cum_idx.values,
        mode="lines", name="^GSPC",
        line=dict(color=THEME["text_dim"], width=1.6, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>^GSPC cumulative: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**dict(
        PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"{ticker} vs ^GSPC — Cumulative Log Returns", font=dict(size=13)),
        xaxis_title="Date", yaxis_title="Cumulative return (%)",
        height=320, showlegend=True,
        legend=dict(orientation="h", y=1.05, x=0),
    ))
    return fig


def make_rolling_detail_figure(rolling_df, ticker):
    colors = {"mean": THEME["blue"], "variance": THEME["orange"],
               "skewness": THEME["purple"], "kurtosis": THEME["green_bright"]}
    fig = go.Figure()
    for col in ["mean", "variance", "skewness", "kurtosis"]:
        if col in rolling_df.columns:
            fig.add_trace(go.Scatter(
                x=rolling_df.index, y=rolling_df[col],
                mode="lines", name=col.title(),
                line=dict(width=1.8, color=colors[col]),
                hovertemplate=f"%{{x|%b %Y}}<br>{col}: %{{y:.4f}}<extra></extra>",
            ))
    fig.update_layout(**dict(
        PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"{ticker} — 30-Day Rolling Moments", font=dict(size=13)),
        xaxis_title="Date", yaxis_title="Value",
        height=300, showlegend=True,
        legend=dict(orientation="h", y=1.05, x=0),
    ))
    return fig


def make_return_hist_kde(ret_s, ticker):
    from src.analytics import estimate_kde
    arr = ret_s.dropna().values
    fig = go.Figure()
    if len(arr) >= 5:
        x, d = estimate_kde(arr)
        fig.add_trace(go.Histogram(
            x=arr, nbinsx=60, histnorm="probability density",
            marker=dict(color=f"rgba(56,139,253,0.25)",
                        line=dict(color=f"rgba(56,139,253,0.5)", width=0.5)),
            name="Histogram",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=d, mode="lines",
            line=dict(color=THEME["blue"], width=2.3), name="KDE",
            fill="tozeroy", fillcolor="rgba(56,139,253,0.10)",
        ))
        med = float(np.median(arr))
        fig.add_vline(x=med, line_dash="dash", line_color=THEME["yellow"],
                      annotation_text=f"Median={med:.3f}",
                      annotation_font_color=THEME["yellow"])
    fig.update_layout(**dict(
        PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"{ticker} — Return Distribution", font=dict(size=13)),
        xaxis_title="Daily return (%)", yaxis_title="Density",
        height=300, barmode="overlay", showlegend=False,
    ))
    return fig


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(page_header_html(
    "Market Map",
    "Interactive S&P 500 market map — tile area = market cap · color = selected metric · click any tile to zoom into sector & open stock analytics",
    "🗺️"
), unsafe_allow_html=True)

returns, stock_ret, idx_ret = load_data()
if returns is None:
    st.warning("⚠️ No data cached. Go to the **Data Pipeline** page first.")
    st.stop()

sector_df  = load_sector()
sector_map : dict = {}
market_caps = None
if sector_df is not None:
    sector_map   = dict(zip(sector_df.index, sector_df["sector"].fillna("Unknown")))
    market_caps  = sector_df["market_cap"]

disp_df = build_display_df(stock_ret, idx_ret)

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🗺️ Market Map")
    color_metric = st.selectbox(
        "Color tiles by",
        ["daily_return","ytd_2025","mean","variance","skewness",
         "kurtosis","corr_sp500","p1","p99"],
        format_func=lambda x: {
            "daily_return": "Latest Daily Return",
            "ytd_2025":     "2025 YTD Return",
            "mean":         "Mean Return",
            "variance":     "Variance",
            "skewness":     "Skewness",
            "kurtosis":     "Kurtosis",
            "corr_sp500":   "Correlation with ^GSPC",
            "p1":           "1st Percentile",
            "p99":          "99th Percentile",
        }.get(x, x),
        index=0,
    )
    size_mode       = st.radio("Tile area based on",
                               ["Market cap", "Absolute metric magnitude"], index=0)
    size_mode_value = "market_cap" if size_mode == "Market cap" else "metric_abs"

    sector_choices  = sorted(set(sector_map.values())) if sector_map else []
    sector_filter   = st.multiselect("Filter sectors", sector_choices, default=[])

    manual_ticker = st.selectbox(
        "Jump to stock",
        [""] + sorted(stock_ret.columns.tolist()),
        index=0,
    )

# ── Build (or retrieve cached) treemap ───────────────────────────────────────
# This figure is NOT rebuilt when a stock is clicked — only when the controls above change.

with st.spinner("Building market map…"):
    fig_map = build_treemap_figure(
        color_metric,
        size_mode_value,
        tuple(sorted(sector_filter)),
        disp_df, sector_map, market_caps,
    )

map_event = st.plotly_chart(
    fig_map,
    use_container_width=True,
    on_select="rerun",
    selection_mode=["points"],
    key="market_map",
)

st.caption("Click a stock tile to open its analytics panel · or use **Jump to stock** in the sidebar.")

# ── Resolve selected ticker ───────────────────────────────────────────────────

selected_ticker = st.session_state.get("selected_ticker", None)

# Native Plotly click event — points[0] is a plain dict, must use .get()
if map_event and map_event.selection and map_event.selection.points:
    pt = map_event.selection.points[0]   # plain dict, NOT AttributeDictionary

    candidate = None

    # 1. "label" key — treemap leaf label = ticker symbol (e.g. "AAPL")
    label_val = pt.get("label", "")
    if label_val and str(label_val) in stock_ret.columns:
        candidate = str(label_val)

    # 2. "id" key — our ids are "stock::AAPL"; strip prefix
    if not candidate:
        id_val = str(pt.get("id", "") or "")
        if id_val.startswith("stock::"):
            candidate = id_val[len("stock::"):]

    # 3. customdata[0] — always contains ticker for leaf nodes
    if not candidate:
        cd = pt.get("customdata") or []
        if len(cd) > 0 and cd[0]:
            candidate = str(cd[0])

    if candidate and candidate in stock_ret.columns:
        st.session_state["selected_ticker"] = candidate
        selected_ticker = candidate

if manual_ticker:
    st.session_state["selected_ticker"] = manual_ticker
    selected_ticker = manual_ticker

if selected_ticker is None and len(disp_df) > 0:
    selected_ticker = disp_df["daily_return"].abs().sort_values(ascending=False).index[0]
    st.session_state["selected_ticker"] = selected_ticker

# ── Sector zoom (shown immediately below the map) ─────────────────────────────

if selected_ticker and selected_ticker in stock_ret.columns:
    selected_sector = sector_map.get(selected_ticker, "Unknown")

    with st.spinner(""):
        fig_zoom = build_sector_zoom(
            selected_sector, color_metric, size_mode_value,
            selected_ticker, disp_df, sector_map, market_caps,
        )

    if fig_zoom is not None:
        st.markdown(
            section_header(
                f"Sector Zoom — {selected_sector}",
                f"Risk–return landscape for all {selected_sector} stocks · selected: {selected_ticker}",
            ),
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_zoom, use_container_width=True)

# ── Stock analytics panel (isolated with st.fragment to prevent map rebuild) ──

@st.fragment
def analytics_panel():
    sel = st.session_state.get("selected_ticker", None)
    if not sel or sel not in stock_ret.columns:
        st.info("Select a stock from the map or from the sidebar.")
        return

    ret_s    = stock_ret[sel].dropna()
    rolling_s = compute_rolling_for_stock(ret_s, window=30)

    common_idx  = ret_s.index.intersection(idx_ret.index)
    ret_common  = ret_s.reindex(common_idx)
    idx_common  = idx_ret.reindex(common_idx)

    sec = sector_map.get(sel, "Unknown")
    row = disp_df.loc[sel]

    st.markdown("---")
    st.markdown(
        section_header(
            f"Stock Analytics — {sel}",
            f"Sector: {sec} · all metrics linked to the homework questions",
        ),
        unsafe_allow_html=True,
    )

    # ── Key metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def _safe(key): return float(row.get(key, np.nan))

    with c1: st.metric("Latest Return",   f"{_safe('daily_return'):+.2f}%")
    with c2: st.metric("Mean",            f"{_safe('mean'):.4f}")
    with c3: st.metric("Variance",        f"{_safe('variance'):.4f}")
    with c4: st.metric("Skewness",        f"{_safe('skewness'):.4f}")
    with c5: st.metric("Kurtosis",        f"{_safe('kurtosis'):.4f}")
    with c6: st.metric("Corr. to ^GSPC",  f"{_safe('corr_sp500'):.4f}")

    # ── Three analytics tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Moments & Distribution", "Time Dynamics", "Market Dependence"])

    with tab1:
        left, right = st.columns([1, 2])
        with left:
            summary = pd.DataFrame({
                "Metric": ["Mean", "Variance", "Skewness", "Kurtosis", "1st pct", "99th pct"],
                "Value":  [_safe("mean"), _safe("variance"), _safe("skewness"),
                           _safe("kurtosis"), _safe("p1"), _safe("p99")],
            })
            st.dataframe(summary.style.format({"Value": "{:.5f}"}),
                         use_container_width=True, hide_index=True)
        with right:
            st.plotly_chart(make_return_hist_kde(ret_s, sel), use_container_width=True)

    with tab2:
        st.plotly_chart(make_stock_timeseries_figure(ret_common, idx_common, sel),
                        use_container_width=True)
        st.plotly_chart(make_rolling_detail_figure(rolling_s, sel), use_container_width=True)

    with tab3:
        beta_like = np.nan
        if len(ret_common) > 20 and np.var(idx_common, ddof=1) > 0:
            beta_like = np.cov(ret_common, idx_common)[0, 1] / np.var(idx_common, ddof=1)

        risk_tbl = pd.DataFrame({
            "Indicator": ["Correlation to ^GSPC", "Beta-like slope", "Daily observations"],
            "Value":     [_safe("corr_sp500"), beta_like, ret_s.shape[0]],
        })
        st.dataframe(risk_tbl.style.format({"Value": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

        st.markdown(
            interpretation_box(
                f"<strong>{sel}</strong> belongs to the <strong>{sec}</strong> sector. "
                "Its unconditional moments, rolling dynamics, and correlation with the "
                "S&P 500 index link directly to the econometric analysis in this dashboard."
            ),
            unsafe_allow_html=True,
        )


analytics_panel()
