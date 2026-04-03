"""
1_Overview.py
-------------
Advanced Finviz-style market map overview with real stock click interactions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from streamlit_plotly_events import plotly_events

from src.config import THEME, SECTOR_CACHE, PLOTLY_LAYOUT_DEFAULTS
from src.utils import section_header, interpretation_box
from src.visualizations import treemap_market_map


st.set_page_config(page_title="Overview | FinEC", page_icon="🗺️", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color:{THEME['bg']}; color:{THEME['text']}; }}
    [data-testid="stSidebar"] {{ background-color:{THEME['bg_secondary']}; border-right:1px solid {THEME['border']}; }}
    div[data-testid="metric-container"] {{ background-color:{THEME['bg_card']}; border:1px solid {THEME['border']}; border-radius:8px; padding:10px 16px; }}
    </style>""",
    unsafe_allow_html=True,
)


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

    moments = load_moments(_stock_ret)
    corr_s = get_index_corr(_stock_ret, _idx_ret)

    latest = _stock_ret.ffill()
    if len(latest) == 0:
        latest_s = pd.Series(dtype=float, name="daily_return")
    else:
        latest_s = latest.iloc[-1].rename("daily_return")

    disp = moments.copy()
    disp["daily_return"] = latest_s.reindex(disp.index)
    disp["corr_sp500"] = corr_s.reindex(disp.index)

    if PRICES_CACHE.exists():
        prices = pd.read_parquet(PRICES_CACHE)
        prices.index = pd.to_datetime(prices.index)
        ytd = compute_ytd_returns(prices, year=2025)
        disp["ytd_2025"] = ytd.reindex(disp.index)
    else:
        disp["ytd_2025"] = np.nan

    disp.index.name = "ticker"
    return disp


@st.cache_data(show_spinner=False)
def compute_rolling_for_stock(series: pd.Series, window: int = 30):
    s = series.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["mean", "variance", "skewness", "kurtosis"])
    out = pd.DataFrame(index=s.index)
    out["mean"] = s.rolling(window).mean()
    out["variance"] = s.rolling(window).var()
    out["skewness"] = s.rolling(window).skew()
    out["kurtosis"] = s.rolling(window).kurt()
    return out


def make_stock_timeseries_figure(ret_s: pd.Series, idx_ret: pd.Series, ticker: str) -> go.Figure:
    fig = go.Figure()

    cum_stock = ret_s.fillna(0).cumsum()
    cum_idx = idx_ret.reindex(ret_s.index).fillna(0).cumsum()

    fig.add_trace(go.Scatter(
        x=cum_stock.index,
        y=cum_stock.values,
        mode="lines",
        name=ticker,
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Cumulative return: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cum_idx.index,
        y=cum_idx.values,
        mode="lines",
        name="^GSPC",
        line=dict(width=1.8, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Cumulative return: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        **dict(
            PLOTLY_LAYOUT_DEFAULTS,
            title=f"{ticker} vs ^GSPC — Cumulative Log Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative return (%)",
            height=340,
        )
    )
    return fig


def make_rolling_detail_figure(rolling_df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    colors = {
        "mean": THEME["blue"],
        "variance": THEME["orange"],
        "skewness": THEME["purple"],
        "kurtosis": THEME["green_bright"],
    }

    for col in ["mean", "variance", "skewness", "kurtosis"]:
        if col in rolling_df.columns:
            fig.add_trace(go.Scatter(
                x=rolling_df.index,
                y=rolling_df[col],
                mode="lines",
                name=col.title(),
                line=dict(width=1.6, color=colors[col]),
            ))

    fig.update_layout(
        **dict(
            PLOTLY_LAYOUT_DEFAULTS,
            title=f"{ticker} — 30-Day Rolling Moments",
            xaxis_title="Date",
            yaxis_title="Value",
            height=360,
        )
    )
    return fig


def make_return_hist_kde(ret_s: pd.Series, ticker: str) -> go.Figure:
    from src.analytics import estimate_kde

    arr = ret_s.dropna().values
    fig = go.Figure()

    if len(arr) >= 5:
        x, d = estimate_kde(arr)
        fig.add_trace(go.Histogram(
            x=arr,
            nbinsx=60,
            histnorm="probability density",
            opacity=0.35,
            name="Histogram",
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=d,
            mode="lines",
            line=dict(width=2.3),
            name="KDE",
        ))

    fig.update_layout(
        **dict(
            PLOTLY_LAYOUT_DEFAULTS,
            title=f"{ticker} — Return Distribution",
            xaxis_title="Daily return (%)",
            yaxis_title="Density",
            height=320,
            barmode="overlay",
        )
    )
    return fig


st.markdown(section_header("Market Map"), unsafe_allow_html=True)

returns, stock_ret, idx_ret = load_data()
if returns is None:
    st.warning("⚠️ No data cached. Go to the **Data Pipeline** page first.")
    st.stop()

sector_df = load_sector()
sector_map = {}
market_caps = None
if sector_df is not None:
    sector_map = dict(zip(sector_df.index, sector_df["sector"].fillna("Unknown")))
    market_caps = sector_df["market_cap"]

disp_df = build_display_df(stock_ret, idx_ret)

with st.sidebar:
    st.markdown("### 🗺️ Market Map")
    color_metric = st.selectbox(
        "Color tiles by",
        [
            "daily_return",
            "ytd_2025",
            "mean",
            "variance",
            "skewness",
            "kurtosis",
            "corr_sp500",
            "p01",
            "p99",
        ],
        format_func=lambda x: {
            "daily_return": "Latest Daily Return",
            "ytd_2025": "2025 YTD Return",
            "mean": "Mean Return",
            "variance": "Variance",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "corr_sp500": "Correlation with ^GSPC",
            "p01": "1st Percentile",
            "p99": "99th Percentile",
        }.get(x, x),
        index=0,
    )

    size_mode = st.radio(
        "Tile area based on",
        ["Market cap", "Absolute metric magnitude"],
        index=0,
    )
    size_mode_value = "market_cap" if size_mode == "Market cap" else "metric_abs"

    sector_choices = sorted(set(sector_map.values())) if sector_map else []
    sector_filter = st.multiselect("Filter sectors", sector_choices, default=[])

    manual_ticker = st.selectbox(
        "Manual stock selection",
        [""] + sorted(stock_ret.columns.tolist()),
        index=0,
    )

plot_df = disp_df.copy()
if sector_filter:
    keep = [t for t in plot_df.index if sector_map.get(t, "Unknown") in sector_filter]
    plot_df = plot_df.loc[plot_df.index.intersection(keep)]

fig_map = treemap_market_map(
    stock_data=plot_df,
    color_metric=color_metric,
    sector_map=sector_map,
    market_caps=market_caps,
    title="S&P 500 Market Map",
    size_mode=size_mode_value,
)

clicked = plotly_events(
    fig_map,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=780,
    key="market_map_click",
)

st.caption("Click a stock tile to open its homework-related analytics panel.")

selected_ticker = st.session_state.get("selected_ticker", None)

if clicked and len(clicked) > 0:
    point_number = clicked[0].get("pointNumber", None)
    if point_number is not None:
        try:
            ticker_candidate = fig_map.data[0].customdata[point_number][0]
            if ticker_candidate in stock_ret.columns:
                st.session_state["selected_ticker"] = ticker_candidate
                selected_ticker = ticker_candidate
        except Exception:
            pass

if manual_ticker:
    st.session_state["selected_ticker"] = manual_ticker
    selected_ticker = manual_ticker

if selected_ticker is None and len(plot_df.index) > 0:
    selected_ticker = plot_df["daily_return"].abs().sort_values(ascending=False).index[0]
    st.session_state["selected_ticker"] = selected_ticker

st.markdown("---")

if selected_ticker and selected_ticker in stock_ret.columns:
    ret_s = stock_ret[selected_ticker].dropna()
    rolling_s = compute_rolling_for_stock(ret_s, window=30)

    common_idx = ret_s.index.intersection(idx_ret.index)
    ret_common = ret_s.reindex(common_idx)
    idx_common = idx_ret.reindex(common_idx)

    sec = sector_map.get(selected_ticker, "Unknown") if sector_map else "Unknown"
    row = disp_df.loc[selected_ticker]

    st.markdown(
        section_header(
            f"Selected Stock: {selected_ticker}",
            f"Sector: {sec} · analytics linked directly to the homework questions",
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Latest Return", f"{float(row.get('daily_return', np.nan)):+.2f}%")
    with c2:
        st.metric("Mean", f"{float(row.get('mean', np.nan)):.3f}")
    with c3:
        st.metric("Variance", f"{float(row.get('variance', np.nan)):.3f}")
    with c4:
        st.metric("Skewness", f"{float(row.get('skewness', np.nan)):.3f}")
    with c5:
        st.metric("Kurtosis", f"{float(row.get('kurtosis', np.nan)):.3f}")
    with c6:
        st.metric("Corr. to ^GSPC", f"{float(row.get('corr_sp500', np.nan)):.3f}")

    tab1, tab2, tab3 = st.tabs([
        "Moments / Distribution",
        "Time Dynamics",
        "Market Dependence",
    ])

    with tab1:
        left, right = st.columns(2)
        with left:
            stock_summary = pd.DataFrame({
                "Metric": ["Mean", "Variance", "Skewness", "Kurtosis", "1st pct", "99th pct"],
                "Value": [
                    row.get("mean", np.nan),
                    row.get("variance", np.nan),
                    row.get("skewness", np.nan),
                    row.get("kurtosis", np.nan),
                    row.get("p01", np.nan),
                    row.get("p99", np.nan),
                ],
            })
            st.dataframe(
                stock_summary.style.format({"Value": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.plotly_chart(make_return_hist_kde(ret_s, selected_ticker), use_container_width=True)

    with tab2:
        st.plotly_chart(make_stock_timeseries_figure(ret_common, idx_common, selected_ticker), use_container_width=True)
        st.plotly_chart(make_rolling_detail_figure(rolling_s, selected_ticker), use_container_width=True)

    with tab3:
        beta_like = np.nan
        if len(ret_common) > 20 and np.var(idx_common) > 0:
            beta_like = np.cov(ret_common, idx_common)[0, 1] / np.var(idx_common)

        risk_tbl = pd.DataFrame({
            "Indicator": ["Correlation to ^GSPC", "Beta-like slope", "Daily observations"],
            "Value": [
                row.get("corr_sp500", np.nan),
                beta_like,
                ret_s.shape[0],
            ],
        })
        st.dataframe(
            risk_tbl.style.format({"Value": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            interpretation_box(
                f"<strong>{selected_ticker}</strong> is connected back to the homework via its "
                "unconditional moments, its rolling dynamics, and its dependence on the market index."
            ),
            unsafe_allow_html=True,
        )
else:
    st.info("Select a stock from the map or from the sidebar.")