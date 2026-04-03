"""
4_Cross_Sectional_Density.py
-----------------------------
Cross-sectional daily return density: 3D surface + time-series moments.

Homework Q4:
  - 3D plot: time × stock return × density
  - Time series: daily cross-sectional variance, skewness, kurtosis
  - Commentary comparing with Q3 (index rolling moments)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import THEME, XSECTION_STRIDE
from src.utils import section_header, interpretation_box

st.set_page_config(page_title="Cross-Sectional Density | FinEC", page_icon="🌊", layout="wide")
st.markdown(f"""<style>
.stApp {{ background-color:{THEME['bg']}; color:{THEME['text']}; }}
[data-testid="stSidebar"] {{ background-color:{THEME['bg_secondary']}; border-right:1px solid {THEME['border']}; }}
div[data-testid="metric-container"] {{ background-color:{THEME['bg_card']}; border:1px solid {THEME['border']}; border-radius:8px; padding:10px 16px; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_stock_returns():
    from src.preprocessing import load_cached_returns, get_stock_returns
    r = load_cached_returns()
    if r is None:
        return None
    return get_stock_returns(r)

@st.cache_data(show_spinner=False)
def get_xsec_moments(_stock_ret):
    from src.analytics import compute_cross_sectional_moments
    return compute_cross_sectional_moments(_stock_ret)

@st.cache_data(show_spinner=False, max_entries=3)
def get_kde_surface(_stock_ret, stride, n_points):
    from src.analytics import build_kde_surface
    return build_kde_surface(_stock_ret, stride=stride, n_points=n_points)

@st.cache_data(show_spinner=False)
def load_rolling_index():
    from src.preprocessing import load_cached_returns, get_index_returns
    from src.analytics import compute_rolling_moments
    r = load_cached_returns()
    if r is None:
        return None
    idx = get_index_returns(r)
    return compute_rolling_moments(idx)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(section_header("Cross-Sectional Return Density", "Q4 — Daily distribution of returns across all stocks"), unsafe_allow_html=True)

stock_ret = load_stock_returns()
if stock_ret is None:
    st.warning("⚠️ No data cached. Go to **Data Pipeline** first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 Controls")
    stride = st.slider(
        "3D surface stride (days between samples)",
        min_value=2, max_value=20, value=XSECTION_STRIDE,
        help="Larger = faster but lower temporal resolution",
    )
    n_points = st.slider("KDE grid resolution", 50, 300, 150, step=25)
    date_range = st.date_input(
        "Date range",
        value=(stock_ret.index.min().date(), stock_ret.index.max().date()),
        min_value=stock_ret.index.min().date(),
        max_value=stock_ret.index.max().date(),
    )

# Apply date range
if len(date_range) == 2:
    s, e = date_range
    stock_ret_f = stock_ret[(stock_ret.index >= pd.Timestamp(s)) &
                             (stock_ret.index <= pd.Timestamp(e))]
else:
    stock_ret_f = stock_ret

# ── Cross-sectional moments ───────────────────────────────────────────────────
xsec_df = get_xsec_moments(stock_ret_f)

c1, c2, c3, c4 = st.columns(4)
if len(xsec_df) > 0:
    with c1: st.metric("Avg Daily Cross-Sec. Variance",  f"{xsec_df['variance'].mean():.4f}")
    with c2: st.metric("Avg Daily Cross-Sec. Skewness",  f"{xsec_df['skewness'].mean():.4f}")
    with c3: st.metric("Avg Daily Cross-Sec. Kurtosis",  f"{xsec_df['kurtosis'].mean():.4f}")
    with c4: st.metric("Trading Days",                    f"{len(xsec_df):,}")

# ── 3D surface ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("3D Density Surface", "Time × Return × Density"), unsafe_allow_html=True)
st.info(
    "⚠️ Building the 3D surface may take 30–90 seconds on first load. "
    "Results are cached for the session.",
    icon="⏱️",
)

with st.spinner("Estimating KDE surface…"):
    dates_surf, x_grid, Z = get_kde_surface(stock_ret_f, stride, n_points)

from src.visualizations import cross_sectional_3d_surface
fig3d = cross_sectional_3d_surface(
    dates_surf, x_grid, Z,
    title="Cross-Sectional Return Density Over Time",
)
st.plotly_chart(fig3d, use_container_width=True)

st.markdown(
    interpretation_box(
        "Each slice of the 3D surface is the Gaussian KDE of daily returns across all ~500 S&P 500 "
        "stocks on that day. The surface encodes three dimensions of information: <br>"
        "• <em>Time</em> (y-axis): reveals structural breaks, e.g. COVID (Mar 2020), 2022 bear market.<br>"
        "• <em>Return</em> (x-axis): the cross-sectional spread of stock returns on each day.<br>"
        "• <em>Density</em> (z-axis): the concentration of stocks around each return level.<br><br>"
        "Peaks that broaden or shift dramatically in the time direction indicate regime changes. "
        "On crisis days, the distribution flattens and shifts left (market-wide selloffs disperse returns). "
        "On calm days, the distribution is tightly peaked near zero."
    ),
    unsafe_allow_html=True,
)

# ── Time-series of cross-sectional moments ─────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Daily Cross-Sectional Moment Time Series"), unsafe_allow_html=True)

from src.visualizations import cross_sectional_moments_plot
fig_ts = cross_sectional_moments_plot(xsec_df)
st.plotly_chart(fig_ts, use_container_width=True)

# ── Comparison with rolling index moments ─────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Comparison: Cross-Sectional vs. Index Rolling Moments",
                            "Are cross-sectional and time-series moments telling the same story?"), unsafe_allow_html=True)

rolling_idx = load_rolling_index()

if rolling_idx is not None and not xsec_df.empty:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from src.config import PLOTLY_LAYOUT_DEFAULTS

    for moment in ["variance", "skewness", "kurtosis"]:
        fig_comp = make_subplots(rows=1, cols=1)

        # Cross-sectional
        cs = xsec_df[moment].dropna()
        fig_comp.add_trace(go.Scatter(
            x=cs.index, y=cs.values,
            mode="lines",
            name=f"Cross-sectional {moment}",
            line=dict(color=THEME["orange"], width=1.5),
        ))

        # Rolling index
        if moment in rolling_idx.columns:
            ri = rolling_idx[moment].dropna()
            fig_comp.add_trace(go.Scatter(
                x=ri.index, y=ri.values,
                mode="lines",
                name=f"30-day rolling index {moment}",
                line=dict(color=THEME["blue"], width=1.5, dash="dash"),
            ))

        fig_comp.update_layout(
            **dict(PLOTLY_LAYOUT_DEFAULTS,
                   title=f"{moment.title()}: Cross-Sectional vs. Rolling Index",
                   xaxis_title="Date",
                   yaxis_title=moment.title(),
                   height=320,
                   showlegend=True),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

st.markdown(
    interpretation_box(
        "<strong>Cross-sectional vs. time-series moments compared:</strong><br><br>"
        "<em>Variance</em>: Cross-sectional variance measures the dispersion of returns <em>across stocks</em> "
        "on a given day, while rolling index variance measures the volatility of the <em>market portfolio</em> "
        "over recent time. Both spike during crises (positive co-movement), but cross-sectional variance "
        "also rises when sector rotation is high even if the index is calm. "
        "Cross-sectional variance is a leading indicator of aggregate market stress.<br><br>"
        "<em>Skewness</em>: Cross-sectional skewness tends to be positive on up-days (a few big winners) "
        "and negative on down-days (many stocks fall together). The rolling index skewness is typically "
        "negative, reflecting the asymmetry of market crashes.<br><br>"
        "<em>Kurtosis</em>: Cross-sectional kurtosis spikes when most stocks cluster near zero return "
        "but a handful have extreme moves — often on earnings days or index rebalancing events."
    ),
    unsafe_allow_html=True,
)

# ── Download ──────────────────────────────────────────────────────────────────
csv = xsec_df.reset_index().to_csv(index=False)
st.download_button("⬇️  Download cross-sectional moments (CSV)", data=csv,
                   file_name="cross_sectional_moments.csv", mime="text/csv")
