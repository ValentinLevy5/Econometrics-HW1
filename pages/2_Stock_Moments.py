"""
2_Stock_Moments.py
------------------
Stock-level descriptive statistics and nonparametric density estimates.

Homework Q2: mean, variance, skewness, kurtosis, 1st and 99th percentile
for every S&P 500 stock; KDE visualizations and commentary.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import THEME, SECTOR_CACHE, MOMENT_LABELS, MOMENT_COLS
from src.utils import section_header, interpretation_box

st.set_page_config(page_title="Stock Moments | FinEC", page_icon="📊", layout="wide")
st.markdown(f"""<style>
.stApp {{ background-color: {THEME['bg']}; color: {THEME['text']}; }}
[data-testid="stSidebar"] {{ background-color: {THEME['bg_secondary']}; border-right:1px solid {THEME['border']}; }}
div[data-testid="metric-container"] {{ background-color:{THEME['bg_card']}; border:1px solid {THEME['border']}; border-radius:8px; padding:10px 16px; }}
</style>""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_returns():
    from src.preprocessing import load_cached_returns, get_stock_returns
    r = load_cached_returns()
    return r, get_stock_returns(r) if r is not None else None

@st.cache_data(show_spinner=False)
def load_moments(_stock_ret):
    from src.analytics import compute_stock_moments
    return compute_stock_moments(_stock_ret)

@st.cache_data(show_spinner=False)
def load_sector():
    if not SECTOR_CACHE.exists():
        return None
    return pd.read_parquet(SECTOR_CACHE)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(section_header("Stock-Level Moments", "Q2 — Descriptive statistics for all S&P 500 stocks"), unsafe_allow_html=True)

returns, stock_ret = load_returns()
if returns is None:
    st.warning("⚠️ No data cached. Go to the **Data Pipeline** page first.")
    st.stop()

sector_df = load_sector()
sector_map = {}
if sector_df is not None:
    sector_map = dict(zip(sector_df.index, sector_df["sector"].fillna("Unknown")))

moments = load_moments(stock_ret)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Filters")

    # Period selector
    period = st.selectbox("Sample Period", ["Full Sample", "Pre-2020", "Post-2021"])

    # Sector filter
    all_sectors = sorted(set(sector_map.values()) - {"Unknown"}) if sector_map else []
    selected_sectors = st.multiselect("Filter by sector", all_sectors, default=[])

    # Highlight sector in KDE
    highlight = st.selectbox("Highlight sector in KDE", ["None"] + all_sectors)
    highlight_sec = None if highlight == "None" else highlight

    # Ticker search
    ticker_search = st.text_input("Search ticker", "").upper()

# Recompute if non-full period
if period != "Full Sample":
    from src.preprocessing import filter_period
    with st.spinner(f"Computing moments for {period}…"):
        sub_ret = filter_period(returns, period=period)
        from src.preprocessing import get_stock_returns as gsr
        stock_sub = gsr(sub_ret)
        from src.analytics import compute_stock_moments
        moments = compute_stock_moments(stock_sub, force_refresh=True)

# Add sector column
moments_display = moments.copy()
moments_display["sector"] = moments_display.index.map(lambda t: sector_map.get(t, "Unknown"))

# Apply filters
if selected_sectors:
    moments_display = moments_display[moments_display["sector"].isin(selected_sectors)]
if ticker_search:
    mask = moments_display.index.str.contains(ticker_search, case=False, na=False)
    moments_display = moments_display[mask]

# ── Key metrics ───────────────────────────────────────────────────────────────
st.markdown(f"**Period: {period}** — {len(moments_display)} stocks")

c1, c2, c3, c4, c5, c6 = st.columns(6)
for col_name, widget_col in zip(MOMENT_COLS, [c1, c2, c3, c4, c5, c6]):
    median_val = moments_display[col_name].median()
    widget_col.metric(
        f"Median {col_name.title()}",
        f"{median_val:.4f}",
    )

# ── KDE Grid ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Nonparametric Density Estimates", "Gaussian KDE for each of the six moment statistics"), unsafe_allow_html=True)

from src.visualizations import kde_moments_grid
fig_kde = kde_moments_grid(moments_display, sector_map=sector_map if sector_map else None, highlight_sector=highlight_sec)
st.plotly_chart(fig_kde, use_container_width=True)

st.markdown(
    interpretation_box(
        "<strong>Mean returns</strong>: The cross-sectional distribution of mean daily returns is tightly "
        "centered near zero, reflecting the difficulty of generating persistent alpha. "
        "A slight positive skew reflects the long-run upward drift of equities.<br><br>"
        "<strong>Variance</strong>: Highly right-skewed — a small number of high-volatility stocks "
        "(e.g. biotech, energy) dominate the right tail.<br><br>"
        "<strong>Skewness</strong>: Most stocks exhibit negative skewness (more frequent small gains "
        "but occasional large losses), consistent with the equity premium literature.<br><br>"
        "<strong>Excess kurtosis</strong>: Universally positive and large, confirming fat tails. "
        "Daily stock returns are far from Gaussian.<br><br>"
        "<strong>1st / 99th percentile</strong>: These tail measures confirm asymmetric tail behaviour. "
        "The left tail (1st pctile) is generally fatter than the right."
    ),
    unsafe_allow_html=True,
)

# ── Ranked table ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Ranked Statistics Table"), unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Mean", "Variance", "Skewness", "Kurtosis", "1st Pctile", "99th Pctile"]
)
for tab, col in zip([tab1, tab2, tab3, tab4, tab5, tab6], MOMENT_COLS):
    with tab:
        ranked = (
            moments_display[[col, "sector"]]
            .sort_values(col, ascending=False)
            .reset_index()
        )
        ranked.columns = ["Ticker", col.title(), "Sector"]
        st.dataframe(ranked, use_container_width=True, hide_index=True, height=400)

# ── Summary statistics of statistics ──────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Summary of Summary Statistics"), unsafe_allow_html=True)

summary = moments_display[MOMENT_COLS].describe().T
summary.index.name = "Statistic"
summary.columns = [c.title() for c in summary.columns]
st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)

# ── Sector averages ───────────────────────────────────────────────────────────
if sector_map:
    st.markdown("---")
    st.markdown(section_header("Sector-Average Moments"), unsafe_allow_html=True)
    from src.sector_analysis import sector_moment_summary
    sec_summary = sector_moment_summary(moments_display.drop(columns=["sector"], errors="ignore"), sector_map)
    st.dataframe(sec_summary.style.format("{:.4f}"), use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
csv = moments_display.reset_index().to_csv(index=False)
st.download_button(
    "⬇️  Download moments table (CSV)",
    data=csv,
    file_name=f"stock_moments_{period.replace(' ','_')}.csv",
    mime="text/csv",
)
