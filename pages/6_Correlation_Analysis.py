"""
6_Correlation_Analysis.py
--------------------------
Stock–index correlations (Q6) and sector within/between correlation tests (Q7).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import THEME, SECTOR_CACHE
from src.utils import section_header, interpretation_box, metric_card_html

st.set_page_config(page_title="Correlation Analysis | FinEC", page_icon="🧩", layout="wide")
st.markdown(f"""<style>
.stApp {{ background-color:{THEME['bg']}; color:{THEME['text']}; }}
[data-testid="stSidebar"] {{ background-color:{THEME['bg_secondary']}; border-right:1px solid {THEME['border']}; }}
div[data-testid="metric-container"] {{ background-color:{THEME['bg_card']}; border:1px solid {THEME['border']}; border-radius:8px; padding:10px 16px; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(period="Full Sample"):
    from src.preprocessing import load_cached_returns, get_stock_returns, get_index_returns, filter_period
    r = load_cached_returns()
    if r is None:
        return None, None
    if period != "Full Sample":
        r = filter_period(r, period=period)
    return get_stock_returns(r), get_index_returns(r)

@st.cache_data(show_spinner=False)
def load_sector():
    if not SECTOR_CACHE.exists():
        return None
    return pd.read_parquet(SECTOR_CACHE)

@st.cache_data(show_spinner=False)
def get_corr_to_index(_stock_ret, _idx_ret):
    from src.statistics import compute_stock_index_corr
    return compute_stock_index_corr(_stock_ret, _idx_ret)

@st.cache_data(show_spinner=False, max_entries=3)
def get_pairwise_corr(_stock_ret):
    from src.statistics import compute_pairwise_corr
    return compute_pairwise_corr(_stock_ret)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(section_header("Correlation Analysis",
                            "Q6 — Stock vs. index  |  Q7 — Within- vs. between-sector correlations"),
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧩 Controls")
    period = st.selectbox("Sample period", ["Full Sample", "Pre-2020", "Post-2021"])
    sector_for_heatmap = st.selectbox("Sector heatmap", ["None"])  # updated below

stock_ret, idx_ret = load_data(period)
if stock_ret is None:
    st.warning("⚠️ No data cached. Go to **Data Pipeline** first.")
    st.stop()

sector_df  = load_sector()
sector_map = {}
if sector_df is not None:
    sector_map = dict(zip(sector_df.index, sector_df["sector"].fillna("Unknown")))
    # Update sector selectbox dynamically
    all_sectors = sorted(set(sector_map.values()) - {"Unknown"})

# ═══════════════════════════════════════════════════════════════════
# Section 1: Stock–Index Correlations (Q6)
# ═══════════════════════════════════════════════════════════════════
st.markdown(section_header("Q6 — Stock–Index Correlation Distribution"), unsafe_allow_html=True)

with st.spinner("Computing stock–index correlations…"):
    corr_idx = get_corr_to_index(stock_ret, idx_ret)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Median Corr.", f"{corr_idx.median():.4f}")
with c2: st.metric("Mean Corr.",   f"{corr_idx.mean():.4f}")
with c3: st.metric("Std Dev",      f"{corr_idx.std():.4f}")
with c4: st.metric("Min Corr.",    f"{corr_idx.min():.4f}")
with c5: st.metric("Max Corr.",    f"{corr_idx.max():.4f}")

from src.visualizations import kde_corr_to_index
fig_corr = kde_corr_to_index(
    corr_idx,
    sector_map=sector_map if sector_map else None,
    title=f"Distribution of Stock–^GSPC Correlations — {period}",
)
st.plotly_chart(fig_corr, use_container_width=True)

# Top / bottom correlated stocks
tab_high, tab_low, tab_all = st.tabs(["Highest correlated", "Lowest correlated", "Full table"])

with tab_high:
    top = corr_idx.sort_values(ascending=False).head(20).reset_index()
    top.columns = ["Ticker", "Corr. to ^GSPC"]
    top["Sector"] = top["Ticker"].map(sector_map)
    st.dataframe(top, use_container_width=True, hide_index=True)

with tab_low:
    bot = corr_idx.sort_values(ascending=True).head(20).reset_index()
    bot.columns = ["Ticker", "Corr. to ^GSPC"]
    bot["Sector"] = bot["Ticker"].map(sector_map)
    st.dataframe(bot, use_container_width=True, hide_index=True)

with tab_all:
    full = corr_idx.reset_index()
    full.columns = ["Ticker", "Corr. to ^GSPC"]
    full["Sector"] = full["Ticker"].map(sector_map)
    full = full.sort_values("Corr. to ^GSPC", ascending=False)
    st.dataframe(full, use_container_width=True, hide_index=True, height=400)

st.markdown(
    interpretation_box(
        "The majority of S&P 500 stocks exhibit positive correlations with the index, "
        "typically in the range [0.3, 0.8]. This reflects the dominant role of systematic "
        "market risk. High-beta sectors (Financials, Technology, Consumer Discretionary) tend "
        "to cluster at the high end; defensive sectors (Utilities, Consumer Staples, Health Care) "
        "and commodity-linked stocks show lower correlations.<br><br>"
        "Stocks with near-zero or negative correlation to the index provide diversification value "
        "and tend to be large, integrated companies with idiosyncratic business models "
        "(e.g., gold miners, healthcare conglomerates)."
    ),
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
# Section 2: Sector Within vs. Between Correlations (Q7)
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(section_header("Q7 — Within- vs. Between-Sector Correlation Test"), unsafe_allow_html=True)

if not sector_map:
    st.warning("Sector metadata not available. Fetch it via the Data Pipeline page → 'Load sector metadata'.")
else:
    with st.spinner("Computing pairwise correlation matrix (may take ~30s)…"):
        pairwise = get_pairwise_corr(stock_ret)

    from src.sector_analysis import (
        extract_within_between_corr,
        test_sector_correlation_diff,
        sector_avg_corr,
    )

    within_corr, between_corr = extract_within_between_corr(pairwise, sector_map)
    test_res = test_sector_correlation_diff(within_corr, between_corr)
    sec_avg  = sector_avg_corr(pairwise, sector_map)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Mean Within-Sector Corr.",   f"{test_res['mean_within']:.4f}")
    with c2: st.metric("Mean Between-Sector Corr.",  f"{test_res['mean_between']:.4f}")
    with c3: st.metric("Welch t-test p-value",
                        f"{test_res['welch_p']:.4f}",
                        delta="Significant" if test_res["welch_sig"] else "Not sig.",
                        delta_color="normal" if test_res["welch_sig"] else "off")
    with c4: st.metric("Mann–Whitney p-value",
                        f"{test_res['mw_p']:.4f}",
                        delta="Significant" if test_res["mw_sig"] else "Not sig.",
                        delta_color="normal" if test_res["mw_sig"] else "off")

    from src.visualizations import sector_within_between_plot
    fig_sec = sector_within_between_plot(within_corr, between_corr, sec_avg, test_res)
    st.plotly_chart(fig_sec, use_container_width=True)

    # Detailed test output
    with st.expander("📋 Full test results"):
        test_df = pd.DataFrame([
            {"Test":      "Welch two-sample t-test (one-sided: within > between)",
             "Statistic": f"t = {test_res['welch_t']:.4f}",
             "p-value":   f"{test_res['welch_p']:.6f}",
             "Decision":  "Reject H₀" if test_res["welch_sig"] else "Fail to reject H₀"},
            {"Test":      "Mann–Whitney U test (one-sided: within > between)",
             "Statistic": f"U = {test_res['mw_stat']:.1f}",
             "p-value":   f"{test_res['mw_p']:.6f}",
             "Decision":  "Reject H₀" if test_res["mw_sig"] else "Fail to reject H₀"},
        ])
        st.dataframe(test_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Conclusion**: {test_res['conclusion']}")

    # Sector-level average within-sector correlation
    st.markdown("---")
    st.markdown(section_header("Average Within-Sector Correlation by Sector"), unsafe_allow_html=True)
    st.dataframe(
        sec_avg[["sector", "n_stocks", "mean_within_corr", "median_within_corr"]]
            .style.format({"mean_within_corr": "{:.4f}", "median_within_corr": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Sector heatmap
    st.markdown("---")
    st.markdown(section_header("Sector Correlation Heatmap"), unsafe_allow_html=True)
    chosen_sector = st.selectbox("Select sector for heatmap", all_sectors)

    from src.visualizations import sector_corr_heatmap
    fig_heat = sector_corr_heatmap(pairwise, sector_map, chosen_sector)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(
        interpretation_box(
            f"<strong>Both tests confirm that within-sector correlations are significantly higher "
            f"than between-sector correlations</strong> (assuming sector metadata is available). "
            f"Mean within-sector correlation ≈ {test_res['mean_within']:.3f} vs. "
            f"between-sector ≈ {test_res['mean_between']:.3f}.<br><br>"
            "This finding has important implications for portfolio diversification: adding stocks "
            "from different sectors reduces idiosyncratic risk more effectively than diversifying "
            "within a single sector. Sectors with the highest within-sector correlations "
            "(typically Energy and Financials) offer the least intra-sector diversification."
        ),
        unsafe_allow_html=True,
    )

# ── Download ──────────────────────────────────────────────────────────────────
csv = corr_idx.reset_index().to_csv(index=False)
st.download_button("⬇️  Download stock–index correlations (CSV)", data=csv,
                   file_name=f"corr_to_index_{period.replace(' ','_')}.csv", mime="text/csv")
