"""
5_Metric_Relationships.py
--------------------------
Pairwise scatterplots of all six stock-level moment statistics.

Homework Q5: C(6,2) = 15 pairs; OLS trend lines; sector coloring; commentary.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.config import THEME, SECTOR_CACHE, MOMENT_COLS, MOMENT_LABELS, PLOTLY_LAYOUT_DEFAULTS
from src.utils import section_header, interpretation_box, page_css, page_header_html


st.set_page_config(page_title="Metric Relationships | FinEC", page_icon="🔗", layout="wide")
st.markdown(page_css(), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_moments(period="Full Sample"):
    """
    Load stock moments for the given period.
    Subperiods use save_cache=False to prevent corrupting the full-sample cache.
    """
    from src.preprocessing import load_cached_returns, get_stock_returns, filter_period
    from src.analytics import compute_stock_moments
    r = load_cached_returns()
    if r is None:
        return None
    is_full = (period == "Full Sample")
    if not is_full:
        r = filter_period(r, period=period)
    return compute_stock_moments(get_stock_returns(r), force_refresh=not is_full, save_cache=is_full)


@st.cache_data(show_spinner=False)
def load_sector():
    if not SECTOR_CACHE.exists():
        return None
    return pd.read_parquet(SECTOR_CACHE)


def safe_scalar(x):
    """
    Convert scipy/numpy outputs to a plain Python float.
    Handles scalar floats, 0-d arrays, 1-element arrays, and matrix outputs.
    """
    if pd.isna(x).all() if isinstance(x, np.ndarray) else pd.isna(x):
        return np.nan

    if isinstance(x, np.ndarray):
        flat = np.asarray(x).ravel()
        if flat.size == 0:
            return np.nan
        return float(flat[0])

    return float(x)


def safe_spearman(x, y):
    """
    Compute Spearman correlation robustly and always return scalar floats.
    """
    rho, pval = spearmanr(np.asarray(x), np.asarray(y))
    rho = safe_scalar(rho)
    pval = safe_scalar(pval)
    return rho, pval


def safe_pearson(x, y):
    """
    Compute Pearson correlation robustly and always return scalar floats.
    """
    r, pval = pearsonr(np.asarray(x), np.asarray(y))
    r = safe_scalar(r)
    pval = safe_scalar(pval)
    return r, pval


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    page_header_html(
        "Pairwise Metric Relationships",
        "Q5 — All C(6,2)=15 pairwise scatterplots · OLS trend lines · sector coloring",
        "🔗",
    ),
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔗 Controls")
    period = st.selectbox("Sample period", ["Full Sample", "Pre-2020", "Post-2021"])
    color_by = st.checkbox("Color by sector", value=True)
    x_metric = st.selectbox(
        "X-axis (custom plot)",
        MOMENT_COLS,
        format_func=lambda c: MOMENT_LABELS[c],
    )
    y_metric = st.selectbox(
        "Y-axis (custom plot)",
        [c for c in MOMENT_COLS if c != x_metric],
        format_func=lambda c: MOMENT_LABELS[c],
    )

moments = load_moments(period)
if moments is None:
    st.warning("⚠️ No data cached. Go to **Data Pipeline** first.")
    st.stop()

sector_df = load_sector()
sector_map = {}
if sector_df is not None:
    sector_map = dict(zip(sector_df.index, sector_df["sector"].fillna("Unknown")))

# ── Full pairplot grid ────────────────────────────────────────────────────────
st.markdown(section_header("All 15 Pairwise Scatterplots"), unsafe_allow_html=True)

from src.visualizations import all_pairs_subplot
fig_all = all_pairs_subplot(moments, sector_map=sector_map if color_by and sector_map else None)
st.plotly_chart(fig_all, use_container_width=True)

# ── Custom interactive plot ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Custom Interactive Scatterplot"), unsafe_allow_html=True)

from src.visualizations import pairwise_scatterplot
fig_custom = pairwise_scatterplot(
    moments,
    x_metric=x_metric,
    y_metric=y_metric,
    sector_map=sector_map if sector_map else None,
    color_by_sector=color_by,
)
st.plotly_chart(fig_custom, use_container_width=True)

# ── Correlation table ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Rank Correlation Matrix (Spearman)"), unsafe_allow_html=True)

spearman_corrs = pd.DataFrame(index=MOMENT_COLS, columns=MOMENT_COLS, dtype=float)
pearson_corrs = pd.DataFrame(index=MOMENT_COLS, columns=MOMENT_COLS, dtype=float)

for ci in MOMENT_COLS:
    for cj in MOMENT_COLS:
        if ci == cj:
            spearman_corrs.loc[ci, cj] = 1.0
            pearson_corrs.loc[ci, cj] = 1.0
            continue

        ab = moments[[ci, cj]].dropna()
        if len(ab) < 10:
            spearman_corrs.loc[ci, cj] = np.nan
            pearson_corrs.loc[ci, cj] = np.nan
        else:
            sp, _ = safe_spearman(ab[ci], ab[cj])
            pc, _ = safe_pearson(ab[ci], ab[cj])

            spearman_corrs.loc[ci, cj] = round(float(sp), 3) if pd.notna(sp) else np.nan
            pearson_corrs.loc[ci, cj] = round(float(pc), 3) if pd.notna(pc) else np.nan

spearman_corrs.index = [MOMENT_LABELS[c] for c in MOMENT_COLS]
spearman_corrs.columns = [MOMENT_LABELS[c] for c in MOMENT_COLS]
pearson_corrs.index = [MOMENT_LABELS[c] for c in MOMENT_COLS]
pearson_corrs.columns = [MOMENT_LABELS[c] for c in MOMENT_COLS]

tab_sp, tab_pc = st.tabs(["Spearman (rank)", "Pearson (linear)"])
with tab_sp:
    st.dataframe(
        spearman_corrs.round(3),
        use_container_width=True,
    )
with tab_pc:
    st.dataframe(
        pearson_corrs.round(3),
        use_container_width=True,
    )

# ── Heatmap of correlations ───────────────────────────────────────────────────
import plotly.graph_objects as go

heat_vals = spearman_corrs.astype(float).values
heat_text = []
for row in heat_vals:
    row_text = []
    for v in row:
        row_text.append("" if pd.isna(v) else f"{v:.3f}")
    heat_text.append(row_text)

fig_heat = go.Figure(
    go.Heatmap(
        z=heat_vals,
        x=spearman_corrs.columns.tolist(),
        y=spearman_corrs.index.tolist(),
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        text=heat_text,
        texttemplate="%{text}",
        colorbar=dict(title="ρ"),
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
    )
)
fig_heat.update_layout(
    **dict(
        PLOTLY_LAYOUT_DEFAULTS,
        title="Spearman Rank Correlation Heatmap — Six Moment Statistics",
        height=380,
    ),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Key findings table ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Key Empirical Findings"), unsafe_allow_html=True)

from itertools import combinations

rows = []
for ci, cj in combinations(MOMENT_COLS, 2):
    ab = moments[[ci, cj]].dropna()
    if len(ab) < 10:
        continue

    sp, sp_p = safe_spearman(ab[ci], ab[cj])
    pc, pc_p = safe_pearson(ab[ci], ab[cj])

    rows.append(
        {
            "Pair": f"{MOMENT_LABELS[ci]} × {MOMENT_LABELS[cj]}",
            "Pearson r": round(float(pc), 3) if pd.notna(pc) else np.nan,
            "Pearson p": round(float(pc_p), 4) if pd.notna(pc_p) else np.nan,
            "Spearman ρ": round(float(sp), 3) if pd.notna(sp) else np.nan,
            "Spearman p": round(float(sp_p), 4) if pd.notna(sp_p) else np.nan,
            "Significant (5%)": "✅" if pd.notna(pc_p) and pc_p < 0.05 else "❌",
        }
    )

findings_df = pd.DataFrame(rows)

if not findings_df.empty:
    findings_df["_sort_abs"] = findings_df["Pearson r"].abs()
    findings_df = findings_df.sort_values("_sort_abs", ascending=False).drop(columns="_sort_abs")
    st.dataframe(findings_df, use_container_width=True, hide_index=True)
else:
    st.info("Not enough data to compute pairwise findings for the selected period.")

st.markdown(
    interpretation_box(
        "<strong>Variance–Kurtosis</strong>: Strong positive correlation — stocks with higher volatility "
        "also tend to have fatter tails. Both are driven by information uncertainty.<br><br>"
        "<strong>Mean–Variance</strong>: Weak positive relationship — consistent with risk-return theory, "
        "though the relationship is noisy in the cross-section of daily returns.<br><br>"
        "<strong>Skewness–Kurtosis</strong>: Negative correlation across sectors — stocks with very "
        "negative skewness (crash-prone) exhibit especially fat tails.<br><br>"
        "<strong>Mean–Skewness</strong>: Stocks with higher mean returns tend to have less negative "
        "skewness — possibly reflecting that momentum winners experience smoother gains.<br><br>"
        "<strong>P1–P99 asymmetry</strong>: The left tail (1st percentile) tends to be larger in magnitude "
        "than the right (99th percentile), confirming the well-documented negative skewness of equity returns."
    ),
    unsafe_allow_html=True,
)

# ── Download ──────────────────────────────────────────────────────────────────
csv = moments.reset_index().to_csv(index=False)
st.download_button(
    "⬇️  Download moments (CSV)",
    data=csv,
    file_name=f"moments_{period.replace(' ', '_')}.csv",
    mime="text/csv",
)