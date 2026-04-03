"""
7_Stability_Analysis.py
-----------------------
Subperiod comparison and KS-based stability analysis.

Homework Q8: Repeat Q1–Q3 for pre-2020 and post-2021 and compare.
Homework Q9: KS tests for temporal stability; discuss whether the world is
             becoming more risky.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.config import THEME, PERIODS, PLOTLY_LAYOUT_DEFAULTS
from src.utils import section_header, interpretation_box

st.set_page_config(page_title="Stability Analysis | FinEC", page_icon="⏳", layout="wide")
st.markdown(f"""<style>
.stApp {{ background-color:{THEME['bg']}; color:{THEME['text']}; }}
[data-testid="stSidebar"] {{ background-color:{THEME['bg_secondary']}; border-right:1px solid {THEME['border']}; }}
div[data-testid="metric-container"] {{ background-color:{THEME['bg_card']}; border:1px solid {THEME['border']}; border-radius:8px; padding:10px 16px; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_full_returns():
    from src.preprocessing import load_cached_returns
    return load_cached_returns()


@st.cache_data(show_spinner=False)
def build_subperiods(_returns):
    from src.stability import build_subperiod_returns
    return build_subperiod_returns(_returns)


@st.cache_data(show_spinner=False)
def get_risk_evolution(_returns_dict):
    from src.stability import risk_evolution_summary
    return risk_evolution_summary(_returns_dict)


@st.cache_data(show_spinner=False)
def get_ks_report(_returns_dict, metric):
    from src.stability import ks_stability_report
    return ks_stability_report(_returns_dict, metric=metric)


@st.cache_data(show_spinner=False)
def get_index_ks(_returns_dict):
    from src.stability import ks_index_pairwise
    return ks_index_pairwise(_returns_dict)


@st.cache_data(show_spinner=False)
def get_period_moments_comp(_returns_dict):
    from src.stability import compare_moments_across_periods
    return compare_moments_across_periods(_returns_dict)


def round_numeric_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    section_header(
        "Stability Analysis",
        "Q8 & Q9 — Subperiod comparisons, KS tests, and risk evolution",
    ),
    unsafe_allow_html=True,
)

returns = load_full_returns()
if returns is None:
    st.warning("⚠️ No data cached. Go to **Data Pipeline** first.")
    st.stop()

with st.spinner("Building subperiod slices…"):
    subperiods = build_subperiods(returns)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⏳ Controls")
    ks_metric = st.selectbox(
        "KS test metric",
        ["returns", "variance", "skewness", "kurtosis"],
        format_func=lambda x: {
            "returns": "Pooled stock returns",
            "variance": "Per-stock variance",
            "skewness": "Per-stock skewness",
            "kurtosis": "Per-stock kurtosis",
        }[x],
    )

# ═══════════════════════════════════════════════════════════════════
# Section 1: Risk Evolution Summary (Q8)
# ═══════════════════════════════════════════════════════════════════
st.markdown(section_header("Q8 — Risk Indicators Across Subperiods"), unsafe_allow_html=True)

risk_df = get_risk_evolution(subperiods)

if not risk_df.empty:
    st.dataframe(
        round_numeric_df(risk_df, decimals=3),
        use_container_width=True,
        hide_index=False,
    )

    from src.visualizations import risk_evolution_bar
    fig_risk = risk_evolution_bar(risk_df)
    st.plotly_chart(fig_risk, use_container_width=True)

# ── Moment comparison table ───────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Pooled Return Distribution Moments by Period"), unsafe_allow_html=True)
mom_comp = get_period_moments_comp(subperiods)
if not mom_comp.empty:
    st.dataframe(
        round_numeric_df(mom_comp, decimals=4),
        use_container_width=True,
        hide_index=False,
    )

# ═══════════════════════════════════════════════════════════════════
# Section 2: KDE Overlay Comparison
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(section_header("Distribution Comparison: KDE Overlays"), unsafe_allow_html=True)

from src.visualizations import stability_kde_overlay

for metric_name, metric_key in [
    ("Pooled Stock Returns", "returns"),
    ("Per-stock Variance", "variance"),
    ("Per-stock Skewness", "skewness"),
    ("Per-stock Kurtosis", "kurtosis"),
]:
    with st.expander(f"📊 {metric_name} — KDE Overlay", expanded=(metric_key == "returns")):
        fig_kde = stability_kde_overlay(
            subperiods,
            metric=metric_key,
            title=f"{metric_name}: Pre-2020 vs. Post-2021 vs. Full Sample",
        )
        st.plotly_chart(fig_kde, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# Section 3: Rolling Index Moments Per Subperiod (Q8)
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(section_header("Q8 — Rolling Index Moments: Subperiod Comparison"), unsafe_allow_html=True)

from src.analytics import compute_rolling_moments
from src.preprocessing import get_index_returns

colors_map = {
    "Full Sample": THEME["blue"],
    "Pre-2020": THEME["green_bright"],
    "Post-2021": THEME["orange"],
}

for moment_key in ["mean", "variance", "skewness", "kurtosis"]:
    fig_roll = go.Figure()
    for period_name, ret in subperiods.items():
        try:
            idx = get_index_returns(ret)
        except KeyError:
            continue
        roll = compute_rolling_moments(idx)
        if moment_key not in roll.columns:
            continue
        s = roll[moment_key].dropna()
        fig_roll.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=period_name,
                line=dict(color=colors_map.get(period_name, THEME["text"]), width=1.5),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{moment_key}: %{{y:.4f}}<extra></extra>",
            )
        )
    fig_roll.update_layout(
        **dict(
            PLOTLY_LAYOUT_DEFAULTS,
            title=f"30-Day Rolling {moment_key.title()} of ^GSPC — All Periods",
            height=320,
            legend=dict(orientation="h", y=1.05),
        ),
    )
    st.plotly_chart(fig_roll, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# Section 4: KS Tests (Q9)
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(section_header("Q9 — Kolmogorov–Smirnov Stability Tests"), unsafe_allow_html=True)
st.markdown(
    "The KS test compares whether two empirical distributions could have been drawn from the "
    "same underlying distribution. We apply it to detect structural breaks between subperiods.",
)

# KS on stock returns / metrics
with st.spinner(f"Running KS tests on {ks_metric}…"):
    ks_df = get_ks_report(subperiods, metric=ks_metric)

if not ks_df.empty:
    ks_display = ks_df.copy()

    if "Reject H₀ (5%)" in ks_display.columns:
        ks_display["Reject H₀ (5%)"] = ks_display["Reject H₀ (5%)"].map(
            lambda x: "✅" if bool(x) else "❌"
        )

    for col in ["KS Stat", "p-value"]:
        if col in ks_display.columns:
            ks_display[col] = pd.to_numeric(ks_display[col], errors="coerce").round(4)

    for col in ["n_A", "n_B"]:
        if col in ks_display.columns:
            ks_display[col] = pd.to_numeric(ks_display[col], errors="coerce")

    st.dataframe(
        ks_display,
        use_container_width=True,
        hide_index=True,
    )

from src.visualizations import ks_results_heatmap
fig_ks = ks_results_heatmap(ks_df)
st.plotly_chart(fig_ks, use_container_width=True)

# KS on index returns
st.markdown("---")
st.markdown(section_header("KS Tests on ^GSPC Index Returns"), unsafe_allow_html=True)

with st.spinner("Running KS tests on ^GSPC…"):
    ks_idx_df = get_index_ks(subperiods)

if not ks_idx_df.empty:
    ks_idx_display = ks_idx_df.copy()
    for col in ks_idx_display.columns:
        if pd.api.types.is_numeric_dtype(ks_idx_display[col]):
            ks_idx_display[col] = ks_idx_display[col].round(4)
    st.dataframe(
        ks_idx_display,
        use_container_width=True,
        hide_index=False,
    )

# ── Per-stock KS test detail ──────────────────────────────────────
if "Pre-2020" in subperiods and "Post-2021" in subperiods:
    st.markdown("---")
    st.markdown(section_header("Per-Stock KS Test: Pre-2020 vs. Post-2021"), unsafe_allow_html=True)

    with st.spinner("Running per-stock KS tests…"):
        from src.statistics import ks_batch_stability
        from src.preprocessing import get_stock_returns

        ks_stock_df = ks_batch_stability(
            get_stock_returns(returns),
            get_stock_returns(subperiods["Pre-2020"]),
            get_stock_returns(subperiods["Post-2021"]),
        )

    pct_reject = ks_stock_df["reject_H0_5pct"].mean() * 100
    st.metric(
        "% stocks with significantly different distributions (pre vs. post)",
        f"{pct_reject:.1f}%",
        help="Fraction of stocks where KS test rejects H₀ at 5% significance level",
    )

    # Distribution of KS statistics
    from src.analytics import estimate_kde

    arr = ks_stock_df["ks_stat"].dropna().values
    if len(arr) >= 5:
        x, d = estimate_kde(arr, x_range=(0, 0.5))

        fig_ks_dist = go.Figure()
        fig_ks_dist.add_trace(
            go.Histogram(
                x=arr,
                nbinsx=50,
                histnorm="probability density",
                marker_color=THEME["orange"],
                opacity=0.4,
                name="Histogram",
            )
        )
        fig_ks_dist.add_trace(
            go.Scatter(
                x=x,
                y=d,
                mode="lines",
                line=dict(color=THEME["orange"], width=2.5),
                name="KDE",
            )
        )
        fig_ks_dist.add_vline(
            x=0.05,
            line_dash="dot",
            line_color=THEME["red_bright"],
            annotation_text="KS=0.05",
            annotation_font_color=THEME["red_bright"],
        )
        fig_ks_dist.update_layout(
            **dict(
                PLOTLY_LAYOUT_DEFAULTS,
                title="Distribution of Per-Stock KS Statistics (Pre-2020 vs. Post-2021)",
                xaxis_title="KS Statistic",
                yaxis_title="Density",
                height=350,
            ),
        )
        st.plotly_chart(fig_ks_dist, use_container_width=True)

    with st.expander("📋 Per-stock KS test table (top 50 by KS statistic)"):
        ks_stock_display = ks_stock_df.head(50).copy()
        for col in ks_stock_display.columns:
            if pd.api.types.is_numeric_dtype(ks_stock_display[col]):
                ks_stock_display[col] = ks_stock_display[col].round(4)
        if "reject_H0_5pct" in ks_stock_display.columns:
            ks_stock_display["reject_H0_5pct"] = ks_stock_display["reject_H0_5pct"].map(
                lambda x: "✅" if bool(x) else "❌"
            )
        st.dataframe(
            ks_stock_display,
            use_container_width=True,
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════════
# Section 5: Is the World Becoming More Risky?
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    section_header(
        "Is the World Becoming More Risky?",
        "Evidence from variance, tails, correlations, and distributional stability",
    ),
    unsafe_allow_html=True,
)

st.markdown(f"""
<div style="background:{THEME['bg_card']};border:1px solid {THEME['border']};
            border-radius:8px;padding:20px 24px;line-height:1.8;
            font-size:0.92rem;color:{THEME['text']};">

<strong style="color:{THEME['orange']}">Evidence FOR increasing risk (post-2020):</strong><br>
• <strong>Higher realised volatility</strong>: Annualised stock volatility is materially higher
  in the post-2021 period than pre-2020. The COVID shock permanently shifted the volatility baseline upward.<br>
• <strong>Fatter tails</strong>: Excess kurtosis is larger post-2021, meaning extreme return events
  are more frequent per unit time than before 2020.<br>
• <strong>Stronger correlations</strong>: Cross-sectional and sector correlations rose during the
  COVID period and remained elevated, reducing the benefit of diversification.<br>
• <strong>KS tests reject distributional equality</strong>: For the majority of stocks, the
  pre-2020 and post-2021 return distributions are statistically distinguishable — the world has
  structurally changed.<br>
• <strong>Larger max drawdowns</strong>: The 2020 and 2022 drawdowns exceeded anything observed
  in the 2016–2019 calm period, suggesting tail risk is more severe.<br><br>

<strong style="color:{THEME['green_bright']}">Evidence AGAINST (caveats):</strong><br>
• <strong>Mean-reversion of volatility</strong>: By 2024–2025, volatility had partially reverted
  toward pre-COVID levels. The "elevated" post-2021 average is partly driven by the 2020–2022 window.<br>
• <strong>Sample composition</strong>: Current S&P 500 constituents are the <em>survivors</em> —
  the most distressed firms were removed from the index. This survivorship bias understates true risk.<br>
• <strong>Structural breaks are normal</strong>: Every decade contains at least one major
  distributional shift. The 2016–2019 period was unusually calm; the "new normal" may simply be
  the historical average.<br>

</div>
""", unsafe_allow_html=True)

st.markdown(
    interpretation_box(
        "<strong>Conclusion (Q9)</strong>: The KS tests overwhelmingly reject the null hypothesis "
        "that pre-2020 and post-2021 return distributions are identical, for both the S&P 500 index "
        "and the majority of individual stocks. The post-2021 period features higher variance, "
        "heavier tails, and altered correlation structure. Whether this represents a permanent "
        "regime change or a temporary elevated-risk episode is uncertain — but the statistical "
        "evidence clearly shows the world is <em>currently</em> more risky than the 2016–2019 baseline."
    ),
    unsafe_allow_html=True,
)

# ── Report download ───────────────────────────────────────────────────────────
st.markdown("---")
try:
    from pathlib import Path

    report_path = Path(__file__).resolve().parent.parent / "report_summary.md"
    if report_path.exists():
        report_text = report_path.read_text()
        st.download_button(
            "⬇️  Download Academic Report (Markdown)",
            data=report_text,
            file_name="financial_econometrics_report.md",
            mime="text/markdown",
        )
except Exception:
    pass