"""
3_Rolling_Index_Moments.py
--------------------------
30-day rolling moments of the S&P 500 index return series.

Homework Q3: time series plots of rolling mean, variance, skewness, kurtosis
for ^GSPC daily returns using a 30-trading-day window.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    THEME,
    ROLLING_WINDOW,
    INDEX_TICKER,
    PLOTLY_LAYOUT_DEFAULTS,
)
from src.utils import section_header, interpretation_box, page_css, page_header_html
from src.visualizations import rolling_moments_plot
from src.analytics import estimate_kde

st.set_page_config(page_title="Rolling Index Moments | FinEC", page_icon="📈", layout="wide")
st.markdown(page_css(), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_index_returns():
    from src.preprocessing import load_cached_returns, get_index_returns
    r = load_cached_returns()
    if r is None:
        return None
    return get_index_returns(r)


@st.cache_data(show_spinner=False)
def compute_rolling(_idx_ret, window):
    from src.analytics import compute_rolling_moments
    return compute_rolling_moments(_idx_ret, window=window)


def add_vertical_marker(fig, dt, label, color):
    """
    Safe replacement for fig.add_vline with Timestamp x-values.
    Avoids Plotly's internal timestamp arithmetic bug.
    """
    dt = pd.Timestamp(dt)

    fig.add_shape(
        type="line",
        x0=dt,
        x1=dt,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color=color,
            width=1,
            dash="dot",
        ),
        opacity=0.7,
    )

    fig.add_annotation(
        x=dt,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        textangle=-90,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=8, color=color),
        bgcolor=THEME["bg_card"],
        bordercolor=THEME["border"],
        borderpad=2,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    page_header_html(
        "Rolling Index Moments",
        f"Q3 — {ROLLING_WINDOW}-day rolling mean, variance, skewness & kurtosis of {INDEX_TICKER} daily returns",
        "📈",
    ),
    unsafe_allow_html=True,
)

idx_ret = load_index_returns()
if idx_ret is None:
    st.warning("⚠️ No data cached. Go to **Data Pipeline** first.")
    st.stop()

if len(idx_ret) == 0:
    st.warning("⚠️ Index return series is empty.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Controls")
    window = st.slider("Rolling window (trading days)", 10, 120, ROLLING_WINDOW, step=5)
    date_range = st.date_input(
        "Date range",
        value=(idx_ret.index.min().date(), idx_ret.index.max().date()),
        min_value=idx_ret.index.min().date(),
        max_value=idx_ret.index.max().date(),
    )
    show_events = st.checkbox("Show market events annotations", value=True)

# Filter by date range
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    idx_filtered = idx_ret[
        (idx_ret.index >= pd.Timestamp(start_d)) &
        (idx_ret.index <= pd.Timestamp(end_d))
    ]
else:
    idx_filtered = idx_ret.copy()

if len(idx_filtered) == 0:
    st.warning("⚠️ No observations in the selected date range.")
    st.stop()

rolling_df = compute_rolling(idx_filtered, window)

# ── Summary metrics ───────────────────────────────────────────────────────────
valid = rolling_df.dropna()
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Latest Roll. Mean (%)", f"{valid['mean'].iloc[-1]:.4f}" if len(valid) else "—")
with c2:
    st.metric("Latest Roll. Variance (%²)", f"{valid['variance'].iloc[-1]:.4f}" if len(valid) else "—")
with c3:
    st.metric("Latest Roll. Skewness", f"{valid['skewness'].iloc[-1]:.4f}" if len(valid) else "—")
with c4:
    st.metric("Latest Roll. Kurtosis", f"{valid['kurtosis'].iloc[-1]:.4f}" if len(valid) else "—")

# ── Rolling moments chart ─────────────────────────────────────────────────────
st.markdown("---")

fig = rolling_moments_plot(
    rolling_df,
    title=f"{window}-Day Rolling Moments — S&P 500 Index ({INDEX_TICKER})",
)

# Add key event annotations safely
if show_events and len(rolling_df) > 0:
    events = {
        "COVID crash": "2020-02-24",
        "COVID recovery": "2020-04-06",
        "2022 rate hikes": "2022-01-05",
        "SVB collapse": "2023-03-10",
    }
    x_min = pd.Timestamp(rolling_df.index.min())
    x_max = pd.Timestamp(rolling_df.index.max())

    for label, date_str in events.items():
        dt = pd.Timestamp(date_str)
        if x_min <= dt <= x_max:
            add_vertical_marker(fig, dt=dt, label=label, color=THEME["yellow"])

st.plotly_chart(fig, use_container_width=True)

# ── Individual metric detail ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Individual Moment Deep-Dive"), unsafe_allow_html=True)

selected_moment = st.selectbox(
    "Select moment for detailed view",
    ["mean", "variance", "skewness", "kurtosis"],
    format_func=lambda x: {
        "mean": "Mean",
        "variance": "Variance",
        "skewness": "Skewness",
        "kurtosis": "Excess Kurtosis",
    }[x],
)

moment_color = {
    "mean": THEME["blue"],
    "variance": THEME["orange"],
    "skewness": THEME["purple"],
    "kurtosis": THEME["green_bright"],
}

fig2 = make_subplots(
    rows=2,
    cols=1,
    row_heights=[0.65, 0.35],
    shared_xaxes=True,
    vertical_spacing=0.04,
)

s = rolling_df[selected_moment].dropna()
fig2.add_trace(
    go.Scatter(
        x=s.index,
        y=s.values,
        mode="lines",
        line=dict(color=moment_color[selected_moment], width=1.8),
        name=selected_moment.title(),
        hovertemplate="%{x|%Y-%m-%d}<br>" + selected_moment + ": %{y:.4f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Add rolling IQR band
iqr_lo = rolling_df[selected_moment].rolling(60, min_periods=20).quantile(0.25)
iqr_hi = rolling_df[selected_moment].rolling(60, min_periods=20).quantile(0.75)

fig2.add_trace(
    go.Scatter(
        x=iqr_hi.index,
        y=iqr_hi.values,
        line=dict(width=0),
        showlegend=False,
        mode="lines",
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)

# Safer rgba instead of hex+"22"
fill_rgba = {
    "mean": "rgba(88,166,255,0.15)",
    "variance": "rgba(255,166,87,0.15)",
    "skewness": "rgba(188,140,255,0.15)",
    "kurtosis": "rgba(63,185,80,0.15)",
}

fig2.add_trace(
    go.Scatter(
        x=iqr_lo.index,
        y=iqr_lo.values,
        fill="tonexty",
        fillcolor=fill_rgba[selected_moment],
        line=dict(width=0),
        showlegend=False,
        mode="lines",
        name="IQR band",
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)

# Zero line on top panel
fig2.add_hline(
    y=0,
    line_dash="dot",
    line_color=THEME["text_dim"],
    line_width=1,
    row=1,
    col=1,
)

# Raw index returns below
fig2.add_trace(
    go.Bar(
        x=idx_filtered.index,
        y=idx_filtered.values,
        marker_color=np.where(idx_filtered.values >= 0, THEME["green"], THEME["red"]),
        name=f"{INDEX_TICKER} daily return",
        opacity=0.7,
        hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:+.4f}%<extra></extra>",
    ),
    row=2,
    col=1,
)

# Event markers on detailed chart too
if show_events and len(rolling_df) > 0:
    x_min = pd.Timestamp(rolling_df.index.min())
    x_max = pd.Timestamp(rolling_df.index.max())
    events = {
        "COVID crash": "2020-02-24",
        "COVID recovery": "2020-04-06",
        "2022 rate hikes": "2022-01-05",
        "SVB collapse": "2023-03-10",
    }
    for label, date_str in events.items():
        dt = pd.Timestamp(date_str)
        if x_min <= dt <= x_max:
            add_vertical_marker(fig2, dt=dt, label=label, color=THEME["yellow"])

fig2.update_layout(
    **dict(
        PLOTLY_LAYOUT_DEFAULTS,
        height=550,
        showlegend=False,
        title=dict(
            text=f"Rolling {selected_moment.title()} with {INDEX_TICKER} Daily Returns",
            font=dict(size=14),
        ),
    ),
)
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=True, gridcolor=THEME["border"])
st.plotly_chart(fig2, use_container_width=True)

# ── Distribution of rolling values ────────────────────────────────────────────
st.markdown("---")
st.markdown(section_header("Distribution of Rolling Estimates"), unsafe_allow_html=True)

arr = rolling_df[selected_moment].dropna().values

if len(arr) >= 5:
    x, d = estimate_kde(arr)

    fig3 = go.Figure()
    fig3.add_trace(
        go.Histogram(
            x=arr,
            nbinsx=60,
            histnorm="probability density",
            marker_color=moment_color[selected_moment],
            opacity=0.4,
            name="Histogram",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=x,
            y=d,
            mode="lines",
            line=dict(color=moment_color[selected_moment], width=2.5),
            name="KDE",
        )
    )
    fig3.add_vline(x=0, line_dash="dot", line_color=THEME["text_dim"], line_width=1)

    fig3.update_layout(
        **dict(
            PLOTLY_LAYOUT_DEFAULTS,
            title=f"Distribution of {window}-Day Rolling {selected_moment.title()}",
            xaxis_title=selected_moment.title(),
            yaxis_title="Density",
            height=350,
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Not enough rolling observations to estimate the distribution for the selected window.")

# ── Interpretation ────────────────────────────────────────────────────────────
st.markdown(
    interpretation_box(
        "<strong>Rolling Mean</strong>: Fluctuates near zero with occasional excursions during "
        "trending markets. The COVID crash (Feb–Mar 2020) and the 2022 bear market show "
        "sustained negative rolling means. <br><br>"
        "<strong>Rolling Variance</strong>: Highly time-varying (heteroskedastic). "
        "Variance spikes sharply during crises and decays slowly, showing volatility clustering. "
        "Post-2021 variance is typically higher than during calmer pre-2020 periods.<br><br>"
        "<strong>Rolling Skewness</strong>: Often turns negative during market stress, reflecting "
        "asymmetric downside risk. Positive spikes can appear during rebound phases. <br><br>"
        "<strong>Rolling Kurtosis</strong>: Frequently exceeds the Gaussian benchmark of zero, "
        "with sharp spikes during crash episodes, confirming time-varying fat tails."
    ),
    unsafe_allow_html=True,
)

# ── Download ──────────────────────────────────────────────────────────────────
csv = rolling_df.reset_index().to_csv(index=False)
st.download_button(
    "⬇️  Download rolling moments (CSV)",
    data=csv,
    file_name="rolling_index_moments.csv",
    mime="text/csv",
)