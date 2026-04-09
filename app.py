"""
app.py
------
Main Streamlit entrypoint — Data Pipeline page.

Navigation to other analytical pages is via the Streamlit sidebar.
This page handles first-run data ingestion with a progress bar, then
shows a high-level data quality summary.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import THEME, PRICES_CACHE, RETURNS_CACHE, SECTOR_CACHE, INDEX_TICKER
from src.utils import section_header, interpretation_box, metric_card_html, page_css, page_header_html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinEC | S&P 500 Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(page_css(), unsafe_allow_html=True)


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600 * 12)
def cached_load_prices(force: bool = False):
    from src.data_loader import load_prices
    return load_prices(force_refresh=force)


@st.cache_data(show_spinner=False, ttl=3600 * 12)
def cached_load_returns(force: bool = False):
    from src.preprocessing import (
        compute_returns, clean_returns,
        save_returns, load_cached_returns,
    )
    if not force:
        cached = load_cached_returns()
        if cached is not None:
            return cached
    prices  = cached_load_prices(force=force)
    returns = compute_returns(prices)
    returns = clean_returns(returns)
    save_returns(returns)
    return returns


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def cached_sector_info(tickers: tuple, force: bool = False):
    from src.data_loader import load_sector_info
    return load_sector_info(list(tickers), force_refresh=force)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown(
        f"""
        <div style="padding:12px 0 16px 0">
            <div style="font-size:1.05rem;font-weight:700;
                        background:linear-gradient(90deg,#388bfd,#2ea043);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                📈 FinEC Dashboard
            </div>
            <div style="font-size:0.73rem;color:{THEME['text_dim']};margin-top:4px">
                Financial Econometrics · S&P 500 Return Moments
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    force_refresh = st.sidebar.button("🔄 Force Refresh Data", use_container_width=True)
    return force_refresh


# ── Main page ─────────────────────────────────────────────────────────────────

def main():
    force_refresh = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        page_header_html(
            "S&P 500 Return Moments Dashboard",
            "Financial Econometrics · Daily adjusted close prices from Yahoo Finance · Percentage log returns · 501 series",
            "📈",
        ),
        unsafe_allow_html=True,
    )

    # ── Data pipeline section ─────────────────────────────────────────────────
    st.markdown(section_header("Data Pipeline", "Download, clean, and cache S&P 500 price data"), unsafe_allow_html=True)

    cache_ok = PRICES_CACHE.exists() and RETURNS_CACHE.exists()
    col1, col2, col3 = st.columns(3)

    with col1:
        status = "✅ Cached" if PRICES_CACHE.exists() else "⏳ Not downloaded"
        st.markdown(metric_card_html("Prices Cache", status), unsafe_allow_html=True)
    with col2:
        status = "✅ Cached" if RETURNS_CACHE.exists() else "⏳ Not computed"
        st.markdown(metric_card_html("Returns Cache", status), unsafe_allow_html=True)
    with col3:
        status = "✅ Cached" if SECTOR_CACHE.exists() else "⏳ Not fetched"
        st.markdown(metric_card_html("Sector Metadata", status), unsafe_allow_html=True)

    # Download button or auto-load
    if not cache_ok or force_refresh:
        st.info(
            "⬇️  First-run: downloading ~503 tickers from Yahoo Finance. "
            "This takes 5–15 min. Please wait.",
            icon="ℹ️",
        )
        prog_bar = st.progress(0, text="Initializing…")

        def cb(frac: float, msg: str):
            prog_bar.progress(min(frac, 1.0), text=msg)

        with st.spinner("Downloading prices from Yahoo Finance…"):
            try:
                from src.data_loader import load_prices
                prices = load_prices(force_refresh=True, progress_callback=cb)
                st.success(f"✅ Prices downloaded: {prices.shape[0]:,} trading days × {prices.shape[1]} tickers")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                st.stop()

        prog_bar.progress(0.7, "Computing log returns…")
        from src.preprocessing import compute_returns, clean_returns, save_returns
        returns = compute_returns(prices)
        returns = clean_returns(returns)
        save_returns(returns)
        prog_bar.progress(1.0, "Done ✓")
        st.cache_data.clear()
        st.rerun()

    # Load from cache
    with st.spinner("Loading cached data…"):
        returns = cached_load_returns(force=False)

    # ── Summary stats ─────────────────────────────────────────────────────────
    from src.preprocessing import get_stock_returns, get_index_returns, return_summary
    stock_ret = get_stock_returns(returns)
    try:
        idx_ret = get_index_returns(returns)
    except KeyError:
        idx_ret = pd.Series(dtype=float)

    st.markdown("---")
    st.markdown(section_header("Dataset Overview"), unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Trading Days",    f"{len(returns):,}")
    with c2:
        st.metric("Stocks",          f"{stock_ret.shape[1]:,}")
    with c3:
        st.metric("Total Series",    f"{returns.shape[1]:,}")
    with c4:
        start = str(returns.index.min().date())
        st.metric("Start Date", start)
    with c5:
        end = str(returns.index.max().date())
        st.metric("End Date", end)

    st.markdown("---")
    # Return summary table
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Data Quality Summary**")
        st.dataframe(return_summary(returns), use_container_width=True, hide_index=True)
    with col_r:
        st.markdown("**S&P 500 Index (^GSPC) Return Statistics**")
        if len(idx_ret) > 0:
            stats = {
                "Mean daily return (%)":    f"{idx_ret.mean():.4f}",
                "Std dev (%)":              f"{idx_ret.std():.4f}",
                "Ann. volatility (%)":      f"{idx_ret.std()*np.sqrt(252):.2f}",
                "Skewness":                 f"{idx_ret.skew():.4f}",
                "Excess kurtosis":          f"{idx_ret.kurt():.4f}",
                "Min daily return (%)":     f"{idx_ret.min():.4f}",
                "Max daily return (%)":     f"{idx_ret.max():.4f}",
                "1st percentile (%)":       f"{np.percentile(idx_ret.dropna(), 1):.4f}",
                "99th percentile (%)":      f"{np.percentile(idx_ret.dropna(), 99):.4f}",
            }
            st.dataframe(
                pd.DataFrame(list(stats.items()), columns=["Statistic", "Value"]),
                use_container_width=True,
                hide_index=True,
            )

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown(
        interpretation_box(
            "The dataset covers 501 series (500 S&P 500 constituents + ^GSPC index) from 2016 to present. "
            "Percentage log returns are computed as <em>r<sub>t</sub> = 100 × Δ log P<sub>t</sub></em>. "
            "Stocks with fewer than 500 valid observations are excluded from the full-sample analysis; "
            "a lower threshold of 200 observations is applied for subperiod analyses. "
            "Note: the constituent list reflects <em>current</em> S&P 500 membership, introducing a mild "
            "survivorship bias — all stocks in the sample have survived to the present day."
        ),
        unsafe_allow_html=True,
    )

    # ── Missing data heatmap (sample) ─────────────────────────────────────────
    with st.expander("📋 Availability heatmap (sample of 60 tickers)"):
        sample_cols = stock_ret.columns[:60].tolist()
        avail = stock_ret[sample_cols].notna().astype(int)
        # Downsample dates for display
        step = max(1, len(avail) // 200)
        avail_ds = avail.iloc[::step]

        import plotly.graph_objects as go
        fig = go.Figure(
            go.Heatmap(
                z=avail_ds.T.values,
                x=[str(d.date()) for d in avail_ds.index],
                y=sample_cols,
                colorscale=[[0, THEME["red"]], [1, THEME["green"]]],
                showscale=False,
                hovertemplate="%{y}<br>%{x}<br>Available: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            height=500,
            title="Data Availability (green = present, red = missing)",
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["bg_secondary"],
            font=dict(color=THEME["text"], size=8),
            margin=dict(l=80, r=20, t=50, b=20),
            xaxis=dict(tickangle=45, tickfont=dict(size=7)),
            yaxis=dict(tickfont=dict(size=7)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Sector fetch (deferred, background) ───────────────────────────────────
    with st.expander("🏢 Load sector metadata (needed for sector analysis pages)"):
        if SECTOR_CACHE.exists():
            sector_df = pd.read_parquet(SECTOR_CACHE)
            st.success(f"✅ Sector metadata loaded: {len(sector_df)} tickers")
            sectors = sector_df["sector"].value_counts()
            st.bar_chart(sectors)
        else:
            if st.button("Fetch sector metadata from Yahoo Finance (~5 min)"):
                tickers_tuple = tuple(stock_ret.columns.tolist())
                with st.spinner("Fetching sector info…"):
                    sector_df = cached_sector_info(tickers_tuple, force=True)
                st.success(f"✅ Fetched {len(sector_df)} tickers")
                st.rerun()

    st.markdown("---")
    st.markdown(
        f"<div style='color:{THEME['text_dim']};font-size:0.8rem;text-align:center'>"
        "Data sourced from Yahoo Finance via yfinance · "
        "Use the sidebar to navigate to analytical pages"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
