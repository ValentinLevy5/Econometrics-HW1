# Financial Econometrics — S&P 500 Return Moments Dashboard

A production-quality interactive analytics dashboard for a university Financial Econometrics course. Covers all nine homework questions on the moments of stock returns and their stability over time, using data sourced exclusively from Yahoo Finance.

---

## Project Overview

This Streamlit application downloads and analyzes daily adjusted close prices for all S&P 500 constituents and the S&P 500 index (`^GSPC`) from 2016 onward. It delivers:

- Finviz-inspired interactive market treemap
- Stock-level descriptive statistics with KDE visualizations
- 30-day rolling moments of index returns
- 3D cross-sectional return density surface over time
- Pairwise metric scatterplots
- Stock–index correlation analysis
- Sector-level within vs. between correlation testing
- Stability analysis across subperiods (KS tests)
- PDF-exportable written report

---

## Project Structure

```
project/
├── app.py                    # Main Streamlit entrypoint (Overview page)
├── requirements.txt
├── README.md
├── report_summary.md         # Written academic report for PDF export
├── data/                     # Raw parquet downloads (auto-created)
├── cache/                    # Computed intermediates (auto-created)
├── src/
│   ├── __init__.py
│   ├── config.py             # App-wide constants, colors, paths
│   ├── data_loader.py        # Yahoo Finance data ingestion & caching
│   ├── preprocessing.py      # Price cleaning, log-return computation
│   ├── analytics.py          # Moments, KDE, rolling stats, cross-sections
│   ├── statistics.py         # Correlation computation, KS tests
│   ├── sector_analysis.py    # Sector grouping, within/between correlations
│   ├── stability.py          # Subperiod KS tests, risk evolution
│   ├── visualizations.py     # All Plotly chart builders
│   └── utils.py              # Shared helpers
└── pages/
    ├── 1_Overview.py
    ├── 2_Stock_Moments.py
    ├── 3_Rolling_Index_Moments.py
    ├── 4_Cross_Sectional_Density.py
    ├── 5_Metric_Relationships.py
    ├── 6_Correlation_Analysis.py
    └── 7_Stability_Analysis.py
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Internet connection (for Yahoo Finance data)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## First-Run Behavior

On first run, the app will:
1. Fetch the current S&P 500 constituent list from Wikipedia
2. Download ~10 years of adjusted close prices from Yahoo Finance for all ~503 tickers + `^GSPC`
3. Compute percentage log returns and descriptive statistics
4. Cache all results as Parquet files in `data/` and `cache/`

**This initial download takes approximately 5–15 minutes depending on your connection.** Subsequent runs are near-instant thanks to persistent caching.

Progress is shown via a Streamlit progress bar on the Data Pipeline page.

---

## Data Notes

- **Source**: Yahoo Finance via `yfinance`. All prices are adjusted close.
- **Returns**: `r_t = 100 × Δ log(P_t)` (percentage log returns)
- **Index**: `^GSPC` (S&P 500 index)
- **Ticker normalization**: Wikipedia tickers with `.` are converted to `-` for yfinance (e.g., `BRK.B → BRK-B`, `BF.B → BF-B`)
- **Missing data**: Stocks with fewer than 252 trading days of data in any subperiod are excluded from that subperiod's analysis
- **Survivorship bias**: The constituent list is the *current* S&P 500. Stocks added after 2016 will have shorter histories, introducing a mild survivorship bias that is noted in the report

---

## App Pages

| Page | Homework Coverage |
|------|------------------|
| Overview | Market map, summary dashboard |
| Stock-Level Moments | Q2 — 6 metrics, KDE, ranked tables |
| Rolling Index Moments | Q3 — 30-day rolling mean/variance/skewness/kurtosis of ^GSPC |
| Cross-Sectional Density | Q4 — 3D density surface + daily cross-sectional moments |
| Metric Relationships | Q5 — Pairwise scatterplots of 6 stock-level metrics |
| Correlation Analysis | Q6, Q7 — Stock–index correlations + sector within/between tests |
| Stability Analysis | Q8, Q9 — Subperiod comparison, KS tests, risk evolution |

---

## Methodological Assumptions

1. **Percentage log returns** are used throughout: `100 × (log P_t − log P_{t-1})`
2. **Rolling window** for index moments: 30 trading days (≈ 1.5 calendar months)
3. **Cross-sectional KDE**: Gaussian kernel, bandwidth via Silverman's rule, evaluated on a fixed grid of 200 points per day. Computed every 5 trading days for performance, then interpolated for display
4. **Sector classification**: GICS sector from Yahoo Finance `info` metadata. Fallback: "Unknown" if unavailable
5. **Within-sector vs. between-sector correlation test**: Welch two-sample t-test and Mann–Whitney U test on the two distributions of pairwise correlations
6. **KS stability test**: Two-sample KS test comparing the distribution of a chosen metric (e.g., daily returns) between pre-2020 and post-2021 periods
7. **Subperiods**:
   - Full sample: 2016-01-01 to present
   - Pre-2020: 2016-01-01 to 2019-12-31
   - Post-2021: 2021-01-01 to present
8. **Market cap for treemap sizing**: fetched from Yahoo Finance `info.marketCap`; if unavailable, equal sizing is used as fallback

---

## Exporting the Report

Navigate to the **Stability Analysis** page → scroll to the bottom → click **Download Report (Markdown)**. Open the downloaded `.md` file and convert to PDF using any Markdown-to-PDF tool (e.g., Pandoc, Typora, or VS Code's Markdown PDF extension).

---

## Known Limitations

- Yahoo Finance data may have gaps for thinly traded or recently listed stocks
- The S&P 500 list is scraped from Wikipedia at runtime; it reflects current constituents, not historical membership
- Very long date ranges may trigger Yahoo Finance rate limits; the app retries with exponential backoff
- 3D density surface is computationally expensive; by default it samples every 5 trading days

---

## License

Academic use only.
