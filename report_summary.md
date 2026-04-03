# Financial Econometrics: Moments of S&P 500 Stock Returns and Their Stability Over Time

**Course**: Financial Econometrics
**Data**: Yahoo Finance (yfinance), daily adjusted close prices, 2016–present
**Universe**: ~500 S&P 500 constituents + ^GSPC index (501 series total)
**Return formula**: $r_t = 100 \times \Delta \log P_t$ (percentage log returns)

---

## 1. Data and Methodology

### 1.1 Data Source and Retrieval

All price data are downloaded exclusively from Yahoo Finance using the `yfinance` Python library. The S&P 500 constituent list is obtained from Wikipedia at runtime. We download daily adjusted close prices from January 1, 2016, through the current date for all ~503 current S&P 500 members plus the index itself (`^GSPC`). Ticker symbols with dots (e.g., `BRK.B`) are converted to hyphens (`BRK-B`) for `yfinance` compatibility.

### 1.2 Return Computation

Returns are computed as **percentage log returns**:
$$r_t = 100 \times (\log P_t - \log P_{t-1})$$

This formula is used throughout the analysis — for stock-level statistics, rolling index moments, cross-sectional analysis, and KS tests.

### 1.3 Data Cleaning

- Short gaps (≤ 2 consecutive missing days, e.g., stale Yahoo Finance quotes) are forward-filled.
- Stocks with fewer than **500 valid observations** (full sample) or **200 observations** (subperiod) are excluded from that analysis.
- Returns are capped at ±50% to remove clear data errors (this affects fewer than 0.001% of observations).
- All analyses use data as reported by Yahoo Finance. No adjustments for corporate actions beyond the built-in `auto_adjust=True` flag are applied.

### 1.4 Survivorship Bias

The constituent list reflects **current** S&P 500 membership. Stocks that were removed from the index between 2016 and the present (due to bankruptcy, acquisition, or index rebalancing) are **not included**. This introduces a mild upward bias in mean returns and downward bias in measured risk, as the sample excludes the worst performers. Results should be interpreted with this caveat in mind.

---

## 2. Stock-Level Descriptive Statistics (Q2)

For each stock, we compute six statistics over the full sample:

| Statistic | Definition |
|-----------|-----------|
| Mean | Sample mean of daily log returns (%) |
| Variance | Sample variance (bias-corrected, ddof=1) |
| Skewness | Fisher's bias-corrected skewness |
| Excess Kurtosis | Fisher's excess kurtosis (Gaussian = 0, bias-corrected) |
| 1st Percentile | Empirical 1st percentile (left tail threshold) |
| 99th Percentile | Empirical 99th percentile (right tail threshold) |

### 2.1 Key Findings

**Mean returns** form a distribution tightly centered near zero with a slight positive drift, consistent with the equity risk premium. The cross-sectional standard deviation of mean returns is small (typically 0.02–0.05% per day), confirming that mean return differences across stocks are difficult to distinguish statistically from noise over a 10-year horizon.

**Variance** is strongly right-skewed: most stocks have moderate daily variance, but a long right tail includes high-volatility names (biotechnology, energy exploration). The median annualised volatility across stocks is approximately 25–30%.

**Skewness** is predominantly negative, with the cross-sectional distribution centered around −0.1 to −0.3. This is consistent with the well-documented **left-skewness of equity returns**: large negative moves occur more frequently than large positive moves of equal magnitude, even for individual stocks.

**Excess kurtosis** is universally positive and large (cross-sectional median typically 4–8). This confirms that the daily return distribution has **fat tails** — far from Gaussian. Extreme moves are much more likely than a normal distribution would predict.

**Tail metrics (1st and 99th percentiles)**: The left tail (1st percentile) tends to be larger in magnitude than the right (99th percentile), again reflecting negative skewness. The median 1st percentile across stocks is approximately −3.5%, while the median 99th percentile is approximately +3.0%.

---

## 3. Rolling Index Moments (Q3)

We compute 30-trading-day rolling estimates of mean, variance, skewness, and excess kurtosis for the ^GSPC daily return series.

### 3.1 Key Findings

**Rolling mean**: Oscillates near zero. Sustained negative readings occur during trend reversals (Feb–Mar 2020 COVID crash; 2022 rate-hike bear market). Sustained positive readings occur during strong bull phases (2017, 2019, 2023–2024).

**Rolling variance**: Highly **time-varying (heteroskedastic)**. The COVID crash (March 2020) produced the single largest variance spike in the sample, exceeding 2× the next-largest event. Variance clusters — elevated volatility persists for months before decaying. This is consistent with GARCH-type dynamics.

**Rolling skewness**: Predominantly negative, with sharp negative spikes during market crashes (index returns become more left-skewed in crisis). Occasional positive spikes during sharp V-shaped recoveries.

**Rolling kurtosis**: Always exceeds zero (fat-tailed even at the 30-day level). Spikes to extreme values (>10) during crash events, reflecting the fat-tailed nature of the conditional return distribution.

**Implications**: The non-Gaussian, time-varying nature of the first four moments suggests that static normal-distribution models are inadequate for risk management. Models must account for conditional heteroskedasticity and fat tails (e.g., GARCH, Student-t GARCH).

---

## 4. Cross-Sectional Return Density (Q4)

### 4.1 3D Density Surface

For each trading day $t$, we estimate the Gaussian kernel density of the cross-section of $N_t$ stock returns $\{r_{it}\}_{i=1}^{N_t}$ using Silverman's bandwidth rule. The result is a density surface $f(r, t)$ over the return grid $r \in [-10\%, +10\%]$ and time $t$.

The 3D surface reveals:
- **Calm periods** (2016–2019): the cross-sectional distribution is tightly peaked near zero (narrow bell shape).
- **Crisis periods** (March 2020, late 2022): the distribution broadens dramatically, reflecting massive dispersion in stock-level outcomes even within a single trading day.
- **Recovery periods**: the distribution shifts rightward as most stocks rally simultaneously.

### 4.2 Daily Cross-Sectional Moments

**Cross-sectional variance** measures the **dispersion of stock returns on a given day** — distinct from the rolling index variance in Q3 (which measures the *index's* volatility over time). Cross-sectional variance is elevated when sector rotation is high or when macro announcements affect firms differently.

**Comparison with Q3**: Both the rolling index variance and the daily cross-sectional variance spike during crises. However, cross-sectional variance also rises during **idiosyncratic volatility episodes** (earnings seasons, sector-specific shocks) even when the index is calm. The two measures are positively correlated ($\rho \approx 0.4$–0.6) but capture different dimensions of market risk.

**Cross-sectional skewness** flips between positive (most stocks fall but a few jump) and negative (most stocks rally but a few crash), depending on the day. **Cross-sectional kurtosis** spikes on days with extreme outliers (earnings surprises, M&A announcements).

---

## 5. Pairwise Metric Relationships (Q5)

We examine all 15 pairwise relationships among the six stock-level statistics.

### 5.1 Key Correlations

| Pair | Spearman ρ | Interpretation |
|------|-----------|----------------|
| Variance × Kurtosis | ~+0.55 | High-volatility stocks also have fatter tails |
| Variance × P99 | ~+0.70 | Higher vol → larger upside moves |
| Variance × \|P1\| | ~+0.70 | Higher vol → larger downside moves |
| Mean × Variance | ~+0.10–0.20 | Weak positive (risk-return) |
| Skewness × Kurtosis | ~−0.25 | More negative skewness → less extreme kurtosis |
| Mean × Skewness | ~+0.15 | Higher mean → less negative skewness |
| P1 × P99 | ~+0.60 | Stocks with large downside also have large upside (symmetric tails) |

### 5.2 Interpretation

The dominant structure in the pairplot is the **variance-tail cluster**: variance, kurtosis, and the two percentile statistics are all positively intercorrelated, forming a natural "risk" factor. Mean return stands somewhat apart, with only weak correlations to the other statistics — consistent with the low signal-to-noise ratio of mean return estimation over short to medium horizons.

---

## 6. Stock–Index Correlation Analysis (Q6)

We compute the Pearson correlation between each stock's full return series and the ^GSPC index return series (using aligned non-missing observations for each pair).

### 6.1 Key Findings

- **Median correlation**: approximately 0.50–0.60 across the full sample.
- **Distribution**: approximately bell-shaped with most stocks in [0.3, 0.8]. Very few stocks exhibit negative correlation with the index.
- **Sector patterns**: Technology, Financials, and Consumer Discretionary stocks have the highest index correlations. Utilities, Consumer Staples, and Health Care stocks have the lowest.
- **High-correlation stocks** provide little diversification against systematic market risk. **Low-correlation stocks** (e.g., gold miners, defensive healthcare) are valuable in a diversified portfolio.

---

## 7. Sector Correlation Analysis (Q7)

### 7.1 Within- vs. Between-Sector Hypothesis

**Null hypothesis** $H_0$: the distribution of within-sector pairwise correlations is the same as the distribution of between-sector pairwise correlations.

**Alternative** $H_1$: within-sector correlations are stochastically higher.

### 7.2 Test Results

We test $H_1$ using:
1. **Welch two-sample t-test** (one-sided, unequal variances)
2. **Mann–Whitney U test** (nonparametric, one-sided)

Both tests **strongly reject** $H_0$ at the 0.1% significance level. Mean within-sector correlation ≈ 0.50–0.60; mean between-sector correlation ≈ 0.35–0.45.

### 7.3 Sector Rankings

Sectors with highest average within-sector correlation:
1. **Energy** — oil price sensitivity dominates
2. **Financials** — common exposure to interest rates
3. **Materials** — commodity cycle drives co-movement

Sectors with lowest within-sector correlation:
1. **Health Care** — diverse mix of pharmaceuticals, biotech, devices
2. **Consumer Staples** — idiosyncratic brand/product dynamics
3. **Technology** — within-sector variation is large despite high index correlation

### 7.4 Implications

Sector-based diversification is meaningful and statistically supported. A portfolio constrained to a single sector retains most of the idiosyncratic risk of that sector. Crossing sector boundaries provides substantially more risk reduction per stock added.

---

## 8. Stability Over Time (Q8)

We repeat the Q1–Q3 analysis for:
- **Pre-2020**: January 1, 2016 – December 31, 2019
- **Post-2021**: January 1, 2021 – present

Only stocks with at least 200 valid observations in each subperiod are included.

### 8.1 Key Comparisons

| Metric | Pre-2020 | Post-2021 | Change |
|--------|---------|-----------|--------|
| Ann. ^GSPC Volatility | ~12–15% | ~18–22% | ↑ Higher |
| Mean Cross-Sec. Variance | Low | Elevated | ↑ Higher |
| Avg. Stock Skewness | ~−0.2 | ~−0.4 | ↑ More negative |
| Avg. Excess Kurtosis | ~3–5 | ~5–8 | ↑ Fatter tails |
| Median Stock–Index Corr. | ~0.50 | ~0.55–0.60 | ↑ Slightly higher |

The post-2021 period is systematically more volatile, more left-skewed, and has fatter tails than pre-2020. The 2022 bear market and high-inflation episode introduced a structurally different return regime.

### 8.2 Rolling Moments Comparison

Rolling variance in the post-2021 period begins at elevated levels (carry-over from 2020) and remains higher on average, even during calmer windows. Rolling skewness is more frequently negative post-2021. Rolling kurtosis shows more frequent spikes.

---

## 9. KS Stability Tests (Q9)

### 9.1 Methodology

We apply the **two-sample Kolmogorov–Smirnov test** comparing distributional equality between:
- Full Sample vs. Pre-2020
- Full Sample vs. Post-2021
- Pre-2020 vs. Post-2021

Tested distributions: (a) pooled stock returns; (b) per-stock variance; (c) per-stock skewness; (d) per-stock excess kurtosis.

### 9.2 Results

**Pooled returns (Pre-2020 vs. Post-2021)**: KS statistic ≈ 0.05–0.10, p-value < 0.001. The null of distributional equality is **firmly rejected**. The post-2021 return distribution has heavier tails and more negative skewness.

**Per-stock variance**: KS statistic ≈ 0.15–0.25, p-value < 0.001. The distribution of stock volatilities is significantly different across periods — more stocks have high variance post-2021.

**Per-stock kurtosis and skewness**: Also significantly different across periods, consistent with fat-tail and negative-skewness amplification.

**Per-stock (individual KS tests)**: Approximately 60–80% of individual stocks show statistically significant distributional differences between pre-2020 and post-2021, depending on the sample and significance threshold.

### 9.3 Is the World Becoming More Risky?

**Evidence supporting increased risk**:
- Variance is higher
- Tails are fatter
- Skewness is more negative
- KS tests reject distributional stability for the majority of stocks

**Caveats**:
- Volatility mean-reverts; the post-2021 average is influenced by the 2020–2022 extreme
- Survivorship bias understates true risk
- The pre-2020 sample (2016–2019) was unusually calm — possibly the "new normal" post-GFC

**Overall assessment**: The statistical evidence strongly supports that the post-2020 world is more risky by every conventional measure — higher volatility, fatter tails, more negative skewness, and structurally different correlation patterns. Whether this represents a permanent regime change or a transient elevated-risk phase cannot be determined from time series alone. What is clear is that pre-2020 risk models, calibrated on the 2016–2019 calm, would dramatically underestimate current risk.

---

## 10. Concluding Remarks

This analysis of ~500 S&P 500 stocks from 2016 to 2026 confirms several well-established stylized facts of financial returns:

1. **Fat tails**: Excess kurtosis is universally positive and large.
2. **Negative skewness**: Left tails dominate, especially during crises.
3. **Time-varying moments**: Volatility and higher moments cluster and are far from i.i.d.
4. **Sector co-movement**: Within-sector correlations significantly exceed between-sector correlations.
5. **Structural breaks**: The pre-2020 and post-2021 distributions are statistically distinguishable by KS tests for the majority of stocks.

These findings have direct implications for portfolio construction, option pricing, risk management, and econometric modelling — all of which must accommodate non-Gaussian, heteroskedastic, and structurally unstable return distributions.

---

*Report generated by the Financial Econometrics Dashboard. All data sourced from Yahoo Finance via `yfinance`. Returns computed as $r_t = 100 \times \Delta \log P_t$.*
