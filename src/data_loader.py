"""
data_loader.py
--------------
Yahoo Finance data ingestion, sector/market-cap metadata, and persistent
Parquet caching.  All price data comes exclusively from yfinance.

Public API
----------
get_sp500_tickers()      → list[str]          yfinance-normalized tickers
load_prices()            → pd.DataFrame        adjusted close prices
load_sector_info()       → pd.DataFrame        sector + market-cap metadata
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    INDEX_TICKER, START_DATE,
    PRICES_CACHE, SECTOR_CACHE,
    DATA_DIR,
)
from src.utils import logger, normalize_ticker, retry

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Hardcoded S&P 500 fallback ticker list ────────────────────────────────────
# Used if Wikipedia is unreachable. Current as of early 2025.
# BRK.B and BF.B are normalized to BRK-B and BF-B for yfinance.
_FALLBACK_SP500 = [
    "A","AAL","AAP","AAPL","ABBV","ABT","ACGL","ACN","ADBE","ADI","ADM","ADP",
    "ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALK",
    "ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN","ANET",
    "ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ATO","AVB","AVGO","AVY",
    "AWK","AXON","AXP","AZO","BA","BAC","BALL","BAX","BBWI","BBY","BDX","BEN",
    "BF-B","BG","BIIB","BIO","BK","BKNG","BKR","BLK","BLV","BMY","BR","BRK-B",
    "BRO","BSX","BX","C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE","CCI",
    "CCL","CDNS","CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF",
    "CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF","COO",
    "COP","COST","CPAY","CPB","CPRT","CPT","CRL","CRM","CSCO","CSX","CTAS",
    "CTLT","CTRA","CTSH","CTVA","CVS","CVX","CZR","D","DAL","DAY","DD","DE",
    "DECK","DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOC","DOV","DOW",
    "DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY","ECL","ED","EFX",
    "EIX","EL","ELV","EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ES",
    "ESS","ETN","ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG",
    "FAST","FCX","FDS","FDX","FE","FFIV","FI","FICO","FIS","FITB","FLT","FMC",
    "FOX","FOXA","FRT","FSLR","FTNT","FTV","GD","GE","GEHC","GEN","GEV","GILD",
    "GIS","GL","GLW","GM","GNRC","GOOGL","GOOG","GPC","GPN","GRMN","GS","GWW",
    "HAL","HAS","HBAN","HCA","HD","HES","HIG","HII","HLT","HOLX","HON","HPE",
    "HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX",
    "IFF","ILMN","INCY","INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","IT",
    "ITW","IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JNPR","JPM","K","KDP",
    "KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX","KO","KR","L",
    "LDOS","LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNT","LOW","LRCX","LUV",
    "LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO",
    "MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MO",
    "MOH","MOS","MPC","MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB",
    "MTCH","MTD","MU","NCLH","NDAQ","NEE","NEM","NFLX","NI","NKE","NOC","NOW",
    "NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI","O","OKE",
    "OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC","PAYX","PCAR",
    "PCG","PEAK","PEG","PEP","PFE","PFG","PGR","PH","PHM","PKG","PLD","PM",
    "PNC","PNR","PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSX","PTC","PWR",
    "PYPL","QCOM","QRVO","RCL","RE","REG","REGN","RF","RJF","RL","RMD","ROK",
    "ROL","ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW","SHW","SJM",
    "SLB","SMCI","SNA","SNPS","SO","SPG","SPGI","SRE","STE","STLD","STT","STX",
    "STZ","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH","TEL",
    "TER","TFC","TFX","TGT","TJX","TMO","TMUS","TPR","TRGP","TRMB","TROW",
    "TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT","TYL","UAL","UDR","UHS",
    "ULTA","UNH","UNP","UPS","URI","USB","V","VICI","VLO","VLTO","VMC","VRSK",
    "VRSN","VRTX","VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WELL","WFC",
    "WM","WMB","WMT","WRB","WST","WTW","WY","WYNN","XEL","XOM","XYL","YUM",
    "ZBH","ZBRA","ZTS",
]


# ── S&P 500 constituent list ──────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """
    Fetch the current S&P 500 constituent list from Wikipedia and normalize
    ticker symbols for yfinance compatibility.

    Wikipedia uses dots (e.g., BRK.B); yfinance expects hyphens (BRK-B).

    Returns
    -------
    list[str]
        Sorted list of yfinance-compatible ticker symbols, with ^GSPC prepended.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, header=0)
        df = tables[0]
        raw_tickers: list[str] = df["Symbol"].dropna().tolist()
        tickers = [normalize_ticker(t) for t in raw_tickers]
        tickers = sorted(set(tickers))
        logger.info("Retrieved %d S&P 500 tickers from Wikipedia.", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning(
            "Wikipedia fetch failed (%s) — using hardcoded fallback ticker list.", exc
        )
        tickers = sorted(set(_FALLBACK_SP500))
        logger.info("Fallback: using %d hardcoded S&P 500 tickers.", len(tickers))
        return tickers


# ── Price data download ───────────────────────────────────────────────────────

def _extract_close(raw: "pd.DataFrame", chunk: list[str]) -> "pd.DataFrame":
    """
    Extract the adjusted-close column(s) from a yfinance download result,
    handling both flat and MultiIndex column structures used across
    yfinance 0.1.x / 0.2.x / 1.x.

    yfinance 1.x returns a MultiIndex with (PriceType, Ticker) ordering.
    yfinance 0.2.x also uses (PriceType, Ticker).
    Single-ticker downloads may return flat columns.

    Returns
    -------
    pd.DataFrame  columns = ticker symbols, dtype float64
    """
    if raw.empty:
        return pd.DataFrame()

    cols = raw.columns

    # ── MultiIndex case ──────────────────────────────────────────────────────
    if isinstance(cols, pd.MultiIndex):
        lvl0 = list(cols.get_level_values(0).unique())
        lvl1 = list(cols.get_level_values(1).unique())

        # (PriceType, Ticker)  — most common in yfinance ≥ 0.2
        if "Close" in lvl0:
            close = raw["Close"].copy()
        # (Ticker, PriceType)  — older group_by='ticker' style
        elif "Close" in lvl1:
            close = raw.xs("Close", axis=1, level=1).copy()
        # Fallback: try 'Adj Close'
        elif "Adj Close" in lvl0:
            close = raw["Adj Close"].copy()
        elif "Adj Close" in lvl1:
            close = raw.xs("Adj Close", axis=1, level=1).copy()
        else:
            logger.warning("Could not locate Close in MultiIndex columns %s", lvl0[:5])
            return pd.DataFrame()

        # Ensure columns are plain strings (strip any residual MultiIndex levels)
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(-1)

        return close.astype(float)

    # ── Flat columns case (single ticker or older yfinance) ─────────────────
    if "Close" in cols:
        close = raw[["Close"]].copy()
        close.columns = [chunk[0]] if len(chunk) >= 1 else ["unknown"]
        return close.astype(float)

    if "Adj Close" in cols:
        close = raw[["Adj Close"]].copy()
        close.columns = [chunk[0]] if len(chunk) >= 1 else ["unknown"]
        return close.astype(float)

    logger.warning("No Close/Adj Close column found. Columns: %s", list(cols)[:10])
    return pd.DataFrame()


def _download_chunk(
    tickers: list[str],
    start: str,
    chunk_size: int = 100,
    sleep_between: float = 1.5,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers in batches,
    compatible with yfinance 0.2.x and 1.x.

    yfinance 1.x changed its internal download API; this function
    normalises the result via _extract_close() regardless of version.

    Parameters
    ----------
    tickers      : list of yfinance ticker strings (deduplicated before use)
    start        : start date string ("YYYY-MM-DD")
    chunk_size   : tickers per yfinance.download() call
    sleep_between: seconds to sleep between chunks

    Returns
    -------
    pd.DataFrame  columns = unique tickers, index = DatetimeIndex
    """
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tickers: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    all_frames: list[pd.DataFrame] = []
    total = len(unique_tickers)

    for i in range(0, total, chunk_size):
        chunk = unique_tickers[i : i + chunk_size]
        logger.info(
            "Downloading tickers %d–%d / %d …",
            i + 1, min(i + chunk_size, total), total,
        )
        try:
            raw = yf.download(
                chunk,
                start=start,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            close = _extract_close(raw, chunk)
            if close.empty:
                logger.warning("Empty close frame for chunk %d–%d — skipping", i, i + chunk_size)
                continue

            all_frames.append(close)

        except Exception as exc:
            logger.warning("Chunk %d–%d failed: %s — skipping", i, i + chunk_size, exc)

        if i + chunk_size < total:
            time.sleep(sleep_between)

    if not all_frames:
        raise RuntimeError("All download chunks failed. Check your internet connection.")

    # Concatenate and drop any duplicate columns that slipped through
    prices = pd.concat(all_frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    logger.info("Download complete: %d tickers × %d days", prices.shape[1], prices.shape[0])
    return prices


def load_prices(
    force_refresh: bool = False,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Return a DataFrame of adjusted close prices for all S&P 500 stocks and
    the ^GSPC index, indexed by trading date, starting from START_DATE.

    Results are cached to PRICES_CACHE (Parquet).  Set *force_refresh=True*
    to bypass the cache.

    Parameters
    ----------
    force_refresh     : if True, re-download even if cache exists
    progress_callback : optional callable(fraction: float, message: str)
                        for Streamlit progress bars

    Returns
    -------
    pd.DataFrame  shape ~ (2500, 503)
    """
    if not force_refresh and PRICES_CACHE.exists():
        logger.info("Loading prices from cache: %s", PRICES_CACHE)
        prices = pd.read_parquet(PRICES_CACHE)
        prices.index = pd.to_datetime(prices.index)
        return prices

    logger.info("Downloading prices from Yahoo Finance (start=%s) …", START_DATE)

    # ── Constituent tickers ──────────────────────────────────────────────────
    tickers = get_sp500_tickers()
    all_tickers = [INDEX_TICKER] + tickers

    if progress_callback:
        progress_callback(0.05, f"Fetching prices for {len(all_tickers)} series…")

    prices = _download_chunk(all_tickers, start=START_DATE, chunk_size=60, sleep_between=1.2)

    # ── Post-processing ──────────────────────────────────────────────────────
    # Forward-fill short gaps (up to 5 days) — e.g. exchange holidays
    prices = prices.ffill(limit=5)

    # Drop columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")

    # Ensure index is a proper DatetimeIndex with business-day frequency
    prices = prices[prices.index.dayofweek < 5]  # keep Mon-Fri only

    if progress_callback:
        progress_callback(0.6, "Caching prices to disk…")

    prices.to_parquet(PRICES_CACHE)
    logger.info("Prices saved to %s  shape=%s", PRICES_CACHE, prices.shape)

    if progress_callback:
        progress_callback(1.0, "Done.")

    return prices


# ── Sector and market-cap metadata ────────────────────────────────────────────

@retry(max_attempts=3, wait=2.0, backoff=2.0)
def _fetch_ticker_info(ticker: str) -> dict:
    """Fetch yfinance .info dict for a single ticker (with retry)."""
    info = yf.Ticker(ticker).info
    return info


def load_sector_info(
    tickers: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch sector and market-cap metadata from Yahoo Finance for all tickers.

    Columns returned:
        ticker, sector, industry, market_cap, name

    Results are cached to SECTOR_CACHE.  Set *force_refresh=True* to bypass.

    Notes
    -----
    - Fetches one ticker at a time to stay within rate limits.
    - Missing / empty responses default sector to "Unknown".
    - Market cap is in USD; NaN if unavailable.
    """
    if not force_refresh and SECTOR_CACHE.exists():
        logger.info("Loading sector info from cache: %s", SECTOR_CACHE)
        df = pd.read_parquet(SECTOR_CACHE)
        return df

    logger.info("Fetching sector metadata for %d tickers…", len(tickers))
    records = []

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            logger.info("  sector info: %d / %d", i, len(tickers))
        try:
            info = _fetch_ticker_info(ticker)
            records.append(
                {
                    "ticker":     ticker,
                    "sector":     info.get("sector", "Unknown") or "Unknown",
                    "industry":   info.get("industry", "Unknown") or "Unknown",
                    "market_cap": info.get("marketCap", np.nan),
                    "name":       info.get("longName", ticker) or ticker,
                }
            )
            time.sleep(0.3)  # gentle rate-limiting
        except Exception as exc:
            logger.warning("Could not fetch info for %s: %s", ticker, exc)
            records.append(
                {
                    "ticker":     ticker,
                    "sector":     "Unknown",
                    "industry":   "Unknown",
                    "market_cap": np.nan,
                    "name":       ticker,
                }
            )

    df = pd.DataFrame(records).set_index("ticker")
    df.to_parquet(SECTOR_CACHE)
    logger.info("Sector info saved to %s", SECTOR_CACHE)
    return df


# ── 2025 cumulative return table ──────────────────────────────────────────────

def compute_ytd_returns(prices: pd.DataFrame, year: int = 2025) -> pd.Series:
    """
    Compute cumulative price return (%) for *year* for each ticker.

    Uses simple price return (not log) for interpretability in the ranking table.
    """
    yr_prices = prices[prices.index.year == year]
    if yr_prices.empty:
        return pd.Series(dtype=float)
    first = yr_prices.iloc[0].replace(0, np.nan)
    last  = yr_prices.iloc[-1].replace(0, np.nan)
    return ((last / first) - 1.0) * 100.0


def compute_ytd_sharpe(returns: pd.DataFrame, year: int = 2025) -> pd.Series:
    """
    Annualized Sharpe ratio (assuming zero risk-free rate) for *year*.
    Returns are percentage log returns.
    """
    yr_ret = returns[returns.index.year == year]
    if yr_ret.empty or len(yr_ret) < 20:
        return pd.Series(dtype=float)
    mu   = yr_ret.mean()
    sig  = yr_ret.std()
    return (mu / sig) * np.sqrt(252)
