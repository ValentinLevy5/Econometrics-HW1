"""
download_data.py
----------------
Standalone script to pre-download all S&P 500 prices and sector metadata
from Yahoo Finance. Run this once before launching the Streamlit app:

    python download_data.py

This populates data/ and cache/ directories so the app starts instantly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_sp500_tickers, load_prices, load_sector_info
from src.preprocessing import compute_returns, clean_returns, save_returns
from src.utils import logger


def main():
    logger.info("=" * 60)
    logger.info("Financial Econometrics — Data Download Script")
    logger.info("=" * 60)

    # 1. Get tickers
    logger.info("Step 1/4: Fetching S&P 500 constituent list…")
    tickers = get_sp500_tickers()
    logger.info("  → %d tickers retrieved", len(tickers))

    # 2. Download prices
    logger.info("Step 2/4: Downloading adjusted close prices from Yahoo Finance…")
    logger.info("  This may take 5–15 minutes for ~500 tickers.")
    prices = load_prices(force_refresh=False)
    logger.info("  → Prices shape: %s", prices.shape)

    # 3. Compute and cache returns
    logger.info("Step 3/4: Computing percentage log returns…")
    returns = compute_returns(prices)
    returns = clean_returns(returns)
    save_returns(returns)
    logger.info("  → Returns shape: %s", returns.shape)

    # 4. Fetch sector metadata
    logger.info("Step 4/4: Fetching sector and market-cap metadata…")
    logger.info("  This fetches one ticker at a time and may take ~5 minutes.")
    stock_tickers = [c for c in returns.columns if c != "^GSPC"]
    sector_df = load_sector_info(stock_tickers, force_refresh=False)
    logger.info("  → Sector info shape: %s", sector_df.shape)

    logger.info("=" * 60)
    logger.info("Download complete. Launch the app with:")
    logger.info("  streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
