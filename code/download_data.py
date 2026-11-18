
from pathlib import Path
import time

import pandas as pd
import yfinance as yf


from pathlib import Path

# Your notebook is inside the code folder
PROJECT_ROOT = Path.cwd().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Raw data folder:", RAW_DIR)

# Yahoo Finance symbols
TICKERS = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW_JONES": "^DJI",
    "NIKKEI225": "^N225",
    "HANG_SENG": "^HSI",
    "SHANGHAI": "000001.SS",
    "DAX": "^GDAXI",
    "FTSE100": "^FTSE",
    "CAC40": "^FCHI",
    "INDIA_VIX": "^INDIAVIX",
    "USD_INR": "INR=X",
}


def download_ticker(name: str, ticker: str) -> bool:
    """Download approximately 10 years of daily market data."""

    print(f"\nDownloading {name} ({ticker})...")

    try:
        df = yf.download(
            ticker,
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=30,
        )

        if df.empty:
            print(f"FAILED: No data returned for {name}")
            return False

        # Handle MultiIndex columns returned by some yfinance versions.
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df = df.reset_index()

        # Convert dates to a consistent, date-only representation.
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        # Remove duplicate dates and sort chronologically.
        df = df.drop_duplicates(subset=["Date"])
        df = df.sort_values("Date")

        # Ensure essential market columns exist.
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)

        if missing:
            print(f"FAILED: {name} is missing columns: {sorted(missing)}")
            return False

        # Save one CSV per market.
        output_file = RAW_DIR / f"{name.lower()}.csv"
        df.to_csv(output_file, index=False)

        print(f"SUCCESS: {name}")
        print(f"Rows: {len(df):,}")
        print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"Saved to: {output_file}")

        return True

    except Exception as exc:
        print(f"FAILED: {name} — {exc}")
        return False


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    print("=" * 60)
    print("10-YEAR GLOBAL MARKET DATA DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {RAW_DIR}")

    for name, ticker in TICKERS.items():
        if download_ticker(name, ticker):
            successful.append(name)
        else:
            failed.append(name)

        # Be polite to the data provider.
        time.sleep(1)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(TICKERS)}")
    print(f"Failed: {len(failed)}/{len(TICKERS)}")

    if successful:
        print("\nDownloaded:", ", ".join(successful))

    if failed:
        print("\nFailed:", ", ".join(failed))
        print("Check the ticker, network connection, or data availability.")


if __name__ == "__main__":
    main()