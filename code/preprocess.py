#!/usr/bin/env python
# coding: utf-8

# In[5]:

display = print

from pathlib import Path
import pandas as pd
import numpy as np

cwd = Path.cwd()

if (cwd / "data" / "raw").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "data" / "raw").exists():
    PROJECT_ROOT = cwd.parent
else:
    raise FileNotFoundError(
        "Cannot find data/raw. Run this notebook from the project root "
        "or its code folder."
    )

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "preprocess"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Raw data:", RAW_DIR)
print("Processed data:", PROCESSED_DIR)
print("Raw CSV files:", len(list(RAW_DIR.glob("*.csv"))))


# In[ ]:


#Validate and clean the market data
def clean_market_file(file_path):
    df = pd.read_csv(file_path)
    original_rows = len(df)

    if "Date" not in df.columns:
        raise ValueError(f"{file_path.name}: missing Date column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"]).copy()

    # Remove duplicate dates, retaining the first observation.
    duplicate_dates = int(df["Date"].duplicated().sum())
    df = df.drop_duplicates(subset=["Date"], keep="first")
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert available market columns to numeric values.
    price_columns = [
        "Open", "High", "Low", "Close", "Adj Close", "Volume"
    ]
    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["Open", "High", "Low", "Close"]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{file_path.name}: missing columns {missing_columns}"
        )

    # Missing OHLC values are retained as missing and reported.
    missing_ohlc_rows = int(df[required].isna().any(axis=1).sum())

    # For indices, prices should normally be positive.
    nonpositive_prices = int(
        (df[required].le(0).any(axis=1) & df[required].notna().all(axis=1)).sum()
    )

    # Check OHLC relationships without automatically deleting observations.
    invalid_ohlc = int(
        (
            (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
            | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
        ).fillna(False).sum()
    )

    # Save cleaned dates in YYYY-MM-DD format.
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    output_path = PROCESSED_DIR / file_path.name
    df.to_csv(output_path, index=False)

    report = {
        "file": file_path.name,
        "original_rows": original_rows,
        "cleaned_rows": len(df),
        "invalid_dates_removed": invalid_dates,
        "duplicate_dates_removed": duplicate_dates,
        "rows_with_missing_OHLC": missing_ohlc_rows,
        "nonpositive_price_rows": nonpositive_prices,
        "invalid_OHLC_rows": invalid_ohlc,
        "missing_values_total": int(df.isna().sum().sum()),
        "start_date": df["Date"].min() if len(df) else None,
        "end_date": df["Date"].max() if len(df) else None,
        "output_file": str(output_path),
    }

    return report


reports = []

for file_path in sorted(RAW_DIR.glob("*.csv")):
    try:
        reports.append(clean_market_file(file_path))
        print("Processed:", file_path.name)
    except Exception as exc:
        print("ERROR:", file_path.name, "-", exc)

validation_report = pd.DataFrame(reports)

validation_report.to_csv(
    PROCESSED_DIR / "validation_report.csv",
    index=False
)

display(validation_report)


# In[7]:


#The CAC 40 row shows one observation with missing OHLC values and two missing cells. That is not necessarily a problem, but we should inspect it rather than inventing values.

# Inspect the complete validation report
report_path = PROCESSED_DIR / "validation_report.csv"
report = pd.read_csv(report_path)

print("Report rows:", len(report))
display(report.to_string(index=False))

# Check whether every market has a report row
print("\nMissing market reports:")
print(
    sorted(
        set(p.name for p in RAW_DIR.glob("*.csv"))
        - set(report["file"].dropna())
    )
)

# Inspect missing values in the CAC 40 dataset
cac40 = pd.read_csv(PROCESSED_DIR / "cac40.csv")

print("\nCAC 40 rows with missing OHLC values:")
display(
    cac40.loc[
        cac40[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    ]
)


# In[8]:


# Remove rows with missing or nonpositive OHLC prices
# from processed copies only. Raw data remains unchanged.

price_cols = ["Open", "High", "Low", "Close"]

for file in sorted(RAW_DIR.glob("*.csv")):
    processed_file = PROCESSED_DIR / file.name

    if not processed_file.exists():
        continue

    df = pd.read_csv(processed_file)
    before = len(df)

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_prices = (
        df[price_cols].notna().all(axis=1)
        & df[price_cols].gt(0).all(axis=1)
    )

    df = df.loc[valid_prices].copy()
    df.to_csv(processed_file, index=False)

    print(f"{file.name}: removed {before - len(df)} incomplete/invalid rows")


# In[9]:


fx = pd.read_csv(PROCESSED_DIR / "usd_inr.csv")

# Check the OHLC relationships directly.
fx["Date"] = pd.to_datetime(fx["Date"])

bad_fx = fx[
    (fx["High"] < fx[["Open", "Close", "Low"]].max(axis=1))
    | (fx["Low"] > fx[["Open", "Close", "High"]].min(axis=1))
]

print("USD/INR flagged rows:", len(bad_fx))
display(bad_fx.head(20))


# In[10]:


#Set paths and market configuration
from pathlib import Path
import pandas as pd
import numpy as np

cwd = Path.cwd()

if (cwd / "data" / "preprocess").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "data" / "preprocess").exists():
    PROJECT_ROOT = cwd.parent
else:
    raise FileNotFoundError("Cannot locate project data directory.")

DATA_DIR = PROJECT_ROOT / "data" / "preprocess"
FEATURE_DIR = DATA_DIR / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

MARKETS = {
    "nifty50": "nifty50.csv",
    "sensex": "sensex.csv",
    "sp500": "sp500.csv",
    "nasdaq": "nasdaq.csv",
    "dow_jones": "dow_jones.csv",
    "nikkei225": "nikkei225.csv",
    "hang_seng": "hang_seng.csv",
    "shanghai": "shanghai.csv",
    "dax": "dax.csv",
    "ftse100": "ftse100.csv",
    "cac40": "cac40.csv",
    "india_vix": "india_vix.csv",
    "usd_inr": "usd_inr.csv",
}

print("Project:", PROJECT_ROOT)
print("Input:", DATA_DIR)
print("Output:", FEATURE_DIR)


# In[11]:


#Load and validate each market
#We will load the processed CSVs and calculate features separately for each market. Importantly, the global-market features will be shifted by one observation before they can be used for a NIFTY prediction. This conservative approach prevents a market's same-date closing data from leaking into the pre-market forecast.
def load_market(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values("Date")
    df = df.drop_duplicates("Date", keep="last")
    df = df.set_index("Date")

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reject invalid OHLC rows instead of fabricating values.
    valid = (
        df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
        & df[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
    )

    df = df.loc[valid].copy()

    # Do not treat zero-volume observations as invalid automatically:
    # index tickers commonly have no meaningful volume.
    return df


market_data = {
    name: load_market(filename)
    for name, filename in MARKETS.items()
}

for name, df in market_data.items():
    print(
        f"{name:12s} rows={len(df):4d} "
        f"from={df.index.min().date()} to={df.index.max().date()}"
    )


# In[12]:


def build_market_features(name, df):
    out = pd.DataFrame(index=df.index)

    close = df["Close"]
    open_price = df["Open"]

    # Daily return, ending at this market's session close.
    out[f"{name}_return_1d"] = close.pct_change(fill_method=None)
    out[f"{name}_log_return_1d"] = np.log(close).diff()

    # Momentum ending at this session close.
    for window in [3, 5, 10, 20]:
        out[f"{name}_momentum_{window}d"] = close.pct_change(
            periods=window, fill_method=None
        )

    # Rolling volatility of daily returns.
    daily_return = close.pct_change(fill_method=None)

    for window in [5, 10, 20]:
        out[f"{name}_volatility_{window}d"] = (
            daily_return.rolling(window, min_periods=window).std()
        )

    # Session range, normalized by close.
    out[f"{name}_range_pct"] = (
        (df["High"] - df["Low"]) / close
    )

    # Only create close-to-open return if it is useful for analysis.
    out[f"{name}_intraday_return"] = (
        close / open_price - 1
    )

    return out


features_by_market = {}

for name, df in market_data.items():
    features_by_market[name] = build_market_features(name, df)

print("Feature engineering completed for", len(features_by_market), "markets.")


# In[13]:


# NIFTY trading sessions define the rows of the dataset.
nifty = market_data["nifty50"]

dataset = pd.DataFrame(index=nifty.index)
dataset.index.name = "Date"

# Indian domestic indicators also use only their prior session.
domestic_markets = ["nifty50", "sensex", "india_vix", "usd_inr"]

# International markets whose same-date closing values are excluded.
global_markets = [
    "sp500", "nasdaq", "dow_jones", "nikkei225",
    "hang_seng", "shanghai", "dax", "ftse100", "cac40"
]

all_features = {
    name: features
    for name, features in features_by_market.items()
}

for name in domestic_markets + global_markets:
    features = all_features[name].copy()

    # Shift on the source market's own calendar before alignment.
    # This uses the previous available observation for that market.
    features = features.shift(1)

    # Exact-date join after shifting. NIFTY dates with no matching
    # source-market session remain missing; no future filling occurs.
    features = features.reindex(dataset.index)

    dataset = dataset.join(features, how="left")

print("Aligned dataset shape:", dataset.shape)
display(dataset.head())


# In[14]:


target = pd.Series(index=nifty.index, dtype="float64", name="target_up")

valid_target = (
    nifty["Open"].notna()
    & nifty["Close"].notna()
    & (nifty["Open"] > 0)
)

target.loc[valid_target] = (
    nifty.loc[valid_target, "Close"]
    > nifty.loc[valid_target, "Open"]
).astype(int)

dataset["target_up"] = target

print("Target distribution:")
print(dataset["target_up"].value_counts(dropna=False))


# In[15]:


# Remove rows where the target is unavailable.
dataset = dataset.dropna(subset=["target_up"]).copy()

# Save the complete feature dataset.
output_path = FEATURE_DIR / "nifty50_premarket_features.csv"
dataset.to_csv(output_path, index_label="Date")

# Save metadata about the dataset.
manifest = pd.DataFrame({
    "feature": dataset.columns,
    "dtype": [str(dataset[col].dtype) for col in dataset.columns],
    "missing_values": [int(dataset[col].isna().sum()) for col in dataset.columns],
})

manifest.to_csv(FEATURE_DIR / "feature_manifest.csv", index=False)

print("Saved:", output_path)
print("Rows:", len(dataset))
print("Feature columns:", len(dataset.columns) - 1)
print("Missing feature cells:", int(dataset.drop(columns="target_up").isna().sum().sum()))
display(manifest)


# In[16]:


# Example: verify that the SP500 return feature is shifted
# before it is aligned to the NIFTY session dates.
name = "sp500"
feature_name = f"{name}_return_1d"

expected = (
    features_by_market[name][feature_name]
    .shift(1)
    .reindex(dataset.index)
)

actual = dataset[feature_name]

comparison = pd.DataFrame({
    "actual": actual,
    "expected": expected,
})

valid = comparison["actual"].notna() & comparison["expected"].notna()

assert np.allclose(
    comparison.loc[valid, "actual"],
    comparison.loc[valid, "expected"],
    rtol=1e-10,
    atol=1e-12,
), "Leakage/alignment test failed."

print("PASS: SP500 features use a prior observation.")
print("Feature dataset:", output_path)


# In[17]:


from pathlib import Path
import pandas as pd

feature_dir = PROJECT_ROOT / "data" / "preprocess" / "features"
dataset_path = feature_dir / "nifty50_premarket_features.csv"
manifest_path = feature_dir / "feature_manifest.csv"

print("Dataset exists:", dataset_path.exists())
print("Manifest exists:", manifest_path.exists())

if dataset_path.exists():
    df = pd.read_csv(dataset_path)

    print("Dataset shape:", df.shape)
    print("Date range:", df["Date"].min(), "to", df["Date"].max())
    print("\nTarget distribution:")
    print(df["target_up"].value_counts(dropna=False))
    print("\nMissing values in top 15 columns:")
    print(df.isna().sum().sort_values(ascending=False).head(15))

if manifest_path.exists():
    print("\nManifest rows:", len(pd.read_csv(manifest_path)))


# In[18]:


import numpy as np
import pandas as pd

feature_path = (
    PROJECT_ROOT
    / "data"
    / "preprocess"
    / "features"
    / "nifty50_premarket_features.csv"
)

df = pd.read_csv(feature_path, parse_dates=["Date"])

target_col = "target_up"
feature_cols = [c for c in df.columns if c not in ["Date", target_col]]

missing_report = pd.DataFrame({
    "feature": feature_cols,
    "missing_count": [df[c].isna().sum() for c in feature_cols],
    "missing_pct": [df[c].isna().mean() * 100 for c in feature_cols],
    "infinite_count": [
        np.isinf(pd.to_numeric(df[c], errors="coerce")).sum()
        for c in feature_cols
    ],
})

missing_report = missing_report.sort_values(
    "missing_pct", ascending=False
)

display(missing_report.head(20))

print("Total rows:", len(df))
print("Total feature columns:", len(feature_cols))
print("Duplicate dates:", df["Date"].duplicated().sum())
print("Missing targets:", df[target_col].isna().sum())
print("Infinite feature values:", missing_report["infinite_count"].sum())

missing_report.to_csv(
    PROJECT_ROOT / "data" / "preprocess" / "features"
    / "feature_quality_report.csv",
    index=False,
)


# In[ ]:




