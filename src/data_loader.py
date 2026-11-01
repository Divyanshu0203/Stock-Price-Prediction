"""Load OHLCV price data from a local CSV or from Yahoo Finance.

Every loader returns a DataFrame indexed by date (UTC, sorted, no duplicates)
with whichever of Open, High, Low, Close, Volume the source provides.
"""
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Different CSVs in this project name the volume column differently.
COLUMN_ALIASES = {"Total Trade Quantity": "Volume"}


def _clean(df):
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    keep = [c for c in OHLCV if c in df.columns]
    if "Close" not in keep:
        raise ValueError("Data has no 'Close' column.")
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return _clean(df)


def load_yahoo(ticker, start="2010-01-01", end=None):
    # Imported here so the CSV workflow still works if yfinance is missing.
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check the ticker symbol "
            "(Indian stocks need a suffix, e.g. RELIANCE.NS) and your internet connection."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    return _clean(df)


def load_data(source, start="2010-01-01"):
    """`source` is either a path ending in .csv or a Yahoo Finance ticker."""
    if str(source).lower().endswith(".csv"):
        return load_csv(source)
    return load_yahoo(source, start=start)