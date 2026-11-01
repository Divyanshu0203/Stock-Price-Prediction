"""Technical-indicator features.

Every feature on day t uses only data up to and including day t.
Features are scale-free (returns, ratios, gaps) so they stay comparable
when the price level changes a lot, which is what broke the raw-price model.
Columns that the dataset does not provide (e.g. Google's CSV has only Close)
are skipped automatically.
"""
import numpy as np
import pandas as pd


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(loss == 0) & (gain > 0)] = 100.0
    out[(loss == 0) & (gain == 0)] = 50.0
    return out


def build_features(df):
    c = df["Close"]
    f = pd.DataFrame(index=df.index)

    f["ret"] = np.log(c).diff()                    # daily log return
    f["sma5_gap"] = c / c.rolling(5).mean() - 1    # distance from 5-day average
    f["sma20_gap"] = c / c.rolling(20).mean() - 1  # distance from 20-day average
    f["vol10"] = f["ret"].rolling(10).std()        # recent volatility
    f["rsi"] = rsi(c) / 100.0                      # momentum, scaled to 0..1

    if {"High", "Low"}.issubset(df.columns):
        f["hl_range"] = (df["High"] - df["Low"]) / c   # intraday range
    if "Open" in df.columns:
        f["oc_ret"] = np.log(c / df["Open"])           # open-to-close move
    if "Volume" in df.columns and (df["Volume"] > 0).mean() > 0.9:
        v = df["Volume"].replace(0, np.nan)
        f["vol_ratio"] = np.log(v / v.rolling(20).mean())  # volume vs 20-day average

    return f.replace([np.inf, -np.inf], np.nan)