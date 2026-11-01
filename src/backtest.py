"""Simple long-only backtest.

signal[k] = 1 means "hold the stock tomorrow", 0 means "stay in cash". The signal
for day k+1 is decided from information available at the close of day k.
Each change of position costs `cost` (0.001 = 0.1% per trade).
"""
import numpy as np


def backtest(signal, next_log_ret, cost=0.001, periods_per_year=252):
    signal = np.asarray(signal, dtype=float)
    simple_ret = np.expm1(np.asarray(next_log_ret, dtype=float))
    trades = np.abs(np.diff(signal, prepend=0.0))        # entering/leaving the market
    strat = signal * simple_ret - trades * cost
    equity = np.cumprod(1.0 + strat)

    years = len(strat) / periods_per_year
    std = strat.std(ddof=1)
    peak = np.maximum.accumulate(equity)
    return {
        "total_return": float(equity[-1] - 1.0),
        "cagr": float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan"),
        "sharpe": float(strat.mean() / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "n_trades": int(round(trades.sum())),
        "equity": equity,
    }