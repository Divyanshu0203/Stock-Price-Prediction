"""Baselines the LSTM has to beat.

  - naive: tomorrow's price = today's price
  - ARIMA: a classic statistical time-series model on daily log returns
"""
import warnings

import numpy as np


def arima_return_forecast(returns, split, order=(1, 0, 1)):
    """One-step-ahead ARIMA forecasts for returns[split:].

    The model is fitted ONLY on returns[:split]. Its parameters are then frozen
    and applied to the full series, so each forecast for day t uses data up to
    day t-1 and nothing from the future (no refitting, so it is fast).
    """
    from statsmodels.tsa.arima.model import ARIMA

    returns = np.asarray(returns, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(returns[:split], order=order, trend="c").fit()
        applied = fit.apply(returns)
    fitted = np.asarray(applied.fittedvalues, dtype=float)
    return fitted[split:]