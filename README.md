*Educational project. Not financial advice.*

# Stock Prediction System

An LSTM-based stock forecasting pipeline that is evaluated honestly: every model is compared against a naive baseline and an ARIMA baseline, and predictions are turned into a backtest with trading costs.

## What it does

- **Data:** OHLCV prices from the CSVs in `data/` or downloaded from Yahoo Finance (`yfinance`).
- **Features:** daily log return, distance from 5/20-day moving averages, 10-day volatility, RSI, intraday range, open-to-close move, volume vs 20-day average. All are scale-free, so a stock moving from 300 to 1000 doesn't break the model.
- **Models:**
  - LSTM regressor predicting the next day's return (converted back to a price).
  - LSTM classifier predicting whether the next day closes up or down.
- **Baselines:** naive ("tomorrow = today") and ARIMA(1,0,1) on returns.
- **Validation:** chronological 80/20 split, scalers fitted on training data only, no shuffling across the split.
- **Backtest:** long-only strategy from each model's signal vs buy-and-hold, with a per-trade cost (default 0.1%).

## Setup (Windows, VS Code)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put the CSV files in a `data/` folder.

## Run

```
python src/train.py                                   # Amazon CSV
python src/train.py --source data/MRF_Stock_Price_Train.csv
python src/train.py --source RELIANCE.NS              # download from Yahoo Finance
python src/train.py --all                             # every CSV in data/
python src/train.py --no-show --epochs 20             # save plots only, faster
```

Plots and `results.json` are saved to `outputs/`.

## Reading the results

- Price-model RMSE/MAE are compared with the naive and ARIMA baselines. Beating them is hard, and matching them is a normal result.
- Classifier accuracy should be judged against the best constant-guess baseline (always up or always down), not against 50%.
- Backtest numbers cover a single stock and period and are not evidence of a tradable strategy.

## Project layout

```
data/            CSV price files
src/data_loader.py   CSV and Yahoo Finance loading
src/features.py      technical indicators
src/models.py        LSTM regressor and classifier
src/baselines.py     ARIMA baseline
src/backtest.py      long-only backtest
src/train.py         main script
outputs/         generated plots and results.json
```

*Educational project. Not financial advice.*