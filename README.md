# Stock Price Prediction System

An end-to-end stock forecasting project: an LSTM trained on price history, evaluated honestly against baselines, and deployed as a small website.

The main finding is deliberately honest: **the LSTM does not beat the "tomorrow = today" baseline.** The project shows how to measure that properly, which matters more than any single accuracy number.

## What it does

- **Data:** OHLCV prices from CSV files in `data/` or downloaded from Yahoo Finance (`yfinance`).
- **Features (8 indicators):** daily log return, distance from the 5- and 20-day averages, 10-day volatility, RSI, intraday range, open-to-close move, volume vs its 20-day average. All are scale-free, so a stock moving from 300 to 1000 doesn't break the model.
- **Models:** an LSTM regressor (next-day return, converted to a price) and an LSTM classifier (next day up or down).
- **Baselines:** naive ("tomorrow = today") and ARIMA(1,0,1).
- **Validation:** chronological 80/20 split, scalers fitted on training data only, no shuffling across the split.
- **Backtest:** long-only strategies from each model's signal vs buy-and-hold, with 0.1% trading cost.
- **Website:** a static page plus a NumPy-only API on Vercel that fetches live prices and runs the trained LSTM.

## Project layout

```
Stock-Prediction-System/
├── data/                  CSV price files
├── src/
│   ├── data_loader.py     CSV and Yahoo Finance loading
│   ├── features.py        technical indicators
│   ├── models.py          LSTM regressor and classifier
│   ├── baselines.py       ARIMA baseline
│   ├── backtest.py        long-only backtest
│   ├── train.py           full evaluation: models vs baselines + backtest + plots
│   └── export_model.py    train a model and export it for the website
├── web/                   the website (deploy this folder to Vercel)
│   ├── api/predict.py     API endpoint
│   ├── api/_lib.py        features, LSTM and data code (NumPy only)
│   ├── public/index.html  the web page
│   ├── models/            exported models (*.npz)
│   ├── dev_server.py      run the site locally
│   └── vercel.json
├── outputs/               generated plots and results.json
├── requirements.txt
└── DEPLOY.md              step-by-step Vercel deployment
```

## Setup (Windows, VS Code)

Python 3.11 or 3.12 is recommended (TensorFlow lags behind the newest Python).

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 1. Evaluate the models

```
python src/train.py                                    # Amazon CSV
python src/train.py --source data/MRF_Stock_Price_Train.csv
python src/train.py --source RELIANCE.NS               # download from Yahoo Finance
python src/train.py --all                              # every CSV in data/
python src/train.py --no-show --epochs 20              # save plots only, faster
```

Plots and `results.json` are saved to `outputs/`.

## 2. Export a model for the website

```
python src/export_model.py --source AMZN
python src/export_model.py --source RELIANCE.NS --start 2012-01-01
```

This trains the LSTM and writes `web/models/<NAME>.npz` (about 100 KB). Before saving, it checks that the website's NumPy code reproduces the training features and the Keras model's predictions, and it refuses to save if either check fails. The held-out test results are stored in the file and shown on the website.

## 3. Run the website locally

```
python web/dev_server.py
```

Open http://localhost:3000. (Opening `index.html` directly or with a static server like Live Server will not work, because the page needs the `/api/predict` endpoint.)

## 4. Deploy on Vercel

See `DEPLOY.md`. In short: push `web/` (including `web/models/*.npz`) to GitHub, import the repo on Vercel, and set **Root Directory** to `web`.

## Results

Amazon CSV (2000-2017), chronological split, 892 held-out days:

| Model | RMSE | MAE |
|---|---|---|
| LSTM | 9.93 | 6.77 |
| ARIMA | 9.91 | 6.75 |
| Naive (yesterday) | 9.93 | 6.77 |

- Direction accuracy: LSTM regressor 52.8%, ARIMA 54.3% (54.3% of days were up days).
- The up/down classifier scored 46.9%, below the 54.3% of always guessing "up".
- Backtest (0.1% cost): buy-and-hold returned 192% (Sharpe 1.19); the LSTM regressor 160%; the classifier 8%.

Live Amazon model on the website (Yahoo data, trained through 2023-07-11, 846 held-out days): RMSE 4.24 vs 4.24 for the naive baseline, direction 50.8% vs 51.6% up days.

**Takeaways**
- The LSTM matches the naive baseline within noise, so there is no demonstrated edge.
- A raw-price LSTM looks accurate on a chart but is just lagging the real price; predicting returns and always comparing with the naive baseline exposes this.
- No strategy beat buy-and-hold in the backtest, and results from one stock and one period are not evidence of a tradable strategy.

## Limitations

- Daily price history alone carries very little predictable signal.
- Each stock gets its own model, trained once; retrain periodically.
- The website uses Yahoo Finance, an unofficial source that may rate-limit or block cloud servers, and prices may be intraday.
- Educational project. Not financial advice.