# Deploying the stock forecast demo on Vercel

Vercel can't run TensorFlow, so the project is split in two:

- **Training (your PC):** `src/export_model.py` trains the LSTM and saves its weights to a tiny file in `web/models/`.
- **Website (Vercel):** a static page plus one Python function that downloads recent prices, computes the indicators and runs the saved LSTM using NumPy only.

## Folder layout

```
Stock-Prediction-System/
├── src/
│   ├── export_model.py      NEW   train + export a model for the website
│   ├── train.py             CHANGED (one line added)
│   └── ...                  (the rest is unchanged)
└── web/                     NEW   this whole folder is what Vercel deploys
    ├── api/
    │   ├── predict.py       the API endpoint
    │   └── _lib.py          features, LSTM and data code (NumPy only)
    ├── public/
    │   └── index.html       the web page
    ├── models/              exported models (AMZN.npz, ...) are created here
    ├── dev_server.py        run the site locally
    ├── vercel.json
    └── requirements.txt     numpy only
```

## 1. Train and export models (on your PC, with the venv active)

```
python src/export_model.py --source AMZN
python src/export_model.py --source MSFT
python src/export_model.py --source RELIANCE.NS --start 2012-01-01
```

Each run downloads prices with `yfinance`, trains, runs two safety checks, and writes `web/models/<NAME>.npz` (about 100 KB). The script **refuses to save** if the website's code doesn't reproduce the training features or the Keras model's predictions. It also prints the held-out results, which the website shows next to every forecast.

## 2. Try it locally

```
python web/dev_server.py
```

Open http://localhost:3000.

## 3. Deploy

1. Commit and push the `web/` folder to GitHub, **including `web/models/*.npz`**.
2. On vercel.com choose **Add New → Project** and import the repo.
3. Set **Root Directory** to `web`. This matters: the repo's top-level `requirements.txt` lists TensorFlow, which would make the deployment fail.
4. Set **Framework Preset** to *Other*, leave the build and output settings empty, and click **Deploy**.

Or with the CLI: `npm i -g vercel`, then `cd web`, then `vercel --prod`.

## Adding or refreshing a stock

Run `export_model.py` again (re-running overwrites that stock's file), commit the `.npz` and push. Vercel redeploys automatically. Retrain now and then, because the model only knows the period it was trained on.

## Things to know

- **Honest results are shown on purpose.** The page displays the model's held-out error next to the "tomorrow = today" baseline. For the Amazon data these are essentially equal, so the page says so.
- **Yahoo Finance is an unofficial source.** It sometimes rate-limits or blocks cloud servers, in which case the page shows an error message. Responses are cached for 10 minutes to reduce requests. For heavier use, switch to a paid market-data API.
- **Prices during market hours** can be intraday, so the "last close" may be a live price.
- **Not financial advice.** The page says so, and it should keep saying so.