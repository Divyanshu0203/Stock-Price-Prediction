"""Stock prediction pipeline: LSTM regressor + LSTM up/down classifier,
compared with naive and ARIMA baselines, plus a simple backtest.

Run from the project root (venv active):
    python src/train.py                         # Amazon CSV
    python src/train.py --source data/MRF_Stock_Price_Train.csv
    python src/train.py --source RELIANCE.NS    # download from Yahoo Finance
    python src/train.py --all                   # every CSV in data/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from backtest import backtest
from data_loader import load_data
from features import build_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


# --------------------------------------------------------------------------
# Data preparation (pure numpy/pandas: no deep-learning library needed)
# --------------------------------------------------------------------------
def prepare_data(df, window=60, train_frac=0.8):
    """Build chronological train/test windows.

    Sample i uses the feature rows [i-window+1 .. i] (days up to and including i)
    and predicts day i+1. Train samples have target days < split, test samples
    have target days >= split, so nothing from the test period is ever trained on.
    Scalers are fitted on days < split only.
    """
    feats = build_features(df)
    data = pd.concat([df["Close"], feats], axis=1).dropna()
    n = len(data)
    split = int(n * train_frac)
    if split < window + 50:
        raise ValueError(f"Only {n} usable rows: not enough data for window={window}.")

    close = data["Close"].to_numpy(float)
    ret = data["ret"].to_numpy(float)
    feature_names = [c for c in data.columns if c != "Close"]
    F = data[feature_names].to_numpy(float)

    feature_scaler = StandardScaler().fit(F[:split])
    return_scaler = StandardScaler().fit(ret[:split].reshape(-1, 1))
    Fs = feature_scaler.transform(F)

    def windows(i_lo, i_hi):
        idx = np.arange(i_lo, i_hi + 1)
        return idx, np.stack([Fs[i - window + 1:i + 1] for i in idx])

    tr_idx, X_train = windows(window - 1, split - 2)   # target days window..split-1
    te_idx, X_test = windows(split - 1, n - 2)         # target days split..n-1

    return {
        "n": n,
        "split": split,
        "feature_names": feature_names,
        "return_scaler": return_scaler,
        "X_train": X_train,
        "X_test": X_test,
        "ret_train": ret[tr_idx + 1],
        "ret_test": ret[te_idx + 1],
        "base_test": close[te_idx],          # yesterday's close for each test day
        "actual_test": close[te_idx + 1],    # the close we are trying to predict
        "ret_all": ret,
        "dates_test": data.index[te_idx + 1],
    }


# --------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def price_metrics(actual, pred):
    return {"rmse": rmse(actual, pred), "mae": float(mean_absolute_error(actual, pred))}


def direction_accuracy(pred_ret, actual_ret):
    return float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))


# --------------------------------------------------------------------------
# One full experiment on one dataset
# --------------------------------------------------------------------------
def run_experiment(name, df, args):
    import keras
    from baselines import arima_return_forecast
    from models import build_classifier, build_regressor

    d = prepare_data(df, window=args.window)
    n_test = len(d["ret_test"])
    n_feat = d["X_train"].shape[2]
    print(f"\n=== {name} ===")
    print(f"usable days: {d['n']}  train samples: {len(d['X_train'])}  test samples: {n_test}")
    print(f"features ({n_feat}): {', '.join(d['feature_names'])}")
    if n_test < 250:
        print("WARNING: small test set (<250 days). Treat these results as very noisy.")

    keras.utils.set_random_seed(args.seed)
    early = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    fit_kwargs = dict(epochs=args.epochs, batch_size=32, validation_split=0.1,
                      verbose=2, callbacks=[early])

    # ---- LSTM regressor: next-day return -> price ----
    reg = build_regressor(args.window, n_feat)
    y_reg = d["return_scaler"].transform(d["ret_train"].reshape(-1, 1))
    reg.fit(d["X_train"], y_reg, **fit_kwargs)
    pred_ret = d["return_scaler"].inverse_transform(reg.predict(d["X_test"], verbose=0)).ravel()

    # ---- LSTM classifier: next day up or down ----
    y_tr = (d["ret_train"] > 0).astype(int)
    y_te = (d["ret_test"] > 0).astype(int)
    clf = build_classifier(args.window, n_feat)
    clf.fit(d["X_train"], y_tr, **fit_kwargs)
    prob_up = clf.predict(d["X_test"], verbose=0).ravel()
    cls_pred = (prob_up > 0.5).astype(int)

    # ---- Baselines ----
    arima_ret = arima_return_forecast(d["ret_all"], d["split"])
    base, actual = d["base_test"], d["actual_test"]
    price = {
        "LSTM": base * np.exp(pred_ret),
        "ARIMA": base * np.exp(arima_ret),
        "Naive": base,
    }

    # ---- Evaluation ----
    res = {"dataset": name, "n_test_days": n_test}
    for k, v in price.items():
        res[f"{k.lower()}_price"] = price_metrics(actual, v)
    res["lstm_vs_arima_rmse_improvement_pct"] = float(
        100 * (res["arima_price"]["rmse"] - res["lstm_price"]["rmse"]) / res["arima_price"]["rmse"]
    )
    res["lstm_vs_naive_rmse_improvement_pct"] = float(
        100 * (res["naive_price"]["rmse"] - res["lstm_price"]["rmse"]) / res["naive_price"]["rmse"]
    )
    res["regressor_direction_acc"] = direction_accuracy(pred_ret, d["ret_test"])
    res["arima_direction_acc"] = direction_accuracy(arima_ret, d["ret_test"])

    res["classifier_accuracy"] = float(np.mean(cls_pred == y_te))
    # Strict baseline: the better of "always guess up" and "always guess down" on the test set.
    res["majority_class_baseline_accuracy"] = float(max(y_te.mean(), 1 - y_te.mean()))
    res["classifier_share_predicted_up"] = float(cls_pred.mean())
    res["share_up_days_in_test"] = float(y_te.mean())

    # ---- Backtest (long-only, with trading costs) ----
    signals = {
        "LSTM classifier": cls_pred,
        "LSTM regressor": (pred_ret > 0).astype(int),
        "ARIMA": (arima_ret > 0).astype(int),
        "Buy & hold": np.ones(n_test),
    }
    bt = {k: backtest(s, d["ret_test"], cost=args.cost) for k, s in signals.items()}
    res["backtest"] = {k: {m: v for m, v in r.items() if m != "equity"} for k, r in bt.items()}

    print_summary(res)
    make_plots(name, d, price, bt, args)
    return res


def print_summary(r):
    print(f"\n--- Results: {r['dataset']} ({r['n_test_days']} test days) ---")
    print(f"{'Price model':<8}{'RMSE':>10}{'MAE':>10}")
    for k in ("lstm", "arima", "naive"):
        m = r[f"{k}_price"]
        print(f"{k.upper():<8}{m['rmse']:>10.3f}{m['mae']:>10.3f}")
    print(f"LSTM vs ARIMA RMSE improvement: {r['lstm_vs_arima_rmse_improvement_pct']:+.2f}%")
    print(f"LSTM vs Naive RMSE improvement: {r['lstm_vs_naive_rmse_improvement_pct']:+.2f}%")
    print(f"Direction accuracy  LSTM regressor: {r['regressor_direction_acc']:.1%}   ARIMA: {r['arima_direction_acc']:.1%}")
    print(f"Up/down classifier accuracy: {r['classifier_accuracy']:.1%}   "
          f"(best constant-guess baseline: {r['majority_class_baseline_accuracy']:.1%}; "
          f"share of up days: {r['share_up_days_in_test']:.1%}; "
          f"classifier predicted up on {r['classifier_share_predicted_up']:.1%} of days)")
    print(f"\n{'Strategy':<18}{'Total ret':>11}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Trades':>8}")
    for k, m in r["backtest"].items():
        print(f"{k:<18}{m['total_return']:>10.1%}{m['cagr']:>9.1%}{m['sharpe']:>8.2f}"
              f"{m['max_drawdown']:>9.1%}{m['n_trades']:>8d}")


def make_plots(name, d, price, bt, args):
    OUT_DIR.mkdir(exist_ok=True)
    k = min(100, len(d["actual_test"]))

    plt.figure(figsize=(12, 6))
    plt.plot(d["actual_test"][-k:], color="black", label="Actual price")
    plt.plot(price["LSTM"][-k:], color="red", label="LSTM")
    plt.plot(price["ARIMA"][-k:], color="tab:blue", linestyle=":", label="ARIMA")
    plt.plot(price["Naive"][-k:], color="gray", linestyle="--", label="Naive (yesterday)")
    plt.title(f"{name}: next-day price prediction (last {k} test days)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(OUT_DIR / f"{name}_prices.png", dpi=120, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    for label, r in bt.items():
        plt.plot(r["equity"], label=label, linewidth=2 if label == "Buy & hold" else 1.2)
    plt.title(f"{name}: backtest, growth of 1 unit (cost {args.cost:.2%} per trade)")
    plt.xlabel("Test days")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.savefig(OUT_DIR / f"{name}_backtest.png", dpi=120, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close()


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(DATA_DIR / "AMAZON_Stock_Train_Price.csv"),
                   help="path to a CSV, or a Yahoo Finance ticker such as RELIANCE.NS or AAPL")
    p.add_argument("--all", action="store_true", help="run every *_Stock_Price_Train.csv in data/")
    p.add_argument("--start", default="2010-01-01", help="start date when downloading a ticker")
    p.add_argument("--window", type=int, default=60, help="days of history per sample")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--cost", type=float, default=0.001, help="trading cost per position change")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args = p.parse_args()

    if args.all:
        sources = sorted(DATA_DIR.glob("*_Stock_Price_Train.csv"))
        if not sources:
            raise SystemExit(f"No CSV files found in {DATA_DIR}")
    else:
        sources = [args.source]

    results = []
    for s in sources:
        name = Path(str(s)).stem.replace("_Stock_Price_Train", "").replace("_Stock_Train_Price", "") \
            if str(s).lower().endswith(".csv") else str(s)
        df = load_data(str(s), start=args.start)
        results.append(run_experiment(name, df, args))

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved plots and results.json to {OUT_DIR}")


if __name__ == "__main__":
    main()