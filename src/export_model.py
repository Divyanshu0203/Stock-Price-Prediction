"""Train an LSTM on one stock and export it for the Vercel web app (web/models/<name>.npz).

    python src/export_model.py --source AMZN
    python src/export_model.py --source RELIANCE.NS --start 2012-01-01
    python src/export_model.py --source data/AMAZON_Stock_Train_Price.csv --symbol AMZN

--source is a Yahoo Finance ticker, or a CSV (then give --symbol: the ticker the website
should download live prices for). Run it once per stock you want on the website.

Before saving, two safety checks run and the export is refused if either fails:
  1. the website's NumPy features match the training features (src/features.py);
  2. the website's NumPy LSTM matches the trained Keras model's predictions.
The saved file also stores the model's HELD-OUT test results (the last 20% of the data,
never used for training), which the website shows next to every forecast.
"""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
COLUMNS = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}


def load_serving_lib():
    spec = importlib.util.spec_from_file_location("serving_lib", WEB / "api" / "_lib.py")
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


def extract_weights(model, d):
    w = model.get_weights()
    if len(w) != 10:
        raise SystemExit(f"Unexpected model structure ({len(w)} weight arrays, expected 10).")
    names = ["w0", "u0", "b0", "w1", "u1", "b1", "d0w", "d0b", "d1w", "d1b"]
    out = {n: np.asarray(a, dtype=np.float64) for n, a in zip(names, w)}
    out["f_mean"] = d["feature_scaler"].mean_.astype(np.float64)
    out["f_scale"] = d["feature_scaler"].scale_.astype(np.float64)
    out["r_mean"] = d["return_scaler"].mean_.astype(np.float64)
    out["r_scale"] = d["return_scaler"].scale_.astype(np.float64)
    return out


def check_features(lib, df, names):
    from features import build_features
    arrays = {k: df[c].to_numpy(float) for k, c in COLUMNS.items() if c in df.columns}
    served = lib.compute_features(arrays, names)
    trained = build_features(df)
    for n in names:
        a, b = served[n], trained[n].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if (np.isfinite(b) & ~np.isfinite(a)).any():
            raise SystemExit(f"Feature check failed: '{n}' has gaps in the website code but not in training.")
        diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        if diff > 1e-6:
            raise SystemExit(f"Feature check failed: '{n}' differs from training by {diff:.2e}. Nothing was saved.")
    print(f"Check 1 passed: website features match training features for all {len(names)} indicators.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Yahoo ticker (e.g. AMZN, RELIANCE.NS) or a CSV path")
    p.add_argument("--symbol", help="live-data ticker for the website (required when --source is a CSV)")
    p.add_argument("--name", help="name shown on the website (default: the symbol)")
    p.add_argument("--start", default="2010-01-01", help="first date to download for a ticker")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    is_csv = args.source.lower().endswith(".csv")
    symbol = (args.symbol or ("" if is_csv else args.source)).strip().upper()
    if not symbol:
        raise SystemExit("--symbol is required when --source is a CSV file.")
    name = (args.name or symbol).strip().upper()

    import keras
    from data_loader import load_data
    from models import build_regressor
    from train import direction_accuracy, prepare_data, rmse

    df = load_data(args.source, start=args.start)
    d = prepare_data(df, window=args.window)
    names = d["feature_names"]
    print(f"{name}: {d['n']} usable days, {len(d['X_train'])} train / {len(d['X_test'])} held-out test samples")

    lib = load_serving_lib()
    check_features(lib, df, names)

    keras.utils.set_random_seed(args.seed)
    model = build_regressor(args.window, len(names))
    y = d["return_scaler"].transform(d["ret_train"].reshape(-1, 1))
    model.fit(d["X_train"], y, epochs=args.epochs, batch_size=32, validation_split=0.1, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    weights = extract_weights(model, d)

    probe = d["X_test"][:100]
    ref = model.predict(probe, verbose=0).ravel()
    got = lib.forward(weights, probe)
    gap = float(np.abs(ref - got).max())
    if gap > 1e-4:
        raise SystemExit(f"Model check failed: NumPy and Keras predictions differ by {gap:.2e}. Nothing was saved.")
    print(f"Check 2 passed: website LSTM matches Keras (largest difference {gap:.1e}).")

    pred_ret = d["return_scaler"].inverse_transform(model.predict(d["X_test"], verbose=0)).ravel()
    pred_price = d["base_test"] * np.exp(pred_ret)
    lstm_rmse, naive_rmse = rmse(d["actual_test"], pred_price), rmse(d["actual_test"], d["base_test"])
    metrics = {
        "n_test_days": int(len(pred_ret)),
        "test_start": str(d["dates_test"][0].date()),
        "test_end": str(d["dates_test"][-1].date()),
        "lstm_rmse": round(lstm_rmse, 6),
        "naive_rmse": round(naive_rmse, 6),
        "lstm_vs_naive_pct": round(100 * (naive_rmse - lstm_rmse) / naive_rmse, 3),
        "direction_acc": round(direction_accuracy(pred_ret, d["ret_test"]), 4),
        "share_up_days": round(float(np.mean(d["ret_test"] > 0)), 4),
    }
    meta = {
        "name": name, "symbol": symbol, "source": args.source, "window": args.window,
        "feature_names": names, "n_train": int(len(d["X_train"])),
        "trained_through": str(df.index[df.index < d["dates_test"][0]][-1].date()),
        "data_through": str(df.index[-1].date()),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }

    out = WEB / "models" / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **weights)
    print(f"\nSaved {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"Held-out results ({metrics['n_test_days']} days): model RMSE {lstm_rmse:.3f} vs naive {naive_rmse:.3f} "
          f"({metrics['lstm_vs_naive_pct']:+.2f}%), direction {metrics['direction_acc']:.1%} "
          f"vs {metrics['share_up_days']:.1%} up days")
    print("Next: python web/dev_server.py")


if __name__ == "__main__":
    main()