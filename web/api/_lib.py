"""Serving code for the stock forecast API.

Uses NumPy only (no TensorFlow, pandas or scikit-learn), so it fits easily inside a
Vercel serverless function. The training code lives in src/; src/export_model.py checks
that this file reproduces the training features and the Keras model before it saves a
model, so the two cannot silently drift apart.
"""
import json
import re
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SYMBOL_RE = re.compile(r"^[A-Za-z0-9.^=\-]{1,20}$")
YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
             "?range={rng}&interval=1d&includeAdjustedClose=true")
USER_AGENT = "Mozilla/5.0 (compatible; stock-forecast-demo/1.0)"
HISTORY_DAYS = 120
MIN_EXTRA_ROWS = 30   # warm-up rows needed by the 20-day indicators, plus a margin
CACHE_SECONDS = 600


class ApiError(Exception):
    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


# ---------------------------------------------------------------------------
# Market data (Yahoo Finance chart endpoint, standard library only)
# ---------------------------------------------------------------------------
def parse_chart(payload):
    """Turn a Yahoo chart JSON payload into float arrays, adjusted for splits and
    dividends the same way yfinance's auto_adjust=True does."""
    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise ApiError(404, "Yahoo Finance: " + str(chart["error"].get("description", "no data for this symbol")))
    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise ApiError(404, "Yahoo Finance returned no price data for this symbol.")

    t = np.array(result["timestamp"], dtype=np.int64)
    quote = result["indicators"]["quote"][0]

    def col(values):
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    arrays = {k: col(quote.get(k, [])) for k in ("open", "high", "low", "close", "volume")}
    if any(a.shape != t.shape for a in arrays.values()):
        raise ApiError(502, "Yahoo Finance returned inconsistent data.")

    adj = (result["indicators"].get("adjclose") or [None])[0]
    if adj and adj.get("adjclose"):
        adjclose = col(adj["adjclose"])
        ok = np.isfinite(adjclose) & np.isfinite(arrays["close"]) & (arrays["close"] != 0)
        factor = np.where(ok, adjclose / np.where(ok, arrays["close"], 1.0), 1.0)
        for k in ("open", "high", "low", "close"):
            arrays[k] = arrays[k] * factor

    keep = np.isfinite(arrays["close"])
    out = {k: v[keep] for k, v in arrays.items()}
    out["t"] = t[keep]
    return out


def fetch_yahoo(symbol, rng="2y", timeout=8):
    url = YAHOO_URL.format(symbol=urllib.parse.quote(symbol, safe=""), rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise ApiError(502, "Yahoo Finance is rate-limiting requests right now. Try again in a minute.")
        raise ApiError(502, f"Yahoo Finance returned HTTP {e.code}.")
    except Exception:
        raise ApiError(502, "Could not reach Yahoo Finance. Try again shortly.")
    return parse_chart(payload)


# ---------------------------------------------------------------------------
# Features (must match src/features.py exactly)
# ---------------------------------------------------------------------------
def _roll_mean(a, w):
    out = np.full(a.shape, np.nan)
    if a.size >= w:
        out[w - 1:] = sliding_window_view(a, w).mean(axis=1)
    return out


def _roll_std(a, w):
    out = np.full(a.shape, np.nan)
    if a.size >= w:
        out[w - 1:] = sliding_window_view(a, w).std(axis=1, ddof=1)
    return out


def _rsi(close, period=14):
    delta = np.full(close.shape, np.nan)
    delta[1:] = np.diff(close)
    gain = _roll_mean(np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0)), period)
    loss = _roll_mean(np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0.0)), period)
    rs = gain / np.where(loss == 0, np.nan, loss)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[(loss == 0) & (gain > 0)] = 100.0
    out[(loss == 0) & (gain == 0)] = 50.0
    return out


def compute_features(d, names=None):
    """d: dict of float arrays with keys close (required) and open, high, low, volume.
    Returns {feature name: array aligned with the input rows}. Only features whose
    inputs exist are returned; `names` limits the output."""
    with np.errstate(all="ignore"):
        c = d["close"]
        f = {}
        ret = np.full(c.shape, np.nan)
        ret[1:] = np.diff(np.log(c))
        f["ret"] = ret
        f["sma5_gap"] = c / _roll_mean(c, 5) - 1
        f["sma20_gap"] = c / _roll_mean(c, 20) - 1
        f["vol10"] = _roll_std(ret, 10)
        f["rsi"] = _rsi(c) / 100.0
        if "high" in d and "low" in d:
            f["hl_range"] = (d["high"] - d["low"]) / c
        if "open" in d:
            f["oc_ret"] = np.log(c / d["open"])
        if "volume" in d and np.mean(d["volume"] > 0) > 0.9:
            v = np.where(d["volume"] > 0, d["volume"], np.nan)
            f["vol_ratio"] = np.log(v / _roll_mean(v, 20))
    f = {k: np.where(np.isfinite(v), v, np.nan) for k, v in f.items()}
    if names is not None:
        missing = [n for n in names if n not in f]
        if missing:
            raise ApiError(502, "Market data is missing fields the model needs: " + ", ".join(missing))
        f = {n: f[n] for n in names}
    return f


# ---------------------------------------------------------------------------
# LSTM forward pass (Keras conventions: gate order i, f, c, o)
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _lstm(x, W, U, b, return_sequences):
    """x: (N, T, n_in) -> (N, T, units) or (N, units)."""
    n, steps, _ = x.shape
    units = U.shape[0]
    xw = x @ W + b
    h = np.zeros((n, units))
    c = np.zeros((n, units))
    outs = []
    for t in range(steps):
        z = xw[:, t, :] + h @ U
        i = _sigmoid(z[:, :units])
        fgate = _sigmoid(z[:, units:2 * units])
        g = np.tanh(z[:, 2 * units:3 * units])
        o = _sigmoid(z[:, 3 * units:])
        c = fgate * c + i * g
        h = o * np.tanh(c)
        if return_sequences:
            outs.append(h)
    return np.stack(outs, axis=1) if return_sequences else h


def forward(w, x):
    """The trained regressor (LSTM 64 -> LSTM 32 -> Dense 16 relu -> Dense 1), dropout is
    inactive at prediction time. x: (N, T, n_features) already scaled. Returns the model's
    scaled output, shape (N,)."""
    x = np.asarray(x, dtype=float)
    h = _lstm(x, w["w0"], w["u0"], w["b0"], True)
    h = _lstm(h, w["w1"], w["u1"], w["b1"], False)
    h = np.maximum(h @ w["d0w"] + w["d0b"], 0.0)
    return (h @ w["d1w"] + w["d1b"]).reshape(-1)


# ---------------------------------------------------------------------------
# Model files and the request handler
# ---------------------------------------------------------------------------
_CACHE = {}


def list_models():
    return sorted(p.stem for p in MODEL_DIR.glob("*.npz")) if MODEL_DIR.is_dir() else []


def load_model(name):
    m = _CACHE.get(name)
    if m is None:
        with np.load(MODEL_DIR / f"{name}.npz", allow_pickle=False) as z:
            weights = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        m = _CACHE[name] = {"w": weights, "meta": meta}
    return m


def make_window(rows, model):
    meta, w = model["meta"], model["w"]
    window = int(meta["window"])
    if rows["close"].size < window + MIN_EXTRA_ROWS:
        raise ApiError(422, "Not enough recent price history to make a forecast.")
    feats = compute_features(rows, meta["feature_names"])
    mat = np.column_stack([feats[n] for n in meta["feature_names"]])[-window:]
    if not np.isfinite(mat).all():
        raise ApiError(422, "Recent price data has gaps, so a forecast can't be made right now.")
    return (mat - w["f_mean"]) / w["f_scale"]


def _day(ts):
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def forecast(name, rows):
    model = load_model(name)
    meta, w = model["meta"], model["w"]
    window = make_window(rows, model)
    scaled_out = float(forward(w, window[None])[0])
    pred_ret = scaled_out * float(w["r_scale"][0]) + float(w["r_mean"][0])
    last = float(rows["close"][-1])
    pred_close = last * float(np.exp(pred_ret))
    if not np.isfinite(pred_close):
        raise ApiError(500, "The model produced an invalid number.")
    keep = slice(-HISTORY_DAYS, None)
    return {
        "ticker": name,
        "symbol": meta["symbol"],
        "as_of": _day(rows["t"][-1]),
        "last_close": round(last, 4),
        "predicted_return_pct": round(pred_ret * 100, 4),
        "predicted_close": round(pred_close, 4),
        "naive_close": round(last, 4),
        "history": [{"d": _day(t), "c": round(float(c), 4)}
                    for t, c in zip(rows["t"][keep], rows["close"][keep])],
        "model": {k: meta.get(k) for k in
                  ("trained_through", "data_through", "window", "feature_names", "n_train", "metrics")},
        "note": "Prices are split/dividend-adjusted and may be delayed or intraday. "
                "Educational demo, not financial advice.",
    }


def _list_response():
    items = []
    for n in list_models():
        meta = load_model(n)["meta"]
        items.append({"ticker": n, "symbol": meta["symbol"], "trained_through": meta.get("trained_through")})
    return {"models": items}


def _handle(raw_path, fetcher):
    query = urllib.parse.parse_qs(urllib.parse.urlparse(raw_path).query)
    ticker = (query.get("ticker", [""])[0]).strip().upper()
    available = list_models()
    if not ticker:
        return 200, _list_response(), 300
    if not SYMBOL_RE.match(ticker):
        raise ApiError(400, "That doesn't look like a valid ticker.", available=available)
    if ticker not in available:
        raise ApiError(404, f"No trained model for '{ticker}'. Available: {', '.join(available) or 'none'}.",
                       available=available)
    symbol = load_model(ticker)["meta"]["symbol"]   # fetch the symbol the model was trained on
    rows = (fetcher or fetch_yahoo)(symbol)
    return 200, forecast(ticker, rows), CACHE_SECONDS


def handle(raw_path, fetcher=None):
    """Returns (http status, JSON-able dict, seconds the CDN may cache the response)."""
    try:
        return _handle(raw_path, fetcher)
    except ApiError as e:
        return e.status, {"error": e.message, **e.extra}, 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 500, {"error": "Unexpected server error."}, 0