#!/usr/bin/env python3
"""
Generate daily predictions for the next N years using the global LSTM.

Outputs:
- DuckDB table: predictions (as_of_ts, ticker, pred_date, pred_price, step_index, model)
- Parquet dataset partitioned by ticker: data/predictions/ticker=XYZ/part-*.parquet
- DuckDB view 'predictions_hive' over the partitioned Parquet

Assumptions:
- You trained with train_global_lstm.py and produced:
  data/models/global_lstm.pt
  data/models/global_meta.json
  data/models/global_scaler.pkl
  data/models/ticker_index.json
- 'prices' and 'features' tables exist in DuckDB
"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import duckdb
import joblib
import torch
import torch.nn as nn
from datetime import datetime, timezone

# -------- Paths --------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DUCKDB_PATH = DATA_DIR / "duckdb" / "investo.duckdb"
PRED_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"

MODEL_PATH = MODELS_DIR / "global_lstm.pt"
META_PATH  = MODELS_DIR / "global_meta.json"
SCALER_PATH = MODELS_DIR / "global_scaler.pkl"
TICKERS_PATH = MODELS_DIR / "ticker_index.json"

# -------- Feature columns (must match training meta) --------
def load_meta_and_assets():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}")
    meta = json.loads(META_PATH.read_text())
    feat_cols = meta["feature_cols"]
    lookback = int(meta["lookback"])
    horizon  = int(meta["horizon"])
    model_name = meta.get("model", "GlobalLSTM")

    scaler = joblib.load(SCALER_PATH)
    ticker_index: Dict[str,int] = json.loads(TICKERS_PATH.read_text())

    return meta, feat_cols, lookback, horizon, model_name, scaler, ticker_index

# -------- Model definition (must match training) --------
class GlobalLSTM(nn.Module):
    def __init__(self, in_dim: int, embed_num: int, embed_dim: int, hidden: int, layers: int, horizon: int, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=embed_num, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=in_dim + embed_dim, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, tid):
        # x: [B, T, F], tid: [B]
        emb = self.embed(tid)                       # [B, E]
        emb_rep = emb.unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, E]
        xcat = torch.cat([x, emb_rep], dim=-1)      # [B, T, F+E]
        y, _ = self.lstm(xcat)
        last = y[:, -1, :]
        return self.head(last)                      # [B, H] relative returns

# -------- Minimal tech-indicator functions (mirror training) --------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return mid, mid + num_std*std, mid - num_std*std

def macd_parts(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast); s = ema(series, slow)
    macd = f - s
    sig = ema(macd, signal)
    return macd, sig, macd - sig

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high-low).abs(),
        (high-prev_close).abs(),
        (low-prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic_kd(high, low, close, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll + 1e-12)
    d = k.rolling(d_period).mean()
    return k, d

def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume.fillna(0)).cumsum()

def compute_features_block(df_block: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the same feature set as training for a dataframe that at least contains:
    date, ticker, adj_close, high, low, close, volume
    Returns dataframe with only feature columns.
    """
    price = df_block["adj_close"]
    out = pd.DataFrame(index=df_block.index)
    out["ret_1d"]  = price.pct_change()
    out["ret_5d"]  = price.pct_change(5)
    out["ret_21d"] = price.pct_change(21)
    out["log_ret_1d"] = np.log(price / price.shift(1))
    out["vol_21"] = out["ret_1d"].rolling(21).std() * np.sqrt(252)

    out["sma_5"]  = price.rolling(5).mean()
    out["sma_10"] = price.rolling(10).mean()
    out["sma_20"] = price.rolling(20).mean()
    out["ema_12"] = ema(price, 12)
    out["ema_26"] = ema(price, 26)

    out["macd"], out["macd_signal"], out["macd_hist"] = macd_parts(price, 12, 26, 9)
    out["rsi_14"] = rsi(price, 14)
    out["bb_mid_20"], out["bb_upper_20"], out["bb_lower_20"] = bollinger_bands(price, 20, 2.0)
    out["stoch_k_14"], out["stoch_d_3"] = stochastic_kd(df_block["high"], df_block["low"], df_block["close"], 14, 3)
    out["atr_14"] = atr(df_block["high"], df_block["low"], df_block["close"], 14)
    out["obv"] = on_balance_volume(price, df_block["volume"])
    return out

# -------- Data pull --------
def load_last_window(con, ticker: str, feat_cols: List[str], lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      hist_tail: last (lookback) rows with columns: date, adj_close, high, low, close, volume, ticker
      feats_tail: scaled feature matrix (lookback x F) for model input
    """
    # Join prices + features for this ticker
    cols = ", ".join([f"f.{c}" for c in feat_cols])
    sql = f"""
        SELECT p.date, p.ticker, p.adj_close, p.high, p.low, p.close, p.volume, {cols}
        FROM prices p
        JOIN features f USING(date, ticker)
        WHERE p.ticker = ?
        ORDER BY p.date
    """
    df = con.execute(sql, [ticker]).df()
    if len(df) < lookback:
        raise RuntimeError(f"{ticker}: not enough rows ({len(df)}) for lookback={lookback}")
    hist = df[["date","ticker","adj_close","high","low","close","volume"]].copy()
    feats = df[feat_cols].copy()
    return hist.tail(lookback).reset_index(drop=True), feats.tail(lookback).reset_index(drop=True)

# -------- Future calendar --------
def make_future_bd_dates(start_date: pd.Timestamp, days: int) -> List[pd.Timestamp]:
    # Simple business-day calendar (Mon-Fri). No holiday filter to avoid extra deps.
    rng = pd.bdate_range(start_date + pd.Timedelta(days=1), periods=days)
    return list(rng)

# -------- Rolling forecast --------
def roll_forecast_ticker(
    ticker: str,
    model: nn.Module,
    scaler,
    ticker_index: Dict[str,int],
    feat_cols: List[str],
    lookback: int,
    horizon: int,
    years: int,
    device: str,
    con,
) -> pd.DataFrame:
    """
    Produce predictions for next (years) years at business-day frequency.
    We forward-fill unknown future OHLC/volume with simple proxies:
      - high=low=close=adj_close (synthetic)
      - volume = last observed volume
    This keeps feature computation consistent for iterative steps.
    """
    hist_tail, feats_tail = load_last_window(con, ticker, feat_cols, lookback)
    last_price = float(hist_tail["adj_close"].iloc[-1])
    last_volume = float(hist_tail["volume"].iloc[-1])

    # scale last window features
    Xwin = feats_tail.values.astype(np.float32)
    Xwin = scaler.transform(Xwin)  # global scaler
    Xwin = torch.from_numpy(Xwin[None, ...]).to(device)  # shape [1, T, F]

    tid = torch.tensor([ticker_index[ticker]], dtype=torch.long, device=device)

    steps_needed = int(np.ceil((years * 365)))  # rough; we'll generate business days anyway
    future_dates = make_future_bd_dates(pd.to_datetime(hist_tail["date"].iloc[-1]), steps_needed)
    # We'll cut to length after generation

    preds_list = []
    synthetic_hist = hist_tail.copy()

    i = 0
    while i < len(future_dates):
        # Predict next 'horizon' relative returns in one shot
        with torch.no_grad():
            yrel = model(Xwin, tid).detach().cpu().numpy()[0]  # length = horizon
        # Convert to prices path
        p = last_price
        chunk_prices = []
        for r in yrel:
            p = p * (1.0 + float(r))
            chunk_prices.append(p)

        # Build synthetic OHLCV rows for this chunk
        chunk_len = min(horizon, len(future_dates) - i)
        ch_dates = future_dates[i:i+chunk_len]
        ch_df = pd.DataFrame({
            "date": [d.date() for d in ch_dates],
            "ticker": ticker,
            "adj_close": chunk_prices[:chunk_len],
            "high": chunk_prices[:chunk_len],
            "low": chunk_prices[:chunk_len],
            "close": chunk_prices[:chunk_len],
            "volume": [last_volume]*chunk_len
        })

        # Append to synthetic history to compute rolling features for NEXT iteration
        # Only the latest 'lookback' rows matter for the next window
        combo = pd.concat([synthetic_hist, ch_df], ignore_index=True)
        feats_all = compute_features_block(combo)
        feats_all = feats_all.iloc[:, :]  # keep order
        # build NEXT window features (last `lookback`)
        next_feats = feats_all.tail(lookback).copy()
        # Some early rows in next_feats may be NaN due to rolling windows; if so, pad by taking more history
        if next_feats.isna().values.any():
            # take as many rows as needed from combo tail to fill a clean window
            # find the minimal tail that yields no NaN after scaling
            for lb in range(lookback, min(len(combo), lookback*3)+1):
                candidate = feats_all.tail(lb).copy()
                if not candidate.tail(lookback).isna().values.any():
                    next_feats = candidate.tail(lookback).copy()
                    break

        # scale and prep for next call
        Xwin = next_feats.values.astype(np.float32)
        Xwin = scaler.transform(Xwin)
        Xwin = torch.from_numpy(Xwin[None, ...]).to(device)

        # update trackers
        last_price = float(ch_df["adj_close"].iloc[-1])
        synthetic_hist = combo  # keep growing (small; just DataFrame, OK locally)

        # collect predictions
        preds_list.append(ch_df[["date","ticker","adj_close"]])
        i += chunk_len

    preds = pd.concat(preds_list, ignore_index=True)
    preds = preds.head(len(future_dates))  # trim if we overshot
    preds = preds.rename(columns={"adj_close":"pred_price"})
    return preds

# -------- Write outputs --------
def ensure_predictions_table(con):
    con.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
      as_of_ts TIMESTAMP,
      ticker   VARCHAR,
      pred_date DATE,
      pred_price DOUBLE,
      step_index INTEGER,
      model VARCHAR
    );
    """)

def write_duckdb(con, preds_df: pd.DataFrame, model_name: str):
    ensure_predictions_table(con)
    as_of = pd.Timestamp.now(tz=timezone.utc)
    preds_df = preds_df.copy()
    preds_df["as_of_ts"] = as_of
    preds_df["step_index"] = np.arange(1, len(preds_df)+1)
    preds_df["model"] = model_name
    preds_df = preds_df.rename(columns={"date":"pred_date"})
    con.register("preds_stage", preds_df)
    con.execute("INSERT INTO predictions SELECT as_of_ts, ticker, pred_date, pred_price, step_index, model FROM preds_stage;")
    con.unregister("preds_stage")

def write_partitioned_parquet(preds_by_ticker: Dict[str, pd.DataFrame]):
    # Hive-style: data/predictions/ticker=XYZ/part-*.parquet
    for t, df in preds_by_ticker.items():
        out_dir = PRED_DIR / f"ticker={t}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df_out = df.rename(columns={"date":"pred_date"})
        # small chunks are fine; overwrite by filename that includes as_of
        as_of_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        df_out.to_parquet(out_dir / f"part-{as_of_tag}.parquet", index=False)

def create_or_replace_hive_view(con):
    # A view over the partitioned parquet dataset
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path_glob = str(PRED_DIR / "ticker=*/part-*.parquet").replace("\\", "/")
    con.execute("DROP VIEW IF EXISTS predictions_hive;")
    con.execute(f"""
        CREATE VIEW predictions_hive AS
        SELECT * FROM read_parquet('{path_glob}', hive_partitioning=1);
    """)

# -------- Main --------
def main(years: int):
    meta, feat_cols, lookback, horizon, model_name, scaler, ticker_index = load_meta_and_assets()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlobalLSTM(
        in_dim=len(feat_cols),
        embed_num=len(ticker_index),
        embed_dim=int(meta.get("embed_dim", 8)),
        hidden=int(meta.get("hidden", 64)),
        layers=int(meta.get("layers", 2)),
        horizon=horizon,
        dropout=float(meta.get("dropout", 0.2))
    ).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    con = duckdb.connect(str(DUCKDB_PATH))

    # Which tickers to predict for? Use keys from ticker_index (trained universe)
    tickers = sorted(ticker_index.keys())
    preds_by_ticker: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        print(f"[predict] {t}: rolling {years}y forecast in chunks of {horizon}d …")
        preds = roll_forecast_ticker(
            ticker=t,
            model=model,
            scaler=scaler,
            ticker_index=ticker_index,
            feat_cols=feat_cols,
            lookback=lookback,
            horizon=horizon,
            years=years,
            device=device,
            con=con,
        )
        preds_by_ticker[t] = preds

    # Write outputs
    # 1) DuckDB table (all tickers)
    combined = []
    for t, df in preds_by_ticker.items():
        df2 = df.copy()
        df2["ticker"] = t
        combined.append(df2)
    all_preds = pd.concat(combined, ignore_index=True)
    write_duckdb(con, all_preds, model_name=model_name)

    # 2) Parquet partitioned by ticker
    write_partitioned_parquet(preds_by_ticker)

    # 3) Create view over partitioned hive dataset
    create_or_replace_hive_view(con)

    # Small sample print
    print(con.execute("""
      SELECT ticker, MIN(pred_date) AS start, MAX(pred_date) AS end, COUNT(*) AS n
      FROM predictions
      GROUP BY 1 ORDER BY ticker
    """).df())

    con.close()
    print("✅ predictions written to DuckDB table `predictions` and Parquet partitions under data/predictions/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10, help="Number of future years to predict (business days)")
    args = ap.parse_args()
    main(args.years)
