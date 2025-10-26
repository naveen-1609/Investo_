#!/usr/bin/env python3
"""
Global LSTM trained on ALL tickers from DuckDB.

- If 'features' table doesn't exist, compute 15+ technical features from 'prices' for all tickers.
- Builds a global StandardScaler across ALL tickers' features.
- Uses a ticker embedding so the single model can learn per-ticker patterns.
- Saves:
    data/models/global_lstm.pt
    data/models/global_meta.json
    data/models/global_scaler.pkl
    data/models/ticker_index.json
"""

import argparse, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import duckdb
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DUCKDB_PATH = DATA_DIR / "duckdb" / "investo.duckdb"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SCALER_PATH = MODELS_DIR / "global_scaler.pkl"
TICKER_INDEX_PATH  = MODELS_DIR / "ticker_index.json"
META_PATH          = MODELS_DIR / "global_meta.json"
MODEL_PATH         = MODELS_DIR / "global_lstm.pt"

# ----------------------------
# Feature engineering (same family as your ingestion)
# ----------------------------
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

def bollinger(series: pd.Series, window: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return mid, mid + k*std, mid - k*std

def macd_parts(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    f = ema(series, fast); s = ema(series, slow)
    macd = f - s
    sig = ema(macd, signal)
    return macd, sig, macd - sig

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic_kd(high, low, close, k_period=14, d_period=3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll + 1e-12)
    d = k.rolling(d_period).mean()
    return k, d

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume.fillna(0)).cumsum()

FEATURE_COLS = [
    "ret_1d","ret_5d","ret_21d","log_ret_1d",
    "sma_5","sma_10","sma_20","ema_12","ema_26",
    "macd","macd_signal","macd_hist",
    "rsi_14",
    "bb_mid_20","bb_upper_20","bb_lower_20",
    "stoch_k_14","stoch_d_3",
    "atr_14","obv","vol_21"
]

DDL_FEATURES = """
CREATE TABLE IF NOT EXISTS features (
  date DATE, ticker VARCHAR,
  ret_1d DOUBLE, ret_5d DOUBLE, ret_21d DOUBLE, log_ret_1d DOUBLE,
  sma_5 DOUBLE, sma_10 DOUBLE, sma_20 DOUBLE,
  ema_12 DOUBLE, ema_26 DOUBLE,
  macd DOUBLE, macd_signal DOUBLE, macd_hist DOUBLE,
  rsi_14 DOUBLE,
  bb_mid_20 DOUBLE, bb_upper_20 DOUBLE, bb_lower_20 DOUBLE,
  stoch_k_14 DOUBLE, stoch_d_3 DOUBLE,
  atr_14 DOUBLE, obv DOUBLE, vol_21 DOUBLE
);
"""

def compute_features_for_all_tickers(con: duckdb.DuckDBPyConnection):
    # pull prices for all tickers
    prices = con.execute("SELECT * FROM prices ORDER BY ticker, date").df()
    if prices.empty:
        raise RuntimeError("No data in 'prices'. Run ingestion first.")
    out = []
    for t, g in prices.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        price = g["adj_close"]

        g["ret_1d"]  = price.pct_change()
        g["ret_5d"]  = price.pct_change(5)
        g["ret_21d"] = price.pct_change(21)
        g["log_ret_1d"] = np.log(price / price.shift(1))
        g["vol_21"] = g["ret_1d"].rolling(21).std() * np.sqrt(252)

        g["sma_5"]  = price.rolling(5).mean()
        g["sma_10"] = price.rolling(10).mean()
        g["sma_20"] = price.rolling(20).mean()
        g["ema_12"] = ema(price, 12)
        g["ema_26"] = ema(price, 26)

        g["macd"], g["macd_signal"], g["macd_hist"] = macd_parts(price, 12, 26, 9)

        g["rsi_14"] = rsi(price, 14)

        g["bb_mid_20"], g["bb_upper_20"], g["bb_lower_20"] = bollinger(price, 20, 2.0)

        g["stoch_k_14"], g["stoch_d_3"] = stochastic_kd(g["high"], g["low"], g["close"], 14, 3)

        g["atr_14"] = atr(g["high"], g["low"], g["close"], 14)

        g["obv"] = obv(price, g["volume"])

        g2 = g[["date","ticker"] + FEATURE_COLS].dropna().reset_index(drop=True)
        out.append(g2)

    feats = pd.concat(out).reset_index(drop=True)
    con.execute(DDL_FEATURES)
    con.register("df_feats_all", feats)
    con.execute("DELETE FROM features;")
    con.execute("INSERT INTO features SELECT * FROM df_feats_all;")
    con.unregister("df_feats_all")
    return len(feats)

def ensure_features_table(con: duckdb.DuckDBPyConnection):
    # Robust check using information_schema (avoids PRAGMA issues)
    exists = con.execute("""
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'features'
        LIMIT 1
    """).fetchone()
    if exists is None:
        print("[features] not found → building features for all tickers from 'prices' …")
        n = compute_features_for_all_tickers(con)
        print(f"[features] built {n} feature rows.")
    else:
        # Optionally check columns, but we trust ingestion’s schema
        pass

# ----------------------------
# Data assembly (global)
# ----------------------------
def load_all_joined(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # Join prices + features for all tickers
    cols = ",".join([f"f.{c}" for c in FEATURE_COLS])
    sql = f"""
        SELECT p.date, p.ticker, p.adj_close, {cols}
        FROM prices p
        JOIN features f USING(date, ticker)
        ORDER BY p.ticker, p.date
    """
    df = con.execute(sql).df().dropna()
    return df

def build_ticker_index(df: pd.DataFrame) -> Dict[str, int]:
    uniq = sorted(df["ticker"].unique().tolist())
    return {t: i for i, t in enumerate(uniq)}

class GlobalSeqDataset(Dataset):
    """
    Builds windows per ticker; concatenates across tickers into a single dataset.

    X: [N, T, F],  ticker_id: [N],  y: [N, H]
    """
    def __init__(self, df: pd.DataFrame, lookback: int, horizon: int, feat_cols: List[str], ticker_index: Dict[str,int]):
        self.X, self.Y, self.TID = [], [], []
        for t, g in df.groupby("ticker"):
            g = g.sort_values("date").reset_index(drop=True)
            valsX = g[feat_cols].values.astype(np.float32)
            ycol  = g["adj_close"].values.astype(np.float32)
            tid = ticker_index[t]
            for i in range(lookback, len(g) - horizon):
                xwin = valsX[i-lookback:i, :]
                base = ycol[i-1]
                future = ycol[i:i+horizon]
                yrel = (future - base) / (base + 1e-12)
                self.X.append(xwin)
                self.Y.append(yrel)
                self.TID.append(tid)
        self.X   = np.asarray(self.X, dtype=np.float32)
        self.Y   = np.asarray(self.Y, dtype=np.float32)
        self.TID = np.asarray(self.TID, dtype=np.int64)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.TID[idx], self.Y[idx]

def timeseries_split_by_ticker(df: pd.DataFrame, lookback: int, horizon: int, split_ratio: float = 0.8):
    """
    Build dataset then split by index (global) so that for each ticker the last chunk lands in val.
    Simpler approach: after dataset creation, use a simple ratio split (it will approximate per-ticker split).
    """
    return split_ratio

# ----------------------------
# Model (with ticker embedding)
# ----------------------------
class GlobalLSTM(nn.Module):
    def __init__(self, in_dim: int, embed_num: int, embed_dim: int, hidden: int, layers: int, horizon: int, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=embed_num, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=in_dim + embed_dim, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, tid):
        # x: [B, T, F], tid: [B]
        emb = self.embed(tid)           # [B, E]
        # repeat embedding across time steps
        emb_rep = emb.unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, E]
        xcat = torch.cat([x, emb_rep], dim=-1)              # [B, T, F+E]
        y, _ = self.lstm(xcat)
        last = y[:, -1, :]
        out = self.head(last)           # [B, H]
        return out

# ----------------------------
# Training loop
# ----------------------------
def train_global(lookback: int, horizon: int, hidden: int, layers: int, embed_dim: int,
                 dropout: float, epochs: int, batch: int, lr: float, val_ratio: float):
    con = duckdb.connect(str(DUCKDB_PATH))
    # 1) ensure features table
    ensure_features_table(con)

    # 2) load joined data
    df = load_all_joined(con)
    if df.empty:
        raise RuntimeError("Joined data is empty. Check 'prices' and 'features'.")

    # 3) global scaler on features
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    joblib.dump(scaler, GLOBAL_SCALER_PATH)

    # 4) ticker index + dataset
    ticker_index = build_ticker_index(df)
    Path(TICKER_INDEX_PATH).write_text(json.dumps(ticker_index, indent=2))

    ds = GlobalSeqDataset(df, lookback, horizon, FEATURE_COLS, ticker_index)
    n = len(ds)
    if n < 100:
        raise RuntimeError(f"Very few training windows: {n}. Increase history or reduce lookback/horizon.")

    # simple global split
    n_train = int(n * (1 - val_ratio))
    train_ds = torch.utils.data.Subset(ds, np.arange(0, n_train))
    val_ds   = torch.utils.data.Subset(ds, np.arange(n_train, n))

    dl_train = DataLoader(train_ds, batch_size=batch, shuffle=True)
    dl_val   = DataLoader(val_ds,   batch_size=max(128, batch), shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlobalLSTM(in_dim=len(FEATURE_COLS),
                       embed_num=len(ticker_index),
                       embed_dim=embed_dim,
                       hidden=hidden, layers=layers,
                       horizon=horizon, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_val = 1e9
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, tid, y in dl_train:
            x, tid, y = x.to(device), tid.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x, tid)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr_sum += loss.item(); tr_n += 1

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for x, tid, y in dl_val:
                x, tid, y = x.to(device), tid.to(device), y.to(device)
                pred = model(x, tid)
                va_sum += loss_fn(pred, y).item(); va_n += 1

        tr = tr_sum / max(1, tr_n)
        va = va_sum / max(1, va_n)
        print(f"[global] epoch {ep:02d}/{epochs}  train_MAE={tr:.5f}  val_MAE={va:.5f}")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model and meta
    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        "model": "GlobalLSTM",
        "feature_cols": FEATURE_COLS,
        "lookback": lookback,
        "horizon": horizon,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "embed_dim": embed_dim,
        "val_ratio": val_ratio,
        "best_val_mae": float(best_val),
        "scaler_path": str(GLOBAL_SCALER_PATH.name),
        "ticker_index_path": str(TICKER_INDEX_PATH.name)
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"✅ Saved: {MODEL_PATH.name}, {META_PATH.name}, {GLOBAL_SCALER_PATH.name}, {TICKER_INDEX_PATH.name}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon",  type=int, default=30)
    ap.add_argument("--hidden",   type=int, default=64)
    ap.add_argument("--layers",   type=int, default=2)
    ap.add_argument("--embed-dim",type=int, default=8)
    ap.add_argument("--dropout",  type=float, default=0.2)
    ap.add_argument("--epochs",   type=int, default=20)
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--val-ratio",type=float, default=0.2)
    args = ap.parse_args()

    train_global(
        lookback=args.lookback, horizon=args.horizon,
        hidden=args.hidden, layers=args.layers, embed_dim=args.embed_dim,
        dropout=args.dropout, epochs=args.epochs, batch=args.batch,
        lr=args.lr, val_ratio=args.val_ratio
    )
   
  