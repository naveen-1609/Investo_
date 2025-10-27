#!/usr/bin/env python3
"""
Local Investo Ingestion:
- Reads tickers from a JSON file ({"tickers": ["AAPL","AMZN", ...]} or ["AAPL", ...])
- Ensures DuckDB is initialized (data/duckdb/investo.duckdb)
- Upserts yfinance OHLCV into `prices`
- Builds 15+ features and upserts into `features`
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple
import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------
# Paths / Configs
# ------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DUCKDB_DIR = DATA_DIR / "duckdb"
DUCKDB_PATH = DUCKDB_DIR / "investo.duckdb"   # <- consistent location
RAW_DIR = DATA_DIR / "raw"
MIN_ROWS_REQUIRED = 2000  # adjust if needed (e.g., 500 for newer tickers)

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ingestion")

# ------------------------------
# DDL
# ------------------------------
DDL_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
  date DATE,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE,
  volume BIGINT, dividends DOUBLE, stock_splits DOUBLE,
  ticker VARCHAR
);
"""

# DDL_FEATURES is now imported from feature_engineering module

def ensure_duckdb_initialized() -> bool:
    already = DUCKDB_PATH.exists()
    DUCKDB_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(DDL_PRICES)
    con.execute(get_features_ddl())
    con.close()
    return already

def connect_duckdb():
    return duckdb.connect(str(DUCKDB_PATH))

# ------------------------------
# Yahoo fetch (with retries)
# ------------------------------
def fetch_yf_one(ticker: str, retries: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(ticker, period="max", interval="1d", auto_adjust=False, actions=True, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={"Date": "date"})
                df["ticker"] = ticker
                req = ["Open","High","Low","Close","Adj Close","Volume","Dividends","Stock Splits"]
                for c in req:
                    if c not in df.columns:
                        df[c] = np.nan
                df = df[["date","Open","High","Low","Close","Adj Close","Volume","Dividends","Stock Splits","ticker"]]
                df.columns = ["date","open","high","low","close","adj_close","volume","dividends","stock_splits","ticker"]
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df.sort_values("date").dropna(subset=["date","ticker"])
            last_err = RuntimeError(f"No data for {ticker}")
        except Exception as e:
            last_err = e
        time.sleep(backoff * (i + 1))
    raise last_err

# ------------------------------
# Feature engineering (imported from shared module)
# ------------------------------
from feature_engineering import build_features_for_ticker, get_features_ddl, get_feature_columns

# ------------------------------
# Upserts
# ------------------------------
def upsert_prices(con, df: pd.DataFrame, replace_existing: bool = False) -> int:
    if df.empty:
        return 0
    ticker = df["ticker"].iloc[0]
    con.execute(DDL_PRICES)

    if replace_existing:
        con.execute("DELETE FROM prices WHERE ticker = ?", [ticker])

    con.register("df_stage", df)
    con.execute("DROP TABLE IF EXISTS _stage;")
    con.execute("CREATE TEMP TABLE _stage AS SELECT * FROM df_stage;")
    con.execute("""
        MERGE INTO prices AS tgt
        USING _stage AS src
        ON tgt.date = src.date AND tgt.ticker = src.ticker
        WHEN MATCHED THEN UPDATE SET
            open=src.open, high=src.high, low=src.low, close=src.close, adj_close=src.adj_close,
            volume=src.volume, dividends=src.dividends, stock_splits=src.stock_splits
        WHEN NOT MATCHED THEN INSERT
            (date, open, high, low, close, adj_close, volume, dividends, stock_splits, ticker)
        VALUES
            (src.date, src.open, src.high, src.low, src.close, src.adj_close, src.volume, src.dividends, src.stock_splits, src.ticker);
    """)
    con.execute("DROP TABLE IF EXISTS _stage;")
    try:
        con.unregister("df_stage")
    except:
        pass
    return len(df)

def upsert_features(con, feats: pd.DataFrame, replace_existing: bool = False) -> int:
    if feats.empty:
        return 0
    ticker = feats["ticker"].iloc[0]
    con.execute(get_features_ddl())

    if replace_existing:
        con.execute("DELETE FROM features WHERE ticker = ?", [ticker])

    con.register("df_feats", feats)
    con.execute("DROP TABLE IF EXISTS _stagef;")
    con.execute("CREATE TEMP TABLE _stagef AS SELECT * FROM df_feats;")
    con.execute("""
        MERGE INTO features AS tgt
        USING _stagef AS src
        ON tgt.date = src.date AND tgt.ticker = src.ticker
        WHEN MATCHED THEN UPDATE SET
          ret_1d=src.ret_1d, ret_5d=src.ret_5d, ret_21d=src.ret_21d, log_ret_1d=src.log_ret_1d,
          sma_5=src.sma_5, sma_10=src.sma_10, sma_20=src.sma_20,
          ema_12=src.ema_12, ema_26=src.ema_26,
          macd=src.macd, macd_signal=src.macd_signal, macd_hist=src.macd_hist,
          rsi_14=src.rsi_14,
          bb_mid_20=src.bb_mid_20, bb_upper_20=src.bb_upper_20, bb_lower_20=src.bb_lower_20,
          stoch_k_14=src.stoch_k_14, stoch_d_3=src.stoch_d_3,
          atr_14=src.atr_14, obv=src.obv, vol_21=src.vol_21
        WHEN NOT MATCHED THEN INSERT
          (date, ticker,
           ret_1d, ret_5d, ret_21d, log_ret_1d,
           sma_5, sma_10, sma_20, ema_12, ema_26,
           macd, macd_signal, macd_hist,
           rsi_14,
           bb_mid_20, bb_upper_20, bb_lower_20,
           stoch_k_14, stoch_d_3,
           atr_14, obv, vol_21)
        VALUES
          (src.date, src.ticker,
           src.ret_1d, src.ret_5d, src.ret_21d, src.log_ret_1d,
           src.sma_5, src.sma_10, src.sma_20, src.ema_12, src.ema_26,
           src.macd, src.macd_signal, src.macd_hist,
           src.rsi_14,
           src.bb_mid_20, src.bb_upper_20, src.bb_lower_20,
           src.stoch_k_14, src.stoch_d_3,
           src.atr_14, src.obv, src.vol_21);
    """)
    con.execute("DROP TABLE IF EXISTS _stagef;")
    try:
        con.unregister("df_feats")
    except:
        pass
    return len(feats)

# ------------------------------
# Orchestration
# ------------------------------
def load_tickers(json_path: Path) -> List[str]:
    obj = json.loads(Path(json_path).read_text())
    if isinstance(obj, dict) and "tickers" in obj:
        return [t.strip().upper() for t in obj["tickers"] if t and isinstance(t, str)]
    if isinstance(obj, list):
        return [t.strip().upper() for t in obj if t and isinstance(t, str)]
    raise ValueError("ticker.json must be {'tickers': [ ... ]} or a list of strings")

def ingest_one(con: duckdb.DuckDBPyConnection, ticker: str, replace_existing: bool) -> Tuple[int, int]:
    log.info(f"↻ {ticker}: fetching yfinance…")
    df = fetch_yf_one(ticker)
    if df.empty or len(df) < MIN_ROWS_REQUIRED:
        log.warning(f"  Skipping {ticker}: insufficient rows ({len(df)}).")
        return 0, 0

    # Persist raw parquet (correct usage)
    raw_path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(raw_path, index=False)

    n_prices = upsert_prices(con, df, replace_existing=replace_existing)
    log.info(f"  ✓ upserted {n_prices} rows into prices")

    feats = build_features_for_ticker(df)
    n_feats = upsert_features(con, feats, replace_existing=replace_existing)
    log.info(f"  ✓ upserted {n_feats} rows into features")

    return n_prices, n_feats

def main(json_path: str, replace_existing: bool):
    existed = ensure_duckdb_initialized()
    log.info(f"🐤 DuckDB at {DUCKDB_PATH} — {'found' if existed else 'created'}")

    tickers = load_tickers(Path(json_path))
    if not tickers:
        raise SystemExit("No tickers found in JSON.")

    con = connect_duckdb()
    total_p = total_f = 0
    for t in tickers:
        p, f = ingest_one(con, t, replace_existing=replace_existing)
        total_p += p; total_f += f
    con.close()
    log.info(f"✅ Done. Prices upserted: {total_p:,} | Features upserted: {total_f:,}")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-json", required=True, help="Path to ticker.json (e.g., {'tickers':['AAPL','NVDA']})")
    ap.add_argument("--replace-existing", action="store_true", help="Delete this ticker's existing rows before insert")
    args = ap.parse_args()
    main(args.tickers_json, replace_existing=args.replace_existing)
