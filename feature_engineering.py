#!/usr/bin/env python3
"""
Shared Feature Engineering Module for Investo

This module contains all technical indicators and feature engineering functions
used by both Ingestion.py and ModelBuilding.py to avoid code duplication.

Features included:
- Returns: 1d, 5d, 21d, log returns
- Moving Averages: SMA (5, 10, 20), EMA (12, 26)
- Technical Indicators: MACD, RSI, Bollinger Bands, Stochastic, ATR, OBV
- Volatility: 21-day rolling volatility
"""

import numpy as np
import pandas as pd
from typing import Tuple

# ------------------------------
# Feature Engineering Functions
# ------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle, upper, lower"""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower

def macd_parts(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: macd line, signal line, histogram"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator: %K and %D"""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(d_period).mean()
    return k, d

def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume"""
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume.fillna(0)).cumsum()

# ------------------------------
# Feature Building Function
# ------------------------------

def build_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all technical features for a single ticker's price data.
    
    Args:
        df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']
    
    Returns:
        DataFrame with features: ['date', 'ticker'] + feature columns
    """
    g = df.copy().reset_index(drop=True)
    price = g["adj_close"]

    # Returns
    g["ret_1d"] = price.pct_change()
    g["ret_5d"] = price.pct_change(5)
    g["ret_21d"] = price.pct_change(21)
    g["log_ret_1d"] = np.log(price / price.shift(1))
    g["vol_21"] = g["ret_1d"].rolling(21).std() * np.sqrt(252)

    # Moving Averages
    g["sma_5"] = price.rolling(5).mean()
    g["sma_10"] = price.rolling(10).mean()
    g["sma_20"] = price.rolling(20).mean()
    g["ema_12"] = ema(price, 12)
    g["ema_26"] = ema(price, 26)

    # Technical Indicators
    g["macd"], g["macd_signal"], g["macd_hist"] = macd_parts(price, 12, 26, 9)
    g["rsi_14"] = rsi(price, 14)
    g["bb_mid_20"], g["bb_upper_20"], g["bb_lower_20"] = bollinger_bands(price, 20, 2.0)
    g["stoch_k_14"], g["stoch_d_3"] = stochastic_kd(g["high"], g["low"], g["close"], 14, 3)
    g["atr_14"] = atr(g["high"], g["low"], g["close"], 14)
    g["obv"] = on_balance_volume(price, g["volume"])

    # Select feature columns
    feature_cols = [
        "ret_1d", "ret_5d", "ret_21d", "log_ret_1d",
        "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14",
        "bb_mid_20", "bb_upper_20", "bb_lower_20",
        "stoch_k_14", "stoch_d_3",
        "atr_14", "obv", "vol_21"
    ]
    
    result = g[["date", "ticker"] + feature_cols].dropna().reset_index(drop=True)
    return result

# ------------------------------
# Constants and Schema
# ------------------------------

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_21d", "log_ret_1d",
    "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_mid_20", "bb_upper_20", "bb_lower_20",
    "stoch_k_14", "stoch_d_3",
    "atr_14", "obv", "vol_21"
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

# ------------------------------
# Utility Functions
# ------------------------------

def get_feature_columns() -> list:
    """Get list of feature column names"""
    return FEATURE_COLS.copy()

def get_features_ddl() -> str:
    """Get DDL for features table"""
    return DDL_FEATURES

def validate_feature_dataframe(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has required feature columns"""
    required_cols = ["date", "ticker"] + FEATURE_COLS
    return all(col in df.columns for col in required_cols)
