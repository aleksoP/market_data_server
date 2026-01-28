from __future__ import annotations

import numpy as np
import pandas as pd


def add_liquidity_proxies(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()
    g["dollar_volume"] = g["close"] * g["volume"]
    g["hl_range"] = (g["high"] - g["low"]) / g["close"]
    return g

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Public API expected by src/features/pipeline.py.
    Minimal proxies using only OHLCV (no L1/L2).

    Adds:
      dollar_volume
      hl_range
      hl_range_pct
    """
    g = df.copy()
    g["dollar_volume"] = g["close"].astype(float) * g["volume"].astype(float)

    hl = (g["high"].astype(float) - g["low"].astype(float)).abs()
    g["hl_range"] = hl

    close = g["close"].astype(float).replace(0.0, np.nan)
    g["hl_range_pct"] = hl / close

    return g
