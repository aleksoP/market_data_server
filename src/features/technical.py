from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_returns(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()
    g["ret_1"] = g["close"].pct_change(fill_method=None)
    g["logret_1"] = np.log(g["close"]).diff()
    return g


def add_rolling_vol(g: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()
    for w in windows:
        g[f"vol_logret_{w}"] = g["logret_1"].rolling(w, min_periods=w).std()
    return g


def add_true_range_atr(g: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()
    prev_close = g["close"].shift(1)
    tr1 = g["high"] - g["low"]
    tr2 = (g["high"] - prev_close).abs()
    tr3 = (g["low"] - prev_close).abs()
    g["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    g[f"atr_{atr_window}"] = g["true_range"].rolling(atr_window, min_periods=atr_window).mean()
    return g
