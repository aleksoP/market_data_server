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

def _add_technical_one_symbol(g: pd.DataFrame, vol_windows: list[int], atr_window: int) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()

    # Explicit no-fill to avoid pandas FutureWarning and accidental leakage-ish behavior
    g["ret_1"] = g["close"].pct_change(fill_method=None)

    # log returns
    close = g["close"].astype(float)
    g["logret_1"] = np.log(close).diff()

    # rolling vol on log returns
    for w in vol_windows:
        g[f"vol_logret_{w}"] = g["logret_1"].rolling(w, min_periods=w).std()

    # True Range + ATR
    prev_close = g["close"].shift(1)
    tr1 = (g["high"] - g["low"]).abs()
    tr2 = (g["high"] - prev_close).abs()
    tr3 = (g["low"] - prev_close).abs()
    g["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    g[f"atr_{atr_window}"] = g["true_range"].rolling(atr_window, min_periods=atr_window).mean()

    return g


def add_technical_features(df: pd.DataFrame, vol_windows: list[int] | None = None, atr_window: int = 14) -> pd.DataFrame:
    """
    Public API expected by src/features/pipeline.py.

    Input columns (minimum):
      timestamp_utc, open, high, low, close, volume, symbol(optional)

    Output adds:
      ret_1, logret_1, vol_logret_{w}, true_range, atr_{atr_window}
    """
    vol_windows = vol_windows or [30, 60, 390]

    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc"])

    if "symbol" in out.columns and out["symbol"].nunique() > 1:
        parts = []
        for sym, g in out.groupby("symbol", sort=True):
            parts.append(_add_technical_one_symbol(g, vol_windows, atr_window))
        return pd.concat(parts, ignore_index=True)

    return _add_technical_one_symbol(out, vol_windows, atr_window)
