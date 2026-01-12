from __future__ import annotations

import pandas as pd


def ensure_regular_index(
    df_long: pd.DataFrame,
    freq: str = "1min",
    on: str = "timestamp_utc",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """
    Ensures a regular timestamp grid *per symbol* without forward filling.
    Missing rows are inserted with NaNs.

    Input long: [timestamp_utc, symbol, ...]
    Output long with complete grids per symbol.
    """
    df = df_long.copy()
    df[on] = pd.to_datetime(df[on], utc=True)
    out = []
    for sym, g in df.groupby(symbol_col, sort=True):
        g = g.sort_values(on).drop_duplicates([on], keep="last")
        if g.empty:
            continue
        full_idx = pd.date_range(g[on].min(), g[on].max(), freq=freq, tz="UTC")
        g2 = g.set_index(on).reindex(full_idx)
        g2.index.name = on
        g2[symbol_col] = sym
        out.append(g2.reset_index())
    if not out:
        return df.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True)


def common_timestamp_index(df_long: pd.DataFrame, on: str = "timestamp_utc") -> pd.DatetimeIndex:
    """
    Union of all timestamps.
    """
    ts = pd.to_datetime(df_long[on], utc=True)
    return pd.DatetimeIndex(sorted(ts.unique()))
