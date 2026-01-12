from __future__ import annotations

import pandas as pd


def add_lagged_returns(g: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Adds backward-looking returns over multiple lags.

    Expects columns:
      - timestamp_utc
      - close

    Produces:
      - ret_lag_{k} for each k in lags
    """
    g = g.sort_values("timestamp_utc").copy()
    for k in lags:
        g[f"ret_lag_{k}"] = g["close"].pct_change(k, fill_method=None)
    return g
