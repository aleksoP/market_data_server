from __future__ import annotations

import pandas as pd


def add_liquidity_proxies(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()
    g["dollar_volume"] = g["close"] * g["volume"]
    g["hl_range"] = (g["high"] - g["low"]) / g["close"]
    return g
