from __future__ import annotations

import pandas as pd


def build_forward_returns(
    bars_long: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """
    horizons in minutes (for 1m bars). Output indexed by [timestamp_utc, symbol].
    Produces fwd_ret_{h}.
    """
    df = bars_long.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values(["symbol", "timestamp_utc"])
    df = df.drop_duplicates(["symbol", "timestamp_utc"], keep="last")

    out = df[["timestamp_utc", "symbol", "close"]].copy()

    for h in horizons:
        out[f"fwd_ret_{h}m"] = out.groupby("symbol")["close"].shift(-h) / out["close"] - 1.0

    out = out.set_index(["timestamp_utc", "symbol"]).sort_index()
    out = out.drop(columns=["close"])
    return out
