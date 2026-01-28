from __future__ import annotations

import pandas as pd


def every_bar_events(bars_long: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal event sampler: every timestamp is an event.
    Returns index [timestamp_utc, symbol] with a single column 'event'=1.
    """
    df = bars_long[["timestamp_utc", "symbol"]].drop_duplicates().copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["event"] = 1
    return df.set_index(["timestamp_utc", "symbol"]).sort_index()
