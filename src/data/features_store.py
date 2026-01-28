from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FeaturesStore:
    root_dir: Path = Path("data/features_1m")

    def _part_path(self, symbol: str, day: str) -> Path:
        return self.root_dir / f"symbol={symbol}" / f"date={day}" / "features.parquet"

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        days = pd.date_range(start=start, end=end, freq="D")
        parts = []
        for d in days:
            p = self._part_path(symbol, d.date().isoformat())
            if p.exists():
                parts.append(pd.read_parquet(p))
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df.sort_values(["symbol", "timestamp_utc"])

    def load_panel(self, symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
        parts = []
        for s in symbols:
            d = self.load(s, start, end)
            if not d.empty:
                parts.append(d)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True).sort_values(["symbol", "timestamp_utc"])
