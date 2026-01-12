from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_TS_CANDIDATES = ["timestamp_utc", "timestamp", "ts", "datetime", "date_time", "time", "date"]



@dataclass(frozen=True)
class BarsStore:
    """
    Reader for partitioned bars store:
      data/bars_1m/symbol=XYZ/date=YYYY-MM-DD/bars.parquet
    """
    root_dir: Path
    bar_freq: str = "1min"

    def _symbol_dir(self, symbol: str) -> Path:
        return self.root_dir / f"symbol={symbol}"

    def list_symbols(self) -> list[str]:
        if not self.root_dir.exists():
            return []
        out = []
        for p in self.root_dir.glob("symbol=*"):
            if p.is_dir():
                out.append(p.name.split("symbol=", 1)[1])
        return sorted(out)

    def list_dates(self, symbol: str) -> list[date]:
        sdir = self._symbol_dir(symbol)
        if not sdir.exists():
            return []
        dates: list[date] = []
        for p in sdir.glob("date=*"):
            if not p.is_dir():
                continue
            d_str = p.name.split("date=", 1)[1]
            try:
                dates.append(datetime.strptime(d_str, "%Y-%m-%d").date())
            except ValueError:
                continue
        return sorted(dates)

    def _iter_partitions(
        self,
        symbol: str,
        start: Optional[date],
        end: Optional[date],
    ) -> Iterable[Path]:
        """
        Yields bars.parquet paths for symbol within [start, end] date bounds (UTC date partitions).
        """
        sdir = self._symbol_dir(symbol)
        if not sdir.exists():
            return []

        candidates = []
        for ddir in sdir.glob("date=*"):
            if not ddir.is_dir():
                continue
            d_str = ddir.name.split("date=", 1)[1]
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            f = ddir / "bars.parquet"
            if f.exists():
                candidates.append(f)

        return sorted(candidates)

    @staticmethod
    def _to_timestamp_utc(s: pd.Series) -> pd.Series:
        """
        Your parquet uses `date` as epoch milliseconds (int).
        We also handle:
          - epoch seconds
          - ISO strings
          - already-datetime
        """
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, utc=True, errors="coerce")

        # Numeric epoch?
        if pd.api.types.is_numeric_dtype(s):
            # Heuristic: ms epochs are ~1e12+, seconds are ~1e9+
            v = pd.to_numeric(s, errors="coerce")
            med = v.dropna().median() if v.notna().any() else None
            if med is None:
                return pd.to_datetime(v, utc=True, errors="coerce")
            if med > 1e11:
                return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
            if med > 1e8:
                return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
            # Fallback
            return pd.to_datetime(v, utc=True, errors="coerce")
        # String / mixed
        return pd.to_datetime(s, utc=True, errors="coerce")

    @staticmethod
    def _detect_timestamp_col(df: pd.DataFrame) -> str:
        for c in _TS_CANDIDATES:
            if c in df.columns:
                return c
        # fallback: first datetime-like column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        raise ValueError(f"Could not detect timestamp column. Columns={list(df.columns)}")

    @staticmethod
    def _normalize_schema(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Ensures:
          - timestamp_utc is tz-aware UTC
          - columns: timestamp_utc, open, high, low, close, volume, symbol
        Accepts common variants (Open/open, etc).
        """
        df = df.copy()

        # Normalize OHLCV names
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("open",):
                rename_map[c] = "open"
            elif lc in ("high",):
                rename_map[c] = "high"
            elif lc in ("low",):
                rename_map[c] = "low"
            elif lc in ("close", "last"):
                rename_map[c] = "close"
            elif lc in ("volume", "vol"):
                rename_map[c] = "volume"
        if rename_map:
            df = df.rename(columns=rename_map)

        ts_col = BarsStore._detect_timestamp_col(df)
        if ts_col != "timestamp_utc":
            df = df.rename(columns={ts_col: "timestamp_utc"})


        # timestamp -> tz-aware UTC (supports epoch ms in your `date` column)
        df["timestamp_utc"] = BarsStore._to_timestamp_utc(df["timestamp_utc"])
        df = df.dropna(subset=["timestamp_utc"])

        # Ensure symbol exists
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        else:
            df["symbol"] = df["symbol"].astype(str)

        keep = ["timestamp_utc", "symbol", "open", "high", "low", "close", "volume"]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing}. Columns={list(df.columns)}")

        df = df[keep]

        # Dedupe/sort
        df = df.sort_values(["symbol", "timestamp_utc"])
        df = df.drop_duplicates(subset=["symbol", "timestamp_utc"], keep="last")

        # Basic dtype normalization
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        return df

    def load_bars(
        self,
        symbol: str,
        start: Optional[str | datetime] = None,
        end: Optional[str | datetime] = None,
    ) -> pd.DataFrame:
        """
        Load a single symbol into long format:
          timestamp_utc, symbol, open, high, low, close, volume
        start/end can be ISO strings or datetimes (interpreted in UTC).
        """
        start_dt = pd.to_datetime(start, utc=True) if start is not None else None
        end_dt = pd.to_datetime(end, utc=True) if end is not None else None

        # Partition day bounds (UTC)
        start_d = start_dt.date() if start_dt is not None else None
        end_d = end_dt.date() if end_dt is not None else None

        parts = list(self._iter_partitions(symbol, start_d, end_d))
        if not parts:
            return pd.DataFrame(columns=["timestamp_utc", "symbol", "open", "high", "low", "close", "volume"])

        frames: list[pd.DataFrame] = []
        for p in parts:
            try:
                df = pd.read_parquet(p)
                df = self._normalize_schema(df, symbol=symbol)
                frames.append(df)
            except Exception as e:
                logger.exception("Failed reading %s: %s", p, e)
                continue

        if not frames:
            return pd.DataFrame(columns=["timestamp_utc", "symbol", "open", "high", "low", "close", "volume"])

        out = pd.concat(frames, ignore_index=True)

        if start_dt is not None:
            out = out[out["timestamp_utc"] >= start_dt]
        if end_dt is not None:
            out = out[out["timestamp_utc"] <= end_dt]

        out = out.sort_values(["symbol", "timestamp_utc"]).drop_duplicates(["symbol", "timestamp_utc"], keep="last")
        return out.reset_index(drop=True)

    def load_panel(
        self,
        symbols: list[str],
        start: Optional[str | datetime] = None,
        end: Optional[str | datetime] = None,
    ) -> pd.DataFrame:
        """
        Load multi-symbol long panel:
          timestamp_utc, symbol, open, high, low, close, volume
        """
        frames = []
        for s in symbols:
            frames.append(self.load_bars(s, start=start, end=end))
        if not frames:
            return pd.DataFrame(columns=["timestamp_utc", "symbol", "open", "high", "low", "close", "volume"])
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["timestamp_utc", "symbol"]).drop_duplicates(["timestamp_utc", "symbol"], keep="last")
        return out.reset_index(drop=True)
