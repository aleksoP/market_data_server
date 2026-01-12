from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_daily_partitioned(
    df: pd.DataFrame,
    root_dir: Path,
    symbol: str,
    ts_col: str = "date",
    partition_tz: str = "UTC",  # e.g. "America/New_York" for US trading-day-ish partitioning
) -> list[Path]:
    """
    Writes df into Parquet partitions:
      root_dir / symbol=XYZ / date=YYYY-MM-DD / bars.parquet

    If file exists, merges + de-dupes by timestamp.
    Uses atomic write (temp file + os.replace) to avoid partial parquet corruption.
    Returns list of paths written.
    """
    if df.empty:
        return []

    if ts_col not in df.columns:
        raise ValueError(f"DataFrame missing '{ts_col}' column")

    # Normalize timestamp (new data)
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])

    # Partition date in desired timezone (default UTC)
    try:
        part_dt = df[ts_col].dt.tz_convert(partition_tz) if partition_tz else df[ts_col]
    except Exception:
        part_dt = df[ts_col]

    df["__date"] = part_dt.dt.strftime("%Y-%m-%d")
    written: list[Path] = []

    for d, g in df.groupby("__date", sort=True):
        part_dir = root_dir / f"symbol={symbol}" / f"date={d}"
        _ensure_dir(part_dir)

        out = part_dir / "bars.parquet"
        tmp = part_dir / "bars.parquet.tmp"

        new_chunk = g.drop(columns=["__date"])

        if out.exists():
            old = pd.read_parquet(out)

            # Normalize timestamp (existing data)
            if ts_col in old.columns:
                old[ts_col] = pd.to_datetime(old[ts_col], utc=True, errors="coerce")
                old = old.dropna(subset=[ts_col])

            merged = pd.concat([old, new_chunk], ignore_index=True, sort=False)
        else:
            merged = new_chunk

        # Normalize + de-dupe by timestamp; keep last
        merged[ts_col] = pd.to_datetime(merged[ts_col], utc=True, errors="coerce")
        merged = merged.dropna(subset=[ts_col])
        merged = merged.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last")

        # Atomic write
        if tmp.exists():
            tmp.unlink()
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, out)

        written.append(out)

    return written
