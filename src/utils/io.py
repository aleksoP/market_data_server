from __future__ import annotations

from pathlib import Path
import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)  # store keys as COLUMNS (scales better)
    tmp.replace(out_path)
