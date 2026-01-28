from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.features_store import FeaturesStore
from src.data.labels_store import LabelsStore


@dataclass(frozen=True)
class DatasetMetadata:
    built_at_utc: str
    start: str
    end: str
    symbols: list[str]
    n_rows: int
    n_features: int
    n_labels: int
    feature_cols: list[str]
    label_cols: list[str]


def build_dataset_window(
    symbols: list[str],
    start: str,
    end: str,
    out_dir: Path = Path("data/datasets"),
    features_root: Path = Path("data/features_1m"),
    labels_root: Path = Path("data/labels_1m"),
    name: str | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = FeaturesStore(root_dir=features_root)
    ls = LabelsStore(root_dir=labels_root)

    X = fs.load_panel(symbols, start, end)
    y = ls.load_panel(symbols, start, end)

    if X.empty:
        raise RuntimeError("No features found for requested range.")
    if y.empty:
        raise RuntimeError("No labels found for requested range.")

    # Merge on keys
    df = X.merge(y, on=["timestamp_utc", "symbol"], how="inner")

    # Identify columns
    key_cols = ["timestamp_utc", "symbol"]
    label_cols = [c for c in y.columns if c not in key_cols]
    feature_cols = [c for c in X.columns if c not in key_cols]

    # Drop rows where any label is missing
    df = df.dropna(subset=label_cols)

    tag = name or f"ds_{start}_to_{end}"
    out_parquet = out_dir / f"{tag}.parquet"
    out_meta = out_dir / f"{tag}.metadata.json"

    df.to_parquet(out_parquet, index=False)

    meta = DatasetMetadata(
        built_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        start=start,
        end=end,
        symbols=symbols,
        n_rows=len(df),
        n_features=len(feature_cols),
        n_labels=len(label_cols),
        feature_cols=feature_cols,
        label_cols=label_cols,
    )

    out_meta.write_text(json.dumps(asdict(meta), indent=2))
    return out_parquet, out_meta
