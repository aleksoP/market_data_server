from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    feature_path: str
    labels_path: str
    built_at_utc: str
    rows: int
    cols: list[str]
    label_cols: list[str]
    feature_cols: list[str]
    hashes: dict[str, str]
    coverage: dict[str, float]


def build_dataset(
    feature_matrix_path: Path,
    labels_path: Path,
    out_dir: Path = Path("data/datasets"),
    dataset_name: Optional[str] = None,
    dropna: bool = True,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(feature_matrix_path)
    y = pd.read_parquet(labels_path)

    # Both should be indexed by [timestamp_utc, symbol]
    if not isinstance(X.index, pd.MultiIndex) or X.index.names != ["timestamp_utc", "symbol"]:
        raise ValueError("Feature matrix must be indexed by [timestamp_utc, symbol].")
    if not isinstance(y.index, pd.MultiIndex) or y.index.names != ["timestamp_utc", "symbol"]:
        raise ValueError("Labels must be indexed by [timestamp_utc, symbol].")

    ds = X.join(y, how="inner")

    # Optional: remove rows without labels / key features
    if dropna:
        ds = ds.dropna(subset=list(y.columns))

    # Conservative: drop rows where core price-derived features are NaN
    core_cols = [c for c in ["ret_1", "logret_1"] if c in ds.columns]
    if dropna and core_cols:
        ds = ds.dropna(subset=core_cols)

    ds = ds.sort_index()

    built_at = datetime.now(timezone.utc).isoformat()
    if dataset_name is None:
        dataset_name = f"ds_{feature_matrix_path.stem}__{labels_path.stem}__{built_at[:10]}"

    out_parquet = out_dir / f"{dataset_name}.parquet"
    ds.to_parquet(out_parquet)

    feature_cols = [c for c in ds.columns if c not in y.columns]
    label_cols = list(y.columns)

    coverage = {
        "rows_total": float(X.shape[0]),
        "rows_with_labels": float(ds.shape[0]),
        "label_coverage": float(ds.shape[0] / max(1, X.shape[0])),
    }

    spec = DatasetSpec(
        name=dataset_name,
        feature_path=str(feature_matrix_path),
        labels_path=str(labels_path),
        built_at_utc=built_at,
        rows=int(ds.shape[0]),
        cols=list(ds.columns),
        label_cols=label_cols,
        feature_cols=feature_cols,
        hashes={
            "feature_path_sha1": _sha1_text(str(feature_matrix_path)),
            "labels_path_sha1": _sha1_text(str(labels_path)),
        },
        coverage=coverage,
    )

    out_meta = out_dir / f"{dataset_name}.metadata.json"
    out_meta.write_text(json.dumps(asdict(spec), indent=2))

    logger.info("Wrote dataset: %s (rows=%d cols=%d)", out_parquet, ds.shape[0], ds.shape[1])
    return out_parquet, out_meta
