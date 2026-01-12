from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipelines.build_dataset import build_dataset

logging.basicConfig(level=logging.INFO)


def _latest_parquet(dir_path: Path) -> Path | None:
    if not dir_path.exists():
        return None
    cands = [p for p in dir_path.glob("*.parquet") if p.is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/feature_matrix.parquet")
    ap.add_argument("--labels", default="data/labels/labels.parquet")
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    feat_path = Path(args.features)
    lab_path = Path(args.labels)

    # Auto-discover if defaults are missing
    if not feat_path.exists():
        auto = _latest_parquet(Path("data/features"))
        if auto is None:
            raise FileNotFoundError(
                "No feature parquet found. Expected data/features/*.parquet. "
                "Run: python -m scripts.build_features ..."
            )
        logging.info("Default features missing; using latest: %s", auto)
        feat_path = auto

    if not lab_path.exists():
        auto = _latest_parquet(Path("data/labels"))
        if auto is None:
            raise FileNotFoundError(
                "No labels parquet found. Expected data/labels/*.parquet. "
                "Run: python -m scripts.build_labels ..."
            )
        logging.info("Default labels missing; using latest: %s", auto)
        lab_path = auto

    out_parquet, out_meta = build_dataset(
        feature_matrix_path=feat_path,
        labels_path=lab_path,
        out_dir=Path(args.out),
        dataset_name=args.name,
    )

    logging.info("Dataset written: %s", out_parquet)
    logging.info("Metadata written: %s", out_meta)


if __name__ == "__main__":
    main()
