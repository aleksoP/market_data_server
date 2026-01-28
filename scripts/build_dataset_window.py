from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipelines.build_dataset_window import build_dataset_window
from src.utils.universe import load_symbols

logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--features-root", default="data/features_1m")
    ap.add_argument("--labels-root", default="data/labels_1m")
    ap.add_argument("--out-dir", default="data/datasets")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    symbols = load_symbols(Path(args.universe))
    out_pq, out_js = build_dataset_window(
        symbols=symbols,
        start=args.start,
        end=args.end,
        out_dir=Path(args.out_dir),
        features_root=Path(args.features_root),
        labels_root=Path(args.labels_root),
        name=args.name,
    )

    print("[build_dataset_window] wrote:", out_pq)
    print("[build_dataset_window] meta:", out_js)


if __name__ == "__main__":
    main()
