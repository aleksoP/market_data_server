from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.bars_store import BarsStore
from src.features.pipeline import FeatureConfig, build_feature_partitions
from src.utils.universe import load_symbols

logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--bars-root", default="data/bars_1m")
    ap.add_argument("--out-root", default="data/features_1m")
    args = ap.parse_args()

    symbols = load_symbols(Path(args.universe))
    store = BarsStore(root_dir=Path(args.bars_root))
    cfg = FeatureConfig()

    build_feature_partitions(
        store=store,
        symbols=symbols,
        start=args.start,
        end=args.end,
        cfg=cfg,
        out_root=Path(args.out_root),
    )

    print(f"[build_features] wrote partitions under: {args.out_root}")


if __name__ == "__main__":
    main()
