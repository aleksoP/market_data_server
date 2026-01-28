from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from pathlib import Path

from src.data.bars_store import BarsStore
from src.features.pipeline import FeatureConfig, build_feature_partitions
from src.utils.universe import load_symbols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--bars-root", default="data/bars_1m")
    ap.add_argument("--out-root", default="data/features_1m")
    args = ap.parse_args()

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=args.days)

    symbols = load_symbols(Path(args.universe))
    store = BarsStore(root_dir=Path(args.bars_root))
    cfg = FeatureConfig()

    build_feature_partitions(store, symbols, start.isoformat(), end.isoformat(), cfg, Path(args.out_root))
    print(f"[update_features] updated last {args.days} days")


if __name__ == "__main__":
    main()
