from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from src.data.bars_store import BarsStore
from src.features.pipeline import FeatureConfig, build_feature_partitions
from src.labeling.pipeline import LabelConfig, build_label_partitions
from src.pipelines.build_dataset_window import build_dataset_window
from src.utils.universe import load_symbols


def _date_str(d: pd.Timestamp) -> str:
    return d.date().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--name", default=None, help="dataset name tag")
    ap.add_argument("--universe", default="config/universe.yaml")

    ap.add_argument("--bars-root", default="data/bars_1m")
    ap.add_argument("--features-root", default="data/features_1m")
    ap.add_argument("--labels-root", default="data/labels_1m")
    ap.add_argument("--out-dir", default="data/datasets")

    ap.add_argument(
        "--refresh-days",
        type=int,
        default=7,
        help="Force rebuild last N days within requested window (default: 7)",
    )

    # Execution modes
    ap.add_argument("--no-features", action="store_true", help="Do not build features partitions")
    ap.add_argument("--no-labels", action="store_true", help="Do not build labels partitions")
    ap.add_argument("--no-dataset", action="store_true", help="Do not build dataset window artifact")

    args = ap.parse_args()

    # Basic validation
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    if start_ts > end_ts:
        raise ValueError(f"--start must be <= --end (got start={args.start} end={args.end})")

    symbols = load_symbols(Path(args.universe))

    features_root = Path(args.features_root)
    labels_root = Path(args.labels_root)

    # Build list of days in requested window (inclusive)
    days = pd.date_range(start=args.start, end=args.end, freq="D")
    day_strs = [_date_str(d) for d in days]
    if not day_strs:
        raise RuntimeError("No days generated for the given start/end range.")

    # Force rebuild the last N days (within the requested range)
    n = max(0, int(args.refresh_days))
    n = min(n, len(day_strs))
    force_days = set(day_strs[-n:]) if n > 0 else set()

    # Detect missing partitions (for reporting)
    missing_feat = 0
    missing_lab = 0
    for sym in symbols:
        for day in day_strs:
            fp = features_root / f"symbol={sym}" / f"date={day}" / "features.parquet"
            lp = labels_root / f"symbol={sym}" / f"date={day}" / "labels.parquet"
            if (not fp.exists()) or (day in force_days):
                missing_feat += 1
            if (not lp.exists()) or (day in force_days):
                missing_lab += 1

    print(f"[make_dataset] symbols={len(symbols)} days={len(day_strs)}")
    print(f"[make_dataset] features to (re)build: {missing_feat} (root={features_root})")
    print(f"[make_dataset] labels   to (re)build: {missing_lab} (root={labels_root})")
    if force_days:
        s = sorted(force_days)
        print(f"[make_dataset] forcing rebuild for days: {s[0]} .. {s[-1]} (count={len(s)})")

    bars_store = BarsStore(root_dir=Path(args.bars_root))

    # Build missing/forced feature partitions
    if not args.no_features:
        fcfg = FeatureConfig()
        build_feature_partitions(
            store=bars_store,
            symbols=symbols,
            start=args.start,
            end=args.end,
            cfg=fcfg,
            out_root=features_root,
            skip_existing=True,
            force_days=force_days,
        )
    else:
        print("[make_dataset] skipping features build (--no-features)")

    # Build missing/forced label partitions
    if not args.no_labels:
        lcfg = LabelConfig()
        build_label_partitions(
            store=bars_store,
            symbols=symbols,
            start=args.start,
            end=args.end,
            cfg=lcfg,
            out_root=labels_root,
            skip_existing=True,
            force_days=force_days,
        )
    else:
        print("[make_dataset] skipping labels build (--no-labels)")

    # Build dataset window artifact (bounded training file)
    if args.no_dataset:
        print("[make_dataset] skipping dataset build (--no-dataset)")
        return

    tag = args.name
    if tag is None:
        tag = f"ds_{args.start}_to_{args.end}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    out_pq, out_meta = build_dataset_window(
        symbols=symbols,
        start=args.start,
        end=args.end,
        out_dir=Path(args.out_dir),
        features_root=features_root,
        labels_root=labels_root,
        name=tag,
    )

    print("[make_dataset] dataset:", out_pq)
    print("[make_dataset] meta   :", out_meta)


if __name__ == "__main__":
    main()
