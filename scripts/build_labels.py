from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.bars_store import BarsStore
from src.labeling.pipeline import LabelConfig, build_label_partitions
from src.utils.universe import load_symbols

logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--bars-root", default="data/bars_1m")
    ap.add_argument("--out-root", default="data/labels_1m")
    args = ap.parse_args()

    symbols = load_symbols(Path(args.universe))
    store = BarsStore(root_dir=Path(args.bars_root))
    cfg = LabelConfig()

    build_label_partitions(
        store=store,
        symbols=symbols,
        start=args.start,
        end=args.end,
        cfg=cfg,
        out_root=Path(args.out_root),
    )

    print(f"[build_labels] wrote partitions under: {args.out_root}")


if __name__ == "__main__":
    main()
