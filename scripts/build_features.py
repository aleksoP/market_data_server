from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.data.bars_store import BarsStore
from src.features.pipeline import FeatureConfig, build_feature_matrix

logging.basicConfig(level=logging.INFO)


def _load_universe(path: Path) -> list[str]:
    obj = yaml.safe_load(path.read_text())

    # 1) Simple forms
    if isinstance(obj, dict) and "symbols" in obj and isinstance(obj["symbols"], list):
        return [str(s) for s in obj["symbols"]]
    if isinstance(obj, list) and all(isinstance(x, (str, int)) for x in obj):
        return [str(x) for x in obj]

    # 2) Common rich forms:
    #    a) {AAPL: {...}, MSFT: {...}}
    if isinstance(obj, dict) and obj and all(isinstance(k, str) for k in obj.keys()):
        # If values are dicts (metadata), keys are symbols
        if all(isinstance(v, dict) for v in obj.values()):
            return sorted([str(k) for k in obj.keys()])

    #    b) {universe: [{symbol: AAPL, ...}, {symbol: MSFT, ...}]}
    if isinstance(obj, dict) and "universe" in obj and isinstance(obj["universe"], list):
        syms = []
        for item in obj["universe"]:
            if isinstance(item, dict) and "symbol" in item:
                syms.append(str(item["symbol"]))
        if syms:
            return syms

    #    c) [{symbol: AAPL, ...}, {symbol: MSFT, ...}]
    if isinstance(obj, list):
        syms = []
        for item in obj:
            if isinstance(item, dict) and "symbol" in item:
                syms.append(str(item["symbol"]))
        if syms:
            return syms

    raise ValueError(f"Unsupported universe schema in {path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/bars_1m")
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/features")
    args = ap.parse_args()

    print("[build_features] starting")  # proof-of-life

    symbols = _load_universe(Path(args.universe))
    store = BarsStore(root_dir=Path(args.root))
    cfg = FeatureConfig()

    out_path = build_feature_matrix(
        store=store,
        symbols=symbols,
        start=args.start,
        end=args.end,
        cfg=cfg,
        out_dir=Path(args.out),
        name="feature_matrix",
    )

    print(f"[build_features] wrote: {out_path}")


if __name__ == "__main__":
    main()
