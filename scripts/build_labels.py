from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.bars_store import BarsStore
from src.labeling.forward_returns import build_forward_returns
from src.labeling.triple_barrier import triple_barrier_labels

logging.basicConfig(level=logging.INFO)


def _load_universe_symbols(path: Path) -> list[str]:
    obj = yaml.safe_load(path.read_text())

    # Your schema: {bars:..., ibkr:..., universe:[{symbol:...}, ...]}
    if isinstance(obj, dict) and "universe" in obj and isinstance(obj["universe"], list):
        syms = []
        for item in obj["universe"]:
            if isinstance(item, dict) and "symbol" in item:
                syms.append(str(item["symbol"]))
        if syms:
            return syms

    # fallback: {symbols:[...]}
    if isinstance(obj, dict) and "symbols" in obj and isinstance(obj["symbols"], list):
        return [str(s) for s in obj["symbols"]]

    # fallback: ["AAPL","MSFT"]
    if isinstance(obj, list) and all(isinstance(x, (str, int)) for x in obj):
        return [str(x) for x in obj]

    # fallback: {AAPL:{...}, MSFT:{...}}
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
        return sorted([str(k) for k in obj.keys()])

    raise ValueError(f"Unsupported universe schema in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/bars_1m")
    ap.add_argument("--universe", default="config/universe.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--features", default="data/features/feature_matrix.parquet")
    ap.add_argument("--out", default="data/labels")
    ap.add_argument("--fwd", default="30,60,390")
    ap.add_argument("--tb_horizon", type=int, default=60)
    ap.add_argument("--tb_vol_col", default="vol_logret_60")
    args = ap.parse_args()

    print("[build_labels] starting")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = _load_universe_symbols(Path(args.universe))
    print(f"[build_labels] symbols={symbols}")

    store = BarsStore(root_dir=Path(args.root))
    bars = store.load_panel(symbols, start=args.start, end=args.end)
    print(f"[build_labels] bars rows={len(bars)}")

    if bars.empty:
        raise RuntimeError("No bars loaded; cannot build labels.")

    # Forward returns
    horizons = [int(x) for x in args.fwd.split(",") if x.strip()]
    y_fwd = build_forward_returns(bars, horizons=horizons)
    print(f"[build_labels] fwd labels shape={y_fwd.shape}")

    # Triple barrier requires vol from features
    feat_path = Path(args.features)
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features at {feat_path}. Run build_features first.")

    X = pd.read_parquet(feat_path).reset_index()
    if args.tb_vol_col not in X.columns:
        raise ValueError(
            f"Vol column '{args.tb_vol_col}' not found in feature matrix. "
            f"Available cols include: {sorted([c for c in X.columns if 'vol' in c])}"
        )

    tmp = bars.merge(
        X[["timestamp_utc", "symbol", args.tb_vol_col]],
        on=["timestamp_utc", "symbol"],
        how="left",
    )

    y_tb = triple_barrier_labels(
        tmp,
        vol_col=args.tb_vol_col,
        max_horizon=args.tb_horizon,
        pt_mult=1.0,
        sl_mult=1.0,
    )
    print(f"[build_labels] tb labels shape={y_tb.shape}")

    # Combine + write
    y = y_fwd.join(y_tb, how="outer").sort_index()
    out_path = out_dir / "labels.parquet"
    y.to_parquet(out_path)

    print(f"[build_labels] wrote: {out_path} rows={y.shape[0]} cols={y.shape[1]}")


if __name__ == "__main__":
    main()
