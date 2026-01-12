from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.alignment import ensure_regular_index
from src.data.bars_store import BarsStore
from src.features.microstructure import add_liquidity_proxies
from src.features.return_matrix import add_lagged_returns
from src.features.technical import add_basic_returns, add_rolling_vol, add_true_range_atr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    freq: str = "1min"
    vol_windows: tuple[int, ...] = (30, 60, 390)  # 30m, 1h, ~1d (US RTH) proxy
    atr_window: int = 14
    return_lags: tuple[int, ...] = (2, 5, 10, 30, 60)


def build_feature_matrix(
    store: BarsStore,
    symbols: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    cfg: FeatureConfig = FeatureConfig(),
    out_dir: Path = Path("data/features"),
    name: str = "feature_matrix",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = store.load_panel(symbols, start=start, end=end)
    if panel.empty:
        raise RuntimeError("No bars loaded; cannot build features.")

    # Insert missing timestamps per symbol without ffill
    panel = ensure_regular_index(panel, freq=cfg.freq)

    def per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = add_basic_returns(g)
        g = add_rolling_vol(g, list(cfg.vol_windows))
        g = add_true_range_atr(g, atr_window=cfg.atr_window)
        g = add_liquidity_proxies(g)
        g = add_lagged_returns(g, list(cfg.return_lags))
        return g

    parts = []
    for sym, g in panel.groupby("symbol", sort=True):
        parts.append(per_symbol(g))
    feats = pd.concat(parts, ignore_index=True)

    # No leakage: everything is computed with backward-looking ops only.
    feats = feats.sort_values(["timestamp_utc", "symbol"])

    # Stable index: MultiIndex [timestamp_utc, symbol]
    feats = feats.set_index(["timestamp_utc", "symbol"]).sort_index()

    out_path = out_dir / f"{name}.parquet"
    feats.to_parquet(out_path)

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "start": start,
        "end": end,
        "feature_config": asdict(cfg),
        "rows": int(feats.shape[0]),
        "cols": list(feats.columns),
    }
    (out_dir / f"{name}.metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info("Wrote feature matrix: %s (rows=%d cols=%d)", out_path, feats.shape[0], feats.shape[1])
    return out_path
