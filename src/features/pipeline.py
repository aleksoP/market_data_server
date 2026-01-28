from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.utils.io import atomic_write_parquet
from src.features.technical import add_technical_features
from src.features.microstructure import add_microstructure_features
from src.features.return_matrix import add_lagged_returns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    vol_windows: tuple[int, ...] = (30, 60, 390)
    atr_window: int = 14
    return_lags: tuple[int, ...] = (1, 5, 15, 30, 60)

    @property
    def lookback_bars(self) -> int:
        return max([self.atr_window, *self.vol_windows, *self.return_lags])


def _compute_features_one_symbol(panel_sym: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = panel_sym.sort_values("timestamp_utc").copy()

    # Technical + microstructure
    g = add_technical_features(g, vol_windows=list(cfg.vol_windows), atr_window=cfg.atr_window)
    g = add_microstructure_features(g)
    g = add_lagged_returns(g, lags=list(cfg.return_lags))

    return g


def build_feature_partitions(
    store,
    symbols: list[str],
    start: str,
    end: str,
    cfg: FeatureConfig,
    out_root: Path = Path("data/features_1m"),
    skip_existing: bool = True,
    force_days: set[str] | None = None,
) -> None:
    """
    Writes:
      data/features_1m/symbol=XYZ/date=YYYY-MM-DD/features.parquet

    Uses lookback to compute rolling features correctly at day boundaries.
    """
    days = pd.date_range(start=start, end=end, freq="D")
    lookback = cfg.lookback_bars
    force_days = force_days or set()

    for sym in symbols:
        logger.info("Features: symbol=%s", sym)

        for d in days:
            day = d.date().isoformat()

            out_path = out_root / f"symbol={sym}" / f"date={day}" / "features.parquet"
            if skip_existing and out_path.exists() and day not in force_days:
                continue

            day_start = pd.Timestamp(day, tz="UTC")
            day_end = day_start + pd.Timedelta(days=1)

            load_start = (day_start - pd.Timedelta(minutes=lookback)).isoformat()
            load_end = (day_end).isoformat()

            bars = store.load_bars(sym, start=load_start, end=load_end)
            if bars.empty:
                continue

            feats = _compute_features_one_symbol(bars, cfg)

            mask = (feats["timestamp_utc"] >= day_start) & (feats["timestamp_utc"] < day_end)
            out = feats.loc[mask].copy()
            if out.empty:
                continue

            atomic_write_parquet(out, out_path)

    logger.info("Features partitions complete: %s", out_root)
