from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import atomic_write_parquet
from src.labeling.forward_returns import build_forward_returns
from src.labeling.triple_barrier import triple_barrier_labels
from src.features.technical import add_technical_features  # reuse for vol feature

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelConfig:
    fwd_horizons: tuple[int, ...] = (30, 60, 390)
    tb_horizon: int = 60
    vol_windows: tuple[int, ...] = (60,)  # what TB uses
    atr_window: int = 14  # not required, but keeps tech calc consistent

    @property
    def lookahead_bars(self) -> int:
        return max([*self.fwd_horizons, self.tb_horizon])

    @property
    def lookback_bars(self) -> int:
        return max([*self.vol_windows, self.atr_window, 390])  # safe default


def build_label_partitions(
    store,
    symbols: list[str],
    start: str,
    end: str,
    cfg: LabelConfig,
    out_root: Path = Path("data/labels_1m"),
    tb_vol_col: str = "vol_logret_60",
    skip_existing: bool = True,
    force_days: set[str] | None = None,
) -> None:
    days = pd.date_range(start=start, end=end, freq="D")
    lb = cfg.lookback_bars
    la = cfg.lookahead_bars
    force_days = force_days or set()

    for sym in symbols:
        logger.info("Labels: symbol=%s", sym)

        for d in days:
            day = d.date().isoformat()

            out_path = out_root / f"symbol={sym}" / f"date={day}" / "labels.parquet"
            if skip_existing and out_path.exists() and day not in force_days:
                continue

            day_start = pd.Timestamp(day, tz="UTC")
            day_end = day_start + pd.Timedelta(days=1)

            load_start = (day_start - pd.Timedelta(minutes=lb)).isoformat()
            load_end = (day_end + pd.Timedelta(minutes=la)).isoformat()

            bars = store.load_bars(sym, start=load_start, end=load_end)
            if bars.empty:
                continue

            bars = add_technical_features(bars, vol_windows=list(cfg.vol_windows), atr_window=cfg.atr_window)

            y_fwd = build_forward_returns(bars, horizons=list(cfg.fwd_horizons)).reset_index()
            y_tb = triple_barrier_labels(
                bars_long=bars,
                vol_col=tb_vol_col,
                max_horizon=cfg.tb_horizon,
                pt_mult=1.0,
                sl_mult=1.0,
            ).reset_index()

            y = y_fwd.merge(y_tb, on=["timestamp_utc", "symbol"], how="outer")

            y["timestamp_utc"] = pd.to_datetime(y["timestamp_utc"], utc=True)
            mask = (y["timestamp_utc"] >= day_start) & (y["timestamp_utc"] < day_end)
            out = y.loc[mask].copy()
            if out.empty:
                continue

            atomic_write_parquet(out, out_path)

    logger.info("Labels partitions complete: %s", out_root)
