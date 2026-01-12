from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import Stock, util

from src.storage.parquet_writer import write_daily_partitioned

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BarsConfig:
    bar_size: str = "1 min"
    what_to_show: str = "TRADES"
    use_rth: bool = False


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _fetch_one_day(
    ib,
    symbol: str,
    exchange: str,
    currency: str,
    end_dt_utc: datetime,
    bars_cfg: BarsConfig,
    max_retries: int = 3,
    backoff_s: float = 3.0,
) -> pd.DataFrame:
    contract = Stock(symbol, exchange, currency)

    # ensure tz-aware UTC
    if end_dt_utc.tzinfo is None:
        end_dt_utc = end_dt_utc.replace(tzinfo=timezone.utc)
    else:
        end_dt_utc = end_dt_utc.astimezone(timezone.utc)

    use_rth = 1 if bars_cfg.use_rth else 0

    # IB UTC notation: yyyymmdd-hh:mm:ss (do NOT append timezone text)
    end_utc_str = end_dt_utc.strftime("%Y%m%d-%H:%M:%S")

    last_reason = None
    for attempt in range(1, max_retries + 1):
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_utc_str,
            durationStr="1 D",
            barSizeSetting=bars_cfg.bar_size,
            whatToShow=bars_cfg.what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )

        if not bars:
            last_reason = "bars is None/empty"
            logger.warning(
                "No bars for %s (end=%s) attempt %d/%d (%s).",
                symbol, end_utc_str, attempt, max_retries, last_reason
            )
            time.sleep(backoff_s * attempt)
            continue

        df = util.df(bars)
        if df is None or df.empty:
            last_reason = "df is None/empty"
            logger.warning(
                "Empty df for %s (end=%s) attempt %d/%d (%s).",
                symbol, end_utc_str, attempt, max_retries, last_reason
            )
            time.sleep(backoff_s * attempt)
            continue

        return df

    logger.warning(
        "Giving up on %s for end=%s after %d retries (%s). Returning empty DF.",
        symbol, end_utc_str, max_retries, last_reason
    )
    return pd.DataFrame()



def fetch_bars_days(
    ib,
    symbol: str,
    exchange: str,
    currency: str,
    days: int,
    bars_cfg: BarsConfig,
) -> pd.DataFrame:
    """
    Fetch N days as N separate 1-day requests (more reliable than '{days} D' in one shot).
    Always returns a DataFrame (possibly empty).
    """
    if days <= 0:
        return pd.DataFrame()

    # Request most-recent day first, then go back day-by-day.
    now_utc = datetime.now(timezone.utc)
    dfs: list[pd.DataFrame] = []

    for i in range(days):
        end_dt = now_utc - pd.Timedelta(days=i)
        df_day = _fetch_one_day(
            ib,
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            end_dt_utc=end_dt,
            bars_cfg=bars_cfg,
        )
        if not df_day.empty:
            dfs.append(df_day)

    if not dfs:
        logger.warning("No data for %s across %d day(s).", symbol, days)
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Standardize columns
    df["symbol"] = symbol
    df["fetched_at_utc"] = _now_utc_str()
    return df


def store_bars(
    df: pd.DataFrame,
    out_root: Path,
    symbol: str,
) -> list[Path]:
    return write_daily_partitioned(df, out_root, symbol=symbol, ts_col="date", partition_tz="America/New_York")


