from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timezone


@dataclass(frozen=True)
class TradingCalendar:
    """
    Minimal calendar helper.

    Current design decision:
    - Your parquet partitions are by UTC date.
    - We do not enforce RTH here (your collector may use use_rth=False).
    """
    partition_tz: str = "UTC"

    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def utc_date(dt: datetime) -> date:
        return TradingCalendar.to_utc(dt).date()
