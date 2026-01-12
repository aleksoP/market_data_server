from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from ib_insync import Stock

logger = logging.getLogger(__name__)


def wait_for_ushmds_ok(
    ib,
    timeout_s: int = 120,
    probe_symbol: str = "SPY",
    probe_exchange: str = "SMART",
    probe_currency: str = "USD",
) -> bool:
    """
    Probe-based readiness check for historical data.
    Avoids relying on farm status events (which can be emitted before handlers attach).
    """
    deadline = time.monotonic() + max(0, int(timeout_s))
    contract = Stock(probe_symbol, probe_exchange, probe_currency)

    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            # Very light call that uses historical data infrastructure.
            # If it returns a timestamp, historical backend is responsive.
            ts = ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=0, formatDate=1)
            if ts:
                logger.info("HMDS probe OK (headTimestamp=%s) on attempt %d", ts, attempt)
                return True

            logger.info("HMDS probe returned empty timestamp (attempt %d); retrying...", attempt)

        except Exception as e:
            logger.info("HMDS probe exception (attempt %d): %r", attempt, e)

        time.sleep(2)

    logger.warning("HMDS readiness probe timed out after %ss", timeout_s)
    return False
