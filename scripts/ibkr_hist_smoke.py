import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.ibkr.connect import connect_ib

log = logging.getLogger("ibkr_hist_smoke")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=91)
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--exchange", default="SMART")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--bar-size", default="1 min")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--use-rth", action="store_true")
    args = ap.parse_args()

    ib = connect_ib(args.host, args.port, client_id=args.client_id)

    # Build a minimal IB-insync style request using whatever connect_ib returns.
    # We intentionally keep duration small to avoid pacing/timeouts.
    end = datetime.now(timezone.utc)
    # IBKR durationStr uses S/D/W/M/Y where M = MONTHS (not minutes).
    duration_str = f"{args.minutes * 60} S"   # minutes -> seconds

    log.info("Requesting %s %s bars for last %s (duration=%s useRTH=%s)",
             args.symbol, args.bar_size, args.minutes, duration_str, args.use_rth)

    # NOTE: This assumes your connect_ib returns an ib_insync.IB instance.
    from ib_insync import Stock
    contract = Stock(args.symbol, args.exchange, args.currency)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end,
        durationStr=duration_str,
        barSizeSetting=args.bar_size,
        whatToShow="TRADES",
        useRTH=1 if args.use_rth else 0,
        formatDate=1
    )

    df = pd.DataFrame([b.__dict__ for b in bars]) if bars else pd.DataFrame()
    log.info("Bars received: %d", len(df))
    if not df.empty:
        log.info("First: %s  Last: %s", df.iloc[0].get("date"), df.iloc[-1].get("date"))

    ib.disconnect()

if __name__ == "__main__":
    main()
