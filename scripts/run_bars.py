from __future__ import annotations
import argparse
import logging

logger = logging.getLogger("run_bars")

from pathlib import Path

import yaml
from src.ibkr.connect import connect_ib
from src.collectors.historical import BarsConfig, fetch_bars_days, store_bars
from src.ibkr.health import wait_for_ushmds_ok

ROOT = Path.home() / "market_data_server"
LOGDIR = ROOT / "logs"
DATADIR = ROOT / "data" / "bars_1m"
LOGDIR.mkdir(parents=True, exist_ok=True)
DATADIR.mkdir(parents=True, exist_ok=True)

def setup_logging() -> None:
    log_file = LOGDIR / "bars.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "universe.yaml"))
    ap.add_argument("--client-id", type=int, default=20)
    ap.add_argument("--mode", choices=["backfill", "update"], default="update")
    ap.add_argument("--backfill-days", type=int, default=30)
    ap.add_argument("--update-days", type=int, default=2)
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("bars")

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    host = cfg["ibkr"]["host"]
    port = int(cfg["ibkr"]["port"])

    bars_cfg = BarsConfig(
        bar_size=cfg["bars"]["bar_size"],
        what_to_show=cfg["bars"]["what_to_show"],
        use_rth=bool(cfg["bars"]["use_rth"]),
    )

    universe = cfg["universe"]
    days = args.backfill_days if args.mode == "backfill" else args.update_days

    log.info("Starting mode=%s days=%s universe=%d host=%s port=%s",
             args.mode, days, len(universe), host, port)

    ib = connect_ib(host, port, client_id=args.client_id, timeout=10, retries=5)

    log.info("Waiting for HMDS readiness (timeout=120s)...")
    ok = wait_for_ushmds_ok(ib, timeout_s=120)
    log.info("HMDS readiness result: %s", ok)

    if not ok:
        logger.warning("HMDS readiness not confirmed after 120s; proceeding (some historical requests may timeout)")


    try:
        for item in universe:
            symbol = item["symbol"]
            exchange = item.get("exchange", "SMART")
            currency = item.get("currency", "USD")

            log.info("Fetching %s (%s/%s) %s", symbol, exchange, currency, bars_cfg)
            df = fetch_bars_days(ib, symbol, exchange, currency, days, bars_cfg)

            if df.empty:
                log.warning("No bars returned for %s", symbol)
                continue
            df = df.sort_values(["date"]).drop_duplicates(subset=["symbol", "date"], keep="last")
            written = store_bars(df, DATADIR, symbol)
            log.info("Stored %s rows for %s into %d partitions", len(df), symbol, len(written))
    finally:
        ib.disconnect()

    log.info("Done.")

if __name__ == "__main__":
    main()
