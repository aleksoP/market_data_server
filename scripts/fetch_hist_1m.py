from ib_insync import IB, Stock, util
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

HOST = "127.0.0.1"
PORT = 4002
CLIENT_ID = 9

OUTDIR = Path.home() / "market_data_server" / "data" / "bars_1m"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)

    contract = Stock("AAPL", "SMART", "USD")

    # 1 day of 1-min bars (RTH=False includes pre/after market)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
    )

    df = util.df(bars)
    if df.empty:
        print("No data returned.")
        return

    # normalize + save
    df["symbol"] = "AAPL"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"AAPL_1m_{ts}.parquet"
    df.to_parquet(out, index=False)
    print("Saved:", out)

    ib.disconnect()

if __name__ == "__main__":
    main()
