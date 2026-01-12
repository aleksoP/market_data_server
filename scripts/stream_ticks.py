from ib_insync import IB, Stock
import time

HOST = "127.0.0.1"
PORT = 4002
CLIENT_ID = 8

def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=5)

    # Example: Apple stock (SMART routing, USD)
    contract = Stock("AAPL", "SMART", "USD")
    ticker = ib.reqMktData(contract, "", False, False)

    print("Streaming... (Ctrl+C to stop)")
    try:
        while True:
            ib.sleep(1)
            # ticker.last can be None if market closed; use close/marketPrice() too
            print({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last": ticker.last,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "close": ticker.close,
                "mktPrice": ticker.marketPrice(),
            })
    except KeyboardInterrupt:
        pass
    finally:
        ib.cancelMktData(contract)
        ib.disconnect()

if __name__ == "__main__":
    main()
