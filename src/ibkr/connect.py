from __future__ import annotations
import time
from ib_insync import IB

def connect_ib(
    host: str,
    port: int,
    client_id: int,
    timeout: int = 10,
    retries: int = 5,
    retry_sleep_s: float = 2.0,
) -> IB:
    last_err = None
    for i in range(1, retries + 1):
        try:
            ib = IB()
            ib.connect(host, port, clientId=client_id, timeout=timeout)
            return ib
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep_s)

    raise RuntimeError(f"IBKR connect failed after {retries} retries: {last_err}")
