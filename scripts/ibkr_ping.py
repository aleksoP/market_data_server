import yaml
from pathlib import Path
from ib_insync import IB

cfg = yaml.safe_load(Path(__file__).resolve().parents[1].joinpath("config/ibkr.yaml").read_text())
c = cfg["ibkr"]

ib = IB()
print(f"Connecting to IBKR {c['host']}:{c['port']} clientId={c['client_id']} ...")
ib.connect(c["host"], int(c["port"]), clientId=int(c["client_id"]), readonly=bool(c.get("readonly", True)))

print("Connected.")
print("Server time:", ib.reqCurrentTime())
print("Managed accounts:", ib.managedAccounts())

ib.disconnect()
print("Disconnected.")
