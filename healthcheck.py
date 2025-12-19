from pathlib import Path
from datetime import datetime, timezone

p = Path("/data/logs")
p.mkdir(parents=True, exist_ok=True)

stamp = datetime.now(timezone.utc).isoformat()
msg = f"[healthcheck] {stamp} UTC - OK\n"

out = p / "healthcheck.log"
out.write_text(out.read_text() + msg if out.exists() else msg)

print(msg.strip())
