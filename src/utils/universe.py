from __future__ import annotations

from pathlib import Path
import yaml


def load_symbols(universe_path: Path) -> list[str]:
    obj = yaml.safe_load(universe_path.read_text())

    # Your schema: {bars:..., ibkr:..., universe:[{symbol:...}, ...]}
    if isinstance(obj, dict) and "universe" in obj and isinstance(obj["universe"], list):
        syms = []
        for item in obj["universe"]:
            if isinstance(item, dict) and "symbol" in item:
                syms.append(str(item["symbol"]))
        if syms:
            return syms

    # Fallbacks
    if isinstance(obj, dict) and "symbols" in obj and isinstance(obj["symbols"], list):
        return [str(s) for s in obj["symbols"]]
    if isinstance(obj, list) and all(isinstance(x, (str, int)) for x in obj):
        return [str(x) for x in obj]
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
        return sorted([str(k) for k in obj.keys()])

    raise ValueError(f"Unsupported universe schema in {universe_path}")
