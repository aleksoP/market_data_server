from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    bars_long: pd.DataFrame,
    vol_col: str,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    max_horizon: int = 60,  # minutes (for 1m bars)
    min_vol: float = 1e-8,
) -> pd.DataFrame:
    """
    Minimal triple barrier labeler.

    For each (t, symbol):
      - reference price: close[t]
      - upper barrier: close[t] * (1 + pt_mult * vol[t])
      - lower barrier: close[t] * (1 - sl_mult * vol[t])
      - scan forward up to max_horizon bars:
          hit upper first => label=+1
          hit lower first => label=-1
          else timeout => label=0
    Output index: [timestamp_utc, symbol]
      columns: tb_label, tb_touched, tb_ret
    """
    df = bars_long.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values(["symbol", "timestamp_utc"]).drop_duplicates(["symbol", "timestamp_utc"], keep="last")

    if vol_col not in df.columns:
        raise ValueError(f"vol_col '{vol_col}' not present in bars_long columns")

    res = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("timestamp_utc").reset_index(drop=True)

        close = g["close"].astype(float).to_numpy()
        vol = g[vol_col].astype(float).to_numpy()
        ts = g["timestamp_utc"].to_numpy()

        n = len(g)
        label = np.full(n, np.nan, dtype=float)
        touched = np.full(n, np.nan, dtype=float)  # +1 upper, -1 lower, 0 timeout
        tb_ret = np.full(n, np.nan, dtype=float)

        for i in range(n):
            v = vol[i]
            if not np.isfinite(v) or v < min_vol:
                continue
            p0 = close[i]
            if not np.isfinite(p0) or p0 <= 0:
                continue

            up = p0 * (1.0 + pt_mult * v)
            dn = p0 * (1.0 - sl_mult * v)

            j_end = min(n - 1, i + max_horizon)
            hit = 0
            hit_j = j_end

            for j in range(i + 1, j_end + 1):
                pj = close[j]
                if not np.isfinite(pj):
                    continue
                if pj >= up:
                    hit = +1
                    hit_j = j
                    break
                if pj <= dn:
                    hit = -1
                    hit_j = j
                    break

            touched[i] = float(hit)
            label[i] = 1.0 if hit == +1 else (-1.0 if hit == -1 else 0.0)
            tb_ret[i] = close[hit_j] / p0 - 1.0

        out = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(ts, utc=True),
                "symbol": sym,
                "tb_label": label,
                "tb_touched": touched,
                "tb_ret": tb_ret,
            }
        )
        res.append(out)

    out = pd.concat(res, ignore_index=True)
    return out.set_index(["timestamp_utc", "symbol"]).sort_index()
