"""Persistencia de resultados de backtest (predicciones + metadatos) para la tabla final."""
import json, numpy as np
from . import config as C
BT = C.RES_DIR / "bt"; BT.mkdir(exist_ok=True, parents=True)

def save(res, meta: dict, tag=""):
    key = f"{res.name}{tag}_H{res.H}"
    q = {f"q{int(k*100)}": v for k, v in res.Q.items()}
    np.savez_compressed(BT / f"{key}.npz", P=res.P, origins=res.origins, **q)
    m = res.metrics(); m.update(meta); m["name"] = res.name; m["H"] = res.H; m["tag"] = tag
    json.dump(m, open(BT / f"{key}.json", "w"), indent=1, default=float)
    res.per_step().to_csv(BT / f"{key}_step.csv", index=False)
    res.per_client().to_csv(BT / f"{key}_client.csv", index=False)
    res.per_hour().to_csv(BT / f"{key}_hour.csv", index=False)
    if res.H == 24 and not tag:
        res.to_long().to_parquet(C.PRED_DIR / f"{res.name}.parquet")
    return m
