"""Elige el mejor modelo PROPIO por MASE de VALIDACIÓN (no prueba)."""
import json, glob
from src import config as C
best = None
for f in glob.glob(str(C.RES_DIR / "train_logs" / "*.json")):
    i = json.load(open(f))
    if best is None or i["best_val_MASE"] < best["best_val_MASE"]:
        best = i; best["name"] = f.split("/")[-1][:-5]
kw = dict(L=best["L"], norm=best["norm"], lr=best["lr"], batch_size=best["batch_size"],
          epochs=best["epochs_run"], samples_per_epoch=best["samples_per_epoch"], patience=3, **best["model_kw"])
out = dict(name=best["name"], arch=best["arch"], kw=kw, best_val_MASE=best["best_val_MASE"], best_epoch=int(min(range(len(best["history"])), key=lambda i: best["history"][i]["val_MASE"]) + 1))
json.dump(out, open(C.RES_DIR / "experiments" / "best_own.json", "w"), indent=1)
print(out)
