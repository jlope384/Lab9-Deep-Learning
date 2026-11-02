"""Sección 6: tabla comparativa + gráficas + archivo de predicciones."""
import json, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C

BT, EXP, FIG = C.RES_DIR / "bt", C.RES_DIR / "experiments", C.FIG_DIR
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def load_bt(H=24):
    rows = []
    for f in glob.glob(str(BT / f"*_H{H}.json")):
        m = json.load(open(f))
        if m.get("tag"): continue
        rows.append(m)
    return pd.DataFrame(rows)


tab = load_bt(24)
t168 = load_bt(168)[["name", "MASE"]].rename(columns={"MASE": "MASE_H168"}) if glob.glob(str(BT / "*_H168.json")) else pd.DataFrame(columns=["name", "MASE_H168"])
tab = tab.merge(t168, on="name", how="left").sort_values("MASE").reset_index(drop=True)
cols = ["family", "name", "impl", "L", "strategy", "covariates", "params_total", "params_trainable",
        "train_s", "infer_s", "hardware", "MAE", "RMSE", "WAPE", "MAPE", "MASE", "MASE_H168", "MASE_xmas",
        "pinball", "pinball_scaled", "coverage_80"]
for c in cols:
    if c not in tab: tab[c] = np.nan
tab = tab[cols]
tab.to_csv(C.RES_DIR / "tabla_comparativa.csv", index=False)
sn = tab.loc[tab.name == "SeasonalNaive24", "MASE"].iloc[0]
tab["mejora_vs_SN24_%"] = (1 - tab.MASE / sn) * 100
tab.to_csv(C.RES_DIR / "tabla_comparativa.csv", index=False)
print(tab[["name", "MASE", "MAE", "WAPE", "MASE_xmas", "train_s", "mejora_vs_SN24_%"]].to_string())

# (1) MASE vs tiempo de entrenamiento (log) + frontera de Pareto
d = tab[tab.name != "Naive"].copy(); d["x"] = (d.train_s.fillna(0) + d.infer_s.fillna(0)).clip(lower=0.1)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.scatter(d.x, d.MASE, c=["C3" if f == "Fundacional" else "C0" for f in d.family])
for _, r in d.iterrows():
    ax.annotate(r["name"], (r.x, r.MASE), fontsize=7, xytext=(3, 3), textcoords="offset points")
ds = d.sort_values("x"); front, best = [], np.inf
for _, r in ds.iterrows():
    if r.MASE < best: front.append(r); best = r.MASE
front = pd.DataFrame(front)
ax.step(front.x, front.MASE, where="post", color="C2", lw=1.5, label="frontera de Pareto")
ax.set(xscale="log", xlabel="costo de cómputo [s] = entrenamiento + inferencia 92 orígenes (log)", ylabel="MASE prueba (H=24)",
       title="Precisión vs costo (Naive, MASE 4.27, fuera de escala)"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "pareto.png"); plt.close()

# (2) MAE por paso del horizonte
fig, ax = plt.subplots(figsize=(8, 4.5))
for n in tab.name:
    f = BT / f"{n}_H24_step.csv"
    if f.exists() and n != "Naive":
        s = pd.read_csv(f); ax.plot(s.h, s.MAE, label=n, lw=1.2)
ax.set(xlabel="paso h", ylabel="MAE [kWh]", title="MAE por paso del horizonte (prueba, sin Naive)")
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "mae_por_paso.png"); plt.close()

# Estrategias multi-paso (LSTM)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for n, lab in (("LSTM-AR", "autorregresiva"), ("LSTM-MO", "multi-salida"), ("Seq2Seq-LSTM", "encoder-decoder")):
    f = BT / f"{n}_H24_step.csv"
    if f.exists():
        s = pd.read_csv(f); ax.plot(s.h, s.MAE, marker=".", label=f"{lab} ({n})")
ax.set(xlabel="paso h", ylabel="MAE [kWh]", title="Estrategias multi-paso con LSTM"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "estrategias_lstm.png"); plt.close()

# Error por hora del día (top modelos)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for n in list(tab.name[:6]) + ["SeasonalNaive24"]:
    f = BT / f"{n}_H24_hour.csv"
    if f.exists():
        s = pd.read_csv(f); ax.plot(s.hour, s.MAE, label=n)
ax.set(xlabel="hora del día", ylabel="MAE [kWh]", title="Error por hora del día"); ax.legend(fontsize=7); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "mae_por_hora.png"); plt.close()

# (3) MASE vs meses de historia
hist = {}
for f in glob.glob(str(EXP / "hist_*.json")):
    m = json.load(open(f)); tag = m["tag"]
    mdl, mo = tag.split("_")[1], int(tag.split("_")[2][:-1])
    hist.setdefault(mdl, {})[mo] = m["test"]["MASE"]
for f in glob.glob(str(BT / "PatchTST_hist*_H24.json")):
    m = json.load(open(f)); hist.setdefault("PatchTST", {})[int(m["hist_months"])] = m["MASE"]
sn24 = float(tab.loc[tab.name == "SeasonalNaive24", "MASE"].iloc[0])
hist["SeasonalNaive24"] = {3: sn24, 12: sn24, 30: sn24}
fig, ax = plt.subplots(figsize=(6.5, 4))
for k, v in hist.items():
    xs = sorted(v); ax.plot(xs, [v[x] for x in xs], marker="o", label=k)
if "Chronos-Bolt" in tab.name.values:
    ax.axhline(float(tab.loc[tab.name == "Chronos-Bolt", "MASE"].iloc[0]), color="k", ls="--", label="Chronos-Bolt (zero-shot)")
ax.set(xlabel="meses de historia de entrenamiento", ylabel="MASE prueba", xticks=[3, 12, 30],
       title="MASE vs historia de entrenamiento"); ax.legend(fontsize=7); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "mase_vs_historia.png"); plt.close()
json.dump(hist, open(C.RES_DIR / "mase_vs_historia.json", "w"), indent=1)

# Tabla de iteraciones + curvas de pérdida
rows = []
for f in sorted(glob.glob(str(EXP / "hp_*.json"))):
    m = json.load(open(f))
    rows.append(dict(tag=m["tag"], arch=m["arch"], L=m["L"], H=m["H"], hist_months=m["hist_months"], norm=m["norm"],
                     lr=m["lr"], batch=m["batch_size"], model_kw=json.dumps(m["model_kw"]), epochs=m["epochs_run"],
                     params=m["params_total"], train_s=round(m["train_s"], 1), epoch_s=round(m["epoch_s_mean"], 2),
                     infer_s=round(m.get("infer_s", np.nan), 2), peak_mem_MB=round(m["cpu_peak_rss_MB"]),
                     val_MASE=round(m["best_val_MASE"], 4), test_MASE=round(m["test"]["MASE"], 4)))
if rows:
    it = pd.DataFrame(rows); it.to_csv(C.RES_DIR / "tabla_iteraciones.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, tag in zip(axs, ["hp_MLP_L168", "hp_MLP_lr0.003", "hp_DLinear_norm-none"]):
        f = EXP / f"{tag}.json"
        if not f.exists(): continue
        h = pd.DataFrame(json.load(open(f))["history"])
        ax.plot(h.epoch, h.train_loss, label="train (L1 norm.)"); ax.plot(h.epoch, h.val_loss, label="val (L1 norm.)")
        ax2 = ax.twinx(); ax2.plot(h.epoch, h.val_MASE, "g--", label="val MASE"); ax2.set_ylabel("val MASE", color="g")
        ax.set(title=tag, xlabel="epoch"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "curvas_perdida.png"); plt.close()

# Pronóstico probabilístico: ejemplo de bandas P10-P90 en Navidad
for n in ("DeepAR", "Chronos-Bolt"):
    f = BT / f"{n}_H24.npz"
    if not f.exists(): continue
    z = np.load(f)
    if "q10" not in z: continue
    from src.prep import get_context
    ctx = get_context(); o = z["origins"]; od = ctx.dates[o]
    sel = np.where((od >= pd.Timestamp("2014-12-20")) & (od <= pd.Timestamp("2014-12-28")))[0]
    s = int(np.argsort(ctx.Y[:, :ctx.val_pos].mean(1))[len(ctx.ids) // 2])
    t = np.concatenate([ctx.dates[o[k]:o[k] + 24] for k in sel])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, np.concatenate([ctx.Y[s, o[k]:o[k] + 24] for k in sel]), "k", lw=1, label="real")
    ax.plot(t, z["P"][s, sel].ravel(), "C0", lw=1, label="mediana")
    ax.fill_between(t, z["q10"][s, sel].ravel(), z["q90"][s, sel].ravel(), alpha=.3, label="P10-P90")
    ax.set(title=f"{n}: cliente {ctx.ids[s]} (mediana de escala), 20-28 dic 2014"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / f"bandas_{n}.png"); plt.close()

# Archivo de predicciones (unique_id, ds, modelo, y_hat)
parts = [pd.read_parquet(f) for f in sorted(glob.glob(str(C.PRED_DIR / "*.parquet")))]
if parts:
    pred = pd.concat(parts, ignore_index=True)
    pred["y_hat"] = pred.y_hat.round(4)
    pred.to_csv(C.RES_DIR / "predicciones_test.csv.gz", index=False, compression="gzip")
    print("predicciones:", pred.shape, pred.modelo.nunique(), "modelos")
