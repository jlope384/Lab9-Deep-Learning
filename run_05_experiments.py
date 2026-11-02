"""Sección 5.2-5.3 + sección 4 (reentrenamiento) + semillas.
Uso: python run_05_experiments.py <bloque> [...]   bloques: hp, hist, local, seeds, retrain, h168
La SELECCIÓN usa solo validación (best_val_MASE). La prueba se calcula una vez por configuración
para las gráficas de análisis (no para elegir)."""
import sys, json, time
import numpy as np, pandas as pd, torch
from src.prep import get_context
from src.backtest import backtest
from src.train import train_model
from src.metrics import point_metrics
from src import store, config as C

ctx = get_context()
blocks = sys.argv[1:]
EXP = C.RES_DIR / "experiments"; EXP.mkdir(exist_ok=True)
HP_KW = dict(epochs=6, samples_per_epoch=12000, patience=2)


def log_run(tag, arch, test=True, **kw):
    fc, info = train_model(arch, ctx, name=tag, verbose=False, **kw)
    if test:
        res = backtest(fc, ctx, "test", kw.get("H", 24), name=tag)
        info["test"] = res.metrics(); info["infer_s"] = res.infer_s
    info["tag"] = tag
    json.dump(info, open(EXP / f"{tag}.json", "w"), indent=1, default=str)
    print(f"{tag}: val_MASE={info['best_val_MASE']:.4f} test_MASE={info.get('test', {}).get('MASE', float('nan')):.4f} "
          f"params={info['params_total']} train_s={info['train_s']:.1f}", flush=True)
    return fc, info


if "hp" in blocks:
    MLPK = dict(hidden=512, layers=2, dropout=0.1, lr=1e-3)
    # Iteración 1: ventana L (48, 168, 336)
    for L in (48, 168, 336):
        log_run(f"hp_DLinear_L{L}", "DLinear", L=L, lr=3e-3, **HP_KW)
        log_run(f"hp_MLP_L{L}", "MLP", L=L, **MLPK, **HP_KW)
    # Iteración 2: normalización (sin, global, por serie, RevIN)
    for norm in ("none", "global", "series", "revin"):
        log_run(f"hp_DLinear_norm-{norm}", "DLinear", norm=norm, lr=3e-3, **HP_KW)
    for norm in ("none", "revin"):
        log_run(f"hp_MLP_norm-{norm}", "MLP", norm=norm, **MLPK, **HP_KW)
    # Iteraciones 3-7 (MLP, una variación a la vez respecto a la base hp_MLP_L168)
    log_run("hp_MLP_hidden128", "MLP", **{**MLPK, "hidden": 128}, **HP_KW)
    log_run("hp_MLP_layers1", "MLP", **{**MLPK, "layers": 1}, **HP_KW)
    log_run("hp_MLP_dropout0.3", "MLP", **{**MLPK, "dropout": 0.3}, **HP_KW)
    log_run("hp_MLP_lr0.003", "MLP", **{**MLPK, "lr": 3e-3}, **HP_KW)
    log_run("hp_MLP_bs1024", "MLP", **MLPK, batch_size=1024, **HP_KW)

if "hist" in blocks:
    # Iteración: meses de historia (3, 12, 30) para DLinear, MLP, LSTM-MO y LightGBM
    from src.lgbm import LGBMForecaster
    for m in (3, 12, 30):
        log_run(f"hist_DLinear_{m}m", "DLinear", hist_months=m, lr=3e-3, **HP_KW)
        log_run(f"hist_LSTM-MO_{m}m", "LSTM_MO", hist_months=m, hidden=64, lr=3e-3, epochs=2, samples_per_epoch=4096, patience=2)
        f = LGBMForecaster(ctx, hist_months=m).fit()
        r = backtest(f, ctx, "test", 24, name=f"hist_LightGBM_{m}m")
        json.dump(dict(tag=f"hist_LightGBM_{m}m", hist_months=m, train_s=f.train_s, test=r.metrics()),
                  open(EXP / f"hist_LightGBM_{m}m.json", "w"), indent=1, default=float)
        print(f"hist_LightGBM_{m}m test_MASE={r.metrics()['MASE']:.4f}", flush=True)

BEST = json.load(open(EXP / "best_own.json")) if (EXP / "best_own.json").exists() else None

if "local" in blocks:
    rng = np.random.default_rng(C.SEED); ten = np.sort(rng.choice(len(ctx.ids), 10, replace=False))
    arch, kw = BEST["arch"], BEST["kw"]
    g = torch.load(C.MODEL_DIR / f"{BEST['name']}.pt") if False else None
    rows = []
    # global: ya entrenado (se reentrena aquí con la misma config para tener el objeto)
    fc_g, _ = train_model(arch, ctx, name="global", verbose=False, **kw)
    rg = backtest(fc_g, ctx, "test", 24)
    for s in ten:
        fc_l, info = train_model(arch, ctx, name=f"local_{s}", series=[s], verbose=False,
                                 **{**kw, "samples_per_epoch": 5000, "batch_size": 64})
        Yt = np.stack([ctx.Y[[s], o:o + 24] for o in ctx.origins("test")], 1)
        Pl = np.stack([fc_l.predict(ctx.Y[:, :o], int(o)) for o in ctx.origins("test")], 1)
        ml = point_metrics(Yt, Pl, ctx.scale[[s]])["MASE"]
        mg = point_metrics(rg.Y[[s]], rg.P[[s]], ctx.scale[[s]])["MASE"]
        rows.append(dict(unique_id=ctx.ids[s], mean_kwh=float(ctx.Y[s, :ctx.val_pos].mean()),
                         MASE_local=ml, MASE_global=mg, local_train_s=info["train_s"]))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(EXP / "local_vs_global.csv", index=False)

if "seeds" in blocks:
    arch, kw = BEST["arch"], BEST["kw"]
    rows = []
    for sd in (0, 1, 2):
        fc, info = train_model(arch, ctx, name=f"seed{sd}", seed=sd, verbose=False, **kw)
        r = backtest(fc, ctx, "test", 24); rows.append(dict(seed=sd, val_MASE=info["best_val_MASE"], **r.metrics()))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(EXP / "seeds.csv", index=False)
    print("std MASE test:", df.MASE.std(ddof=1))

if "retrain" in blocks:
    # Reentrenamiento mensual con ventana expandible: al inicio de oct/nov/dic se reentrena con
    # TODOS los datos hasta las 23:00 del día anterior y epochs fijos (= best_epoch original).
    arch, kw = BEST["arch"], dict(BEST["kw"])
    kw["epochs"] = BEST.get("best_epoch", kw.get("epochs", 10))
    origins = ctx.origins("test"); od = ctx.dates[origins]
    P = np.zeros((len(ctx.ids), len(origins), 24), np.float32)
    for mo in (10, 11, 12):
        t_hi = ctx.dates.get_loc(pd.Timestamp(f"2014-{mo:02d}-01"))
        fc, info = train_model(arch, ctx, name=f"retrain_{mo}", t_hi=t_hi, verbose=False, **kw)
        sel = np.where(od.month == mo)[0]
        r = backtest(fc, ctx, "test", 24, origins=origins[sel])
        P[:, sel] = r.P
        print("refit", mo, info["train_s"], flush=True)
    Y = np.stack([ctx.Y[:, o:o + 24] for o in origins], 1)
    m = point_metrics(Y, P, ctx.scale)
    json.dump(dict(model=BEST["name"], monthly_refit=m), open(EXP / "retrain_dl.json", "w"), indent=1)
    print("retrain", m, flush=True)

if "h168" in blocks:
    for arch, kw in (("DLinear", dict(lr=3e-3)), ("NLinear", dict(lr=3e-3)),
                     ("MLP", dict(lr=1e-3, hidden=512, layers=2, dropout=0.1))):
        fc, info = train_model(arch, ctx, H=168, L=336, name=f"{arch}_H168", verbose=False, **{**HP_KW, **kw})
        r = backtest(fc, ctx, "test", 168, name=arch)
        store.save(r, dict(L=336, train_s=info["train_s"]), tag="")
        print(arch, "H168", r.metrics()["MASE"], flush=True)

if "hist_lstm" in blocks:   # mejor recurrente con presupuesto mayor (misma config que LSTM-MO-long)
    for m in (3, 12, 30):
        log_run(f"hist_LSTM-MO_{m}m", "LSTM_MO", hist_months=m, hidden=64, lr=3e-3, epochs=8,
                samples_per_epoch=12000, patience=3)
