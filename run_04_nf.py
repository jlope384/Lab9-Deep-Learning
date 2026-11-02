"""Modelos de NeuralForecast: entrenados UNA vez (train, early stopping con val) y evaluados
con cross_validation(n_windows=92, step_size=24, refit=False) = 92 orígenes diarios de prueba.
La salida se pasa por el mismo backtest() mediante PrecomputedForecaster."""
import sys, time, json, logging, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TCN, PatchTST, iTransformer, DeepAR
from neuralforecast.losses.pytorch import MAE, DistributionLoss
from src.prep import get_context
from src.backtest import backtest, PrecomputedForecaster
from src import store, config as C, data as D

ctx = get_context()
which = sys.argv[1:] or ["NHITS", "TCN", "PatchTST", "iTransformer", "DeepAR"]
import os
hist_months = int(os.environ.get("NF_HIST", "0")) or None
MAIN = bool(os.environ.get("NF_HIST"))
for a in list(which):
    if a.startswith("hist="): hist_months = int(a[5:]); which.remove(a)
cal = list(ctx.cal_cols)
df = D.to_long(ctx.panel)
cal_df = pd.DataFrame(ctx.cal, columns=cal); cal_df["ds"] = ctx.dates
df = df.merge(cal_df, on="ds", how="left")
# z-score por serie con estadísticos de ENTRENAMIENTO (mismo criterio que los modelos propios).
# Se usa scaler_type="identity": el escalado por ventana de NF también normaliza las covariables
# futuras y explota cuando son casi constantes en la ventana de entrada (mes, feriado).
from src.data import Scaler
_sc = Scaler("series").fit(ctx.Y[:, :ctx.val_pos])
MU = dict(zip(ctx.ids, _sc.mu.ravel())); SD = dict(zip(ctx.ids, _sc.sd.ravel()))
df["y"] = (df["y"] - df["unique_id"].map(MU)) / df["unique_id"].map(SD)
for c in df.columns:
    if df[c].dtype == "float64": df[c] = df[c].astype("float32")
if hist_months:   # limitar historia de entrenamiento (val y test intactos)
    start = ctx.dates[ctx.val_pos - hist_months * 730 - 168]
    df = df[df.ds >= start]
H, L = 24, 168
VAL, TEST = 0, 2208   # sin validación interna: presupuesto fijo de pasos (ver README)
tr = dict(enable_progress_bar=False, logger=False, enable_model_summary=False, accelerator="cpu")
common = dict(h=H, input_size=L, max_steps=150, val_check_steps=150, early_stop_patience_steps=-1,
              batch_size=32, random_seed=C.SEED, scaler_type="identity", **tr)
MODELS = {
 "NHITS": lambda: NHITS(futr_exog_list=cal, loss=MAE(), windows_batch_size=256, learning_rate=1e-3, **common),
 "TCN": lambda: TCN(futr_exog_list=cal, loss=MAE(), encoder_hidden_size=64, learning_rate=1e-3,
                   **{**common, "input_size": L}),
 "PatchTST": lambda: PatchTST(loss=MAE(), patch_len=24, stride=12, hidden_size=64, n_heads=4,
                   encoder_layers=2, windows_batch_size=256, learning_rate=1e-3, **common),
 "iTransformer": lambda: iTransformer(n_series=len(ctx.ids), loss=MAE(), hidden_size=128, n_heads=4,
                   e_layers=2, learning_rate=1e-3, **{**common, "batch_size": len(ctx.ids)}),
 "DeepAR": lambda: DeepAR(futr_exog_list=cal, loss=DistributionLoss("StudentT", level=[80]),
                   lstm_hidden_size=64, lstm_n_layers=2, trajectory_samples=50, windows_batch_size=256,
                   learning_rate=1e-3, **{**common, "max_steps": 150}),
}
FAM = {"NHITS": ("Residual (MLP jerárquico)", "multi-salida", "calendario"),
       "TCN": ("Convolucional", "multi-salida (MQ)", "calendario"),
       "PatchTST": ("Atención", "multi-salida", "no"),
       "iTransformer": ("Atención", "multi-salida", "no"),
       "DeepAR": ("Probabilístico", "autorregresiva (muestreo)", "calendario")}
for name in which:
    torch.manual_seed(C.SEED)
    m = MODELS[name]()
    nf = NeuralForecast(models=[m], freq="h")
    t = time.perf_counter()
    cv = nf.cross_validation(df=df, n_windows=92, step_size=24, val_size=VAL, refit=False)
    tot = time.perf_counter() - t
    cv["ds"] = pd.to_datetime(cv["ds"]); cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    mu, sd = cv["unique_id"].map(MU), cv["unique_id"].map(SD)
    for c in cv.columns:
        if c not in ("unique_id", "ds", "cutoff"):
            cv[c] = cv[c] * sd + mu      # volver a kWh
    qcols = {0.1: f"{name}-lo-80", 0.9: f"{name}-hi-80", 0.5: f"{name}-median"} if name == "DeepAR" else None
    col = f"{name}-median" if name == "DeepAR" else name
    tag = f"_hist{hist_months}" if (hist_months and not MAIN) else ""
    pf = PrecomputedForecaster(name + tag, cv, ctx, col, qcols)
    res = backtest(pf, ctx, "test", H, name=name)
    p_tot = sum(p.numel() for p in nf.models[0].parameters())
    meta = dict(family=FAM[name][0], L=L, strategy=FAM[name][1], covariates=FAM[name][2],
                params_total=p_tot, params_trainable=p_tot, train_s=tot, hardware=C.hardware_str(),
                impl="NeuralForecast", note="train_s incluye entrenamiento + inferencia de cross_validation",
                hist_months=hist_months or 30)
    mm = store.save(res, meta, tag)
    print("TEST", name, tag, {k: round(v, 4) for k, v in mm.items() if isinstance(v, float)}, flush=True)
