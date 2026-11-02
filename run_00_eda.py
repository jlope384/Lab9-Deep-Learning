"""Sección 2: exploración, conteo de ventanas y auditoría de las 4 fugas (con asserts)."""
import json
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C, data as D
from src.prep import get_context
from src.windows import WindowDataset
from src.data import Scaler

ctx = get_context()
panel_all = pd.read_parquet(C.DATA_DIR / "panel_all.parquet")
panel = ctx.panel
out = {}
# ---------- escalas ----------
means_all = panel_all.loc[:C.TRAIN_END].mean()
means_sel = panel.loc[:C.TRAIN_END].mean()
out["scale_all"] = means_all.describe()[["min", "50%", "max"]].round(3).to_dict()
out["scale_sel"] = means_sel.describe()[["min", "50%", "max"]].round(3).to_dict()
out["n_clients_filter"], out["n_clients_used"] = panel_all.shape[1], panel.shape[1]
out["hours_per_series"] = panel.shape[0]
out["obs_panel_used"], out["obs_panel_filtered"] = int(panel.size), int(panel_all.size)

# ---------- perfiles (consumo normalizado por la media de cada cliente) ----------
rel = panel / panel.loc[:C.TRAIN_END].mean()
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
prof_h = rel.groupby(rel.index.hour).mean().mean(1)
prof_d = rel.groupby(rel.index.dayofweek).mean().mean(1)
hm = rel.mean(1).groupby([rel.index.dayofweek, rel.index.hour]).mean().unstack()
ax[0].plot(prof_h.index, prof_h.values, marker="o"); ax[0].set(title="Perfil por hora del día", xlabel="hora", ylabel="consumo / media del cliente")
ax[1].bar(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], prof_d.values); ax[1].set(title="Perfil por día de la semana")
im = ax[2].imshow(hm.values, aspect="auto", cmap="viridis"); ax[2].set(title="Día de semana × hora", xlabel="hora", yticks=range(7), yticklabels=["L", "M", "X", "J", "V", "S", "D"])
plt.colorbar(im, ax=ax[2]); plt.tight_layout(); plt.savefig(C.FIG_DIR / "perfiles.png", dpi=130); plt.close()

# ---------- serie diaria con feriados ----------
hol, pt = D.pt_holidays(panel.index)
daily = rel.mean(1).resample("D").mean()
hol_days = sorted({d for d in pd.to_datetime(list(pt.keys())) if daily.index[0] <= d <= daily.index[-1]})
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(daily.index, daily.values, lw=0.8)
ax.scatter(hol_days, daily.reindex(hol_days).values, color="red", s=18, zorder=3, label="feriado PT")
for a, b, c in [(C.VAL_START, C.VAL_END, "orange"), (C.TEST_START, C.TEST_END, "green")]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.12, color=c)
ax.set(title="Consumo diario relativo del panel (rojo: feriados de Portugal; naranja=val, verde=test)"); ax.legend()
plt.tight_layout(); plt.savefig(C.FIG_DIR / "serie_diaria_feriados.png", dpi=130); plt.close()
out["holidays"] = {str(k): v for k, v in sorted(pt.items())}

# ---------- Navidad y Año Nuevo ----------
fig, ax = plt.subplots(figsize=(12, 4)); xmas = {}
for y in (2012, 2013, 2014):
    s = daily.loc[f"{y}-12-15":f"{y}-12-31"]
    if y < 2014: s = pd.concat([s, daily.loc[f"{y+1}-01-01":f"{y+1}-01-07"]])
    ax.plot(range(len(s)), s.values, marker=".", label=str(y))
    base = daily.loc[f"{y}-11-17":f"{y}-12-14"]
    for d in ("12-24", "12-25", "12-31"):
        day = pd.Timestamp(f"{y}-{d}")
        same_dow = base[base.index.dayofweek == day.dayofweek].mean()
        xmas[f"{y}-{d}"] = round(float(daily[day] / same_dow), 3)
    if y < 2014:
        day = pd.Timestamp(f"{y+1}-01-01"); same_dow = base[base.index.dayofweek == day.dayofweek].mean()
        xmas[f"{y+1}-01-01"] = round(float(daily[day] / same_dow), 3)
ax.set(title="15-dic a 7-ene: consumo diario relativo", xlabel="días desde 15-dic"); ax.legend()
plt.tight_layout(); plt.savefig(C.FIG_DIR / "navidad.png", dpi=130); plt.close()
out["xmas_ratio_vs_same_weekday_prev4w"] = xmas

# ---------- escalas (histograma) ----------
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(np.log10(means_all + 1e-3), bins=40, alpha=0.6, label="317 filtrados")
ax.hist(np.log10(means_sel + 1e-3), bins=40, alpha=0.6, label="100 usados")
ax.set(xlabel="log10(consumo medio horario, kWh)", title="Escalas entre clientes"); ax.legend()
plt.tight_layout(); plt.savefig(C.FIG_DIR / "escalas.png", dpi=130); plt.close()

# ---------- conteo de ventanas ----------
sc = Scaler("series").fit(ctx.Y[:, :ctx.val_pos]); Yn = sc.transform(ctx.Y)
L, H = 168, 24
dtr = WindowDataset(Yn, ctx.cal, L, H, 0, ctx.val_pos)
dva = WindowDataset(Yn, ctx.cal, L, H, ctx.val_pos, ctx.test_pos)
dte = WindowDataset(Yn, ctx.cal, L, H, ctx.test_pos, ctx.T)
out["windows"] = {k: dict(per_series=d.n_o, panel=len(d)) for k, d in
                  [("train", dtr), ("val", dva), ("test", dte)]}
out["windows"]["test_daily_origins"] = len(ctx.origins("test"))
out["windows"]["val_daily_origins"] = len(ctx.origins("val"))
x, xf, y, s = dtr[0]
out["tensor_shapes"] = dict(x=list(x.shape), xf=list(xf.shape), y=list(y.shape))

# ---------- AUDITORÍA DE FUGAS ----------
audit = []
# (1) Split aleatorio de ventanas: los splits son por tiempo y los objetivos no se traslapan.
tr_lo, tr_hi = dtr.target_positions(); va_lo, va_hi = dva.target_positions(); te_lo, te_hi = dte.target_positions()
assert tr_hi < ctx.val_pos <= va_lo and va_hi < ctx.test_pos <= te_lo
audit.append(f"1) Split temporal: objetivos train hasta {ctx.dates[tr_hi]}, val {ctx.dates[va_lo]}..{ctx.dates[va_hi]}, "
             f"test desde {ctx.dates[te_lo]}. Ningún objetivo cruza de bloque. El muestreo aleatorio (shuffle) "
             f"ocurre SOLO dentro de las ventanas de entrenamiento.")
# (2) Escalado con toda la serie: estadísticos solo de train y verificación de que difieren de los globales.
mu_full = ctx.Y.mean(1, keepdims=True)
assert np.allclose(sc.mu, ctx.Y[:, :ctx.val_pos].mean(1, keepdims=True))
Y_pert = ctx.Y.copy(); Y_pert[:, ctx.val_pos:] *= 10       # alterar el futuro no cambia el scaler
assert np.allclose(Scaler("series").fit(Y_pert[:, :ctx.val_pos]).mu, sc.mu)
audit.append(f"2) Escalado: mu/sd por serie calculados con Y[:, :val_pos] (hasta {C.TRAIN_END}). "
             f"Multiplicar val+test por 10 no cambia los estadísticos. Diferencia media |mu_train - mu_full| = "
             f"{float(np.abs(sc.mu - mu_full).mean()):.2f} kWh (lo que se habría filtrado).")
# (3) Early stopping con prueba: el trainer solo recibe orígenes de validación.
vo = ctx.origins("val")
assert ctx.dates[vo.max() + 23] < pd.Timestamp(C.TEST_START)
audit.append(f"3) Early stopping: train_model() evalúa en {len(vo)} orígenes de validación, último objetivo "
             f"{ctx.dates[vo.max()+23]} < {C.TEST_START}. La prueba solo se toca en backtest(split='test') tras entrenar.")
# (4) Covariables mal alineadas: el calendario futuro corresponde exactamente a las horas objetivo,
#     y la entrada de la red no cambia si se altera el futuro (test de perturbación).
t = int(dte.origins[0])
x1, xf1, y1, _ = dte[0]
assert torch.allclose(xf1, torch.as_tensor(D.calendar_features(ctx.dates[t:t + H]).values))
Yn2 = Yn.copy(); Yn2[:, t:] = 999.0
x2, _, _, _ = WindowDataset(Yn2, ctx.cal, L, H, ctx.test_pos, ctx.T)[0]
assert torch.equal(x1, x2), "la entrada depende del futuro!"
from src.lgbm import make_rows
Xa, _ = make_rows(Yn, ctx.cal, t, t + 1, series=[0]); Xb, _ = make_rows(Yn2, ctx.cal, t, t + 1, series=[0])
assert np.array_equal(Xa, Xb)
audit.append("4) Alineación: XF = calendario de [t, t+H) (determinístico, conocido a futuro); la entrada X usa "
             "[t-L, t). Poner y[t:] = 999 no altera X (ni las filas de rezagos de LightGBM): toda entrada es "
             "calculable a las 00:00 del día pronosticado con datos hasta las 23:00 del día previo.")
audit.append("Extra) backtest() pasa Y[:, :o] al modelo; para librerías se verifica que cutoff == o-1.")
out["leakage_audit"] = audit
json.dump(out, open(C.RES_DIR / "eda_summary.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "holidays"}, indent=1, default=str))
