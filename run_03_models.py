"""Entrena y evalúa (backtest de prueba, 92 orígenes) LightGBM, Chronos y modelos PyTorch."""
import sys, json, time, torch
from src.prep import get_context
from src.backtest import backtest
from src.train import train_model
from src import store, config as C
ctx = get_context()
stages = sys.argv[1:] or ["lgbm", "linear", "chronos", "lstm"]
HW = C.hardware_str()
LOG = C.RES_DIR / "train_logs"; LOG.mkdir(exist_ok=True)

def run_torch(arch, name, family, strategy, cov, **kw):
    fc, info = train_model(arch, ctx, name=name, **kw)
    json.dump(info, open(LOG / f"{name}.json", "w"), indent=1, default=str)
    torch.save(fc.model.state_dict(), C.MODEL_DIR / f"{name}.pt")
    res = backtest(fc, ctx, "test", 24)
    m = store.save(res, dict(family=family, L=info["L"], strategy=strategy, covariates=cov,
        params_total=info["params_total"], params_trainable=info["params_trainable"],
        train_s=info["train_s"], hardware=HW, impl="PyTorch (propio)", best_val_MASE=info["best_val_MASE"]))
    print("TEST", name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)
    return fc, info

if "lgbm" in stages:
    from src.lgbm import LGBMForecaster
    f = LGBMForecaster(ctx).fit()
    res = backtest(f, ctx, "test", 24)
    m = store.save(res, dict(family="Árboles", L=168, strategy="recursiva", covariates="calendario",
        params_total=f.bst.num_trees(), params_trainable=f.bst.num_trees(), train_s=f.train_s,
        hardware=HW, impl="LightGBM", note=f"{f.n_rows} filas, {f.bst.num_trees()} árboles"))
    print("TEST LightGBM", m, flush=True)

if "linear" in stages:
    run_torch("DLinear", "DLinear", "Lineal", "multi-salida", "no", epochs=20, samples_per_epoch=40000, lr=3e-3)
    run_torch("NLinear", "NLinear", "Lineal", "multi-salida", "no", epochs=20, samples_per_epoch=40000, lr=3e-3)
    run_torch("MLP", "MLP", "MLP", "multi-salida", "calendario", epochs=20, samples_per_epoch=40000, lr=1e-3, hidden=512, layers=2, dropout=0.1)

if "chronos" in stages:
    from src.chronos_model import ChronosForecaster
    ch = ChronosForecaster()
    res = backtest(ch, ctx, "test", 24)
    m = store.save(res, dict(family="Fundacional", L=ch.context, strategy="multi-salida (cuantiles)",
        covariates="no", params_total=ch.params_total, params_trainable=0, train_s=0.0,
        hardware=HW, impl="chronos-forecasting (zero-shot)"))
    print("TEST Chronos", m, flush=True)

if "lstm" in stages:
    kw = dict(epochs=4, samples_per_epoch=8192, lr=3e-3, hidden=64, batch_size=256, patience=2)
    run_torch("LSTM_MO", "LSTM-MO", "Recurrente", "multi-salida", "calendario", **kw)
    run_torch("Seq2Seq", "Seq2Seq-LSTM", "Encoder-decoder", "encoder-decoder", "calendario (decoder)", **kw)
    run_torch("LSTM_AR", "LSTM-AR", "Recurrente", "autorregresiva", "calendario", **kw)

if "lstm_long" in stages:   # misma arquitectura multi-salida con más presupuesto (mejor recurrente)
    run_torch("LSTM_MO", "LSTM-MO-long", "Recurrente", "multi-salida", "calendario",
              epochs=12, samples_per_epoch=16384, lr=3e-3, hidden=64, batch_size=256, patience=3)
