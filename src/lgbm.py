"""LightGBM global con rezagos 1-24 y 168 + calendario (estrategia recursiva de un paso).

* Global: un solo modelo para todas las series, con la serie normalizada (z-score de train).
* Rezagos: y_{t-1..t-24}, y_{t-168}. Para pronosticar 24 h se aplica recursivamente: la
  predicción de t+1 se convierte en el rezago 1 de t+2 (igual que mlforecast).
* Fuga #4 evitada: en la fila con objetivo y_t, todos los rezagos son y_{t-k} con k>=1, y el
  calendario es el de t (determinístico, conocido a futuro).
"""
import time
import numpy as np
import lightgbm as lgb
from . import config as C
from .data import Scaler

LAGS = list(range(1, 25)) + [168]


def make_rows(Yn, cal, t_from, t_to, step=1, series=None):
    series = np.arange(Yn.shape[0]) if series is None else series
    ts = np.arange(max(t_from, 168), t_to, step)
    X, y = [], []
    for s in series:
        lag = np.stack([Yn[s, ts - k] for k in LAGS], 1)
        X.append(np.concatenate([lag, cal[ts], np.full((len(ts), 1), s)], 1)); y.append(Yn[s, ts])
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.float32)


class LGBMForecaster:
    name = "LightGBM"

    def __init__(self, ctx, H=24, hist_months=None, params=None, row_step=3, seed=C.SEED):
        self.ctx, self.H = ctx, H
        self.scaler = Scaler("series").fit(ctx.Y[:, :ctx.val_pos])
        self.Yn = self.scaler.transform(ctx.Y)
        self.params = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                           seed=seed, verbose=-1, num_threads=2)
        self.params.update(params or {})
        self.hist_months, self.row_step = hist_months, row_step

    def fit(self, t_hi=None):
        ctx = self.ctx
        t_hi = ctx.val_pos if t_hi is None else t_hi
        t_lo = 0 if self.hist_months is None else t_hi - int(self.hist_months * 730)
        t0 = time.perf_counter()
        Xtr, ytr = make_rows(self.Yn, ctx.cal, t_lo, t_hi, self.row_step)       # solo train
        Xva, yva = make_rows(self.Yn, ctx.cal, ctx.val_pos, ctx.test_pos, 2)    # early stopping
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=[Xtr.shape[1] - 1])
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        self.bst = lgb.train(self.params, dtr, 1000, valid_sets=[dva],
                             callbacks=[lgb.early_stopping(30, verbose=False)])
        self.train_s = time.perf_counter() - t0
        self.n_rows = len(ytr)
        return self

    def predict(self, Y_hist, o):
        n = Y_hist.shape[0]
        buf = np.concatenate([self.scaler.transform(Y_hist[:, -168:]), np.zeros((n, self.H), np.float32)], 1)
        sid = np.arange(n)[:, None]
        for h in range(self.H):
            p = 168 + h
            lag = np.stack([buf[:, p - k] for k in LAGS], 1)
            X = np.concatenate([lag, np.repeat(self.ctx.cal[o + h][None], n, 0), sid], 1)
            buf[:, p] = self.bst.predict(X, num_threads=2)
        return self.scaler.inverse(buf[:, 168:])
