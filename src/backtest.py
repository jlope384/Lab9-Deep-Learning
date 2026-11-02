"""Protocolo único de evaluación: backtest(modelo) con origen móvil diario (PDF sección 4).

Todo modelo expone `predict(Y_hist, o)`:
    Y_hist : (n_series, o) consumo en kWh HASTA la hora o-1 (23:00 del día anterior)
    o      : posición del origen (00:00 del día pronosticado) en la grilla horaria
  y devuelve (n_series, H) en kWh o un dict {"mean": ..., 0.1: ..., 0.5: ..., 0.9: ...}.
La función corta explícitamente Y[:, :o] => es imposible que el modelo vea el futuro.
"""
import time
import numpy as np
import pandas as pd
from . import config as C
from .metrics import point_metrics, per_client, per_step, pinball


class Context:
    """Contenedor del panel: Y (kWh), fechas, calendario, ids, escala MASE, posiciones de split."""
    def __init__(self, Y, dates, cal, ids, scale, val_pos, test_pos):
        self.Y, self.dates, self.cal, self.ids = Y, dates, cal, ids
        self.scale, self.val_pos, self.test_pos = scale, val_pos, test_pos
        self.T = Y.shape[1]

    def origins(self, split="test", H=C.H):
        lo, hi = (self.val_pos, self.test_pos) if split == "val" else (self.test_pos, self.T)
        return np.arange(lo, hi - H + 1, 24)   # 00:00 de cada día


def backtest(model, ctx: Context, split="test", H=C.H, name=None, origins=None):
    origins = ctx.origins(split, H) if origins is None else origins
    n = ctx.Y.shape[0]
    P = np.zeros((n, len(origins), H), np.float32)
    Q = {}
    t0 = time.perf_counter()
    for k, o in enumerate(origins):
        out = model.predict(ctx.Y[:, :o], int(o))          # <- solo historia antes del origen
        if isinstance(out, dict):
            P[:, k] = out["mean"]
            for q in (0.1, 0.5, 0.9):
                if q in out:
                    Q.setdefault(q, np.zeros_like(P))[:, k] = out[q]
        else:
            P[:, k] = out
    infer_s = time.perf_counter() - t0
    Yt = np.stack([ctx.Y[:, o:o + H] for o in origins], 1)
    res = BacktestResult(name or getattr(model, "name", type(model).__name__), ctx, origins, H, P, Yt, Q, infer_s)
    return res


class BacktestResult:
    def __init__(self, name, ctx, origins, H, P, Y, Q, infer_s):
        self.name, self.ctx, self.origins, self.H = name, ctx, origins, H
        self.P, self.Y, self.Q, self.infer_s = P, Y, Q, infer_s

    def metrics(self):
        m = point_metrics(self.Y, self.P, self.ctx.scale)
        m["infer_s"] = self.infer_s
        if self.Q:
            m.update(pinball(self.Y, self.Q))
        # semana de Navidad 22-28 dic
        od = self.ctx.dates[self.origins]
        xm = (od >= pd.Timestamp(C.XMAS_START)) & (od <= pd.Timestamp(C.XMAS_END))
        if xm.any():
            m["MASE_xmas"] = point_metrics(self.Y[:, xm], self.P[:, xm], self.ctx.scale)["MASE"]
            m["MAE_xmas"] = point_metrics(self.Y[:, xm], self.P[:, xm], self.ctx.scale)["MAE"]
        return m

    def per_client(self):
        return per_client(self.Y, self.P, self.ctx.scale, self.ctx.ids)

    def per_step(self):
        return per_step(self.Y, self.P, self.ctx.scale)

    def per_hour(self):
        hours = np.array([(self.ctx.dates[o:o + self.H].hour) for o in self.origins])  # (n_o, H)
        ae = np.abs(self.P - self.Y).mean(0)  # (n_o, H)
        return pd.DataFrame({"hour": hours.ravel(), "AE": ae.ravel()}).groupby("hour").AE.mean().rename("MAE").reset_index()

    def to_long(self):
        ds = np.stack([self.ctx.dates[o:o + self.H] for o in self.origins])  # (n_o,H)
        n = len(self.ctx.ids)
        return pd.DataFrame({"unique_id": np.repeat(self.ctx.ids, ds.size),
                             "ds": np.tile(ds.ravel(), n),
                             "modelo": self.name, "y_hat": self.P.reshape(n, -1).ravel()})


class PrecomputedForecaster:
    """Adaptador para librerías (StatsForecast/NeuralForecast) que pronostican vía
    `cross_validation`: guarda la salida indexada por cutoff y la entrega a backtest(),
    verificando que el cutoff de la librería sea exactamente la hora o-1 (sin fuga)."""
    def __init__(self, name, cv_df, ctx, col, qcols=None):
        self.name, self.ctx = name, ctx
        self.H = cv_df.groupby(["unique_id", "cutoff"]).size().max()
        cv = cv_df.sort_values(["cutoff", "unique_id", "ds"])
        self.tab = {}
        uid_pos = {u: i for i, u in enumerate(ctx.ids)}
        for cutoff, g in cv.groupby("cutoff"):
            o = ctx.dates.get_loc(pd.Timestamp(cutoff)) + 1
            arrs = {}
            for key, c in [("mean", col)] + list((qcols or {}).items()):
                M = np.zeros((len(ctx.ids), self.H), np.float32)
                for u, gg in g.groupby("unique_id"):
                    M[uid_pos[u]] = gg[c].values[:self.H]
                arrs[key] = M
            self.tab[o] = arrs

    def predict(self, Y_hist, o):
        assert o in self.tab, f"origen {o} no está en la salida de la librería"
        assert Y_hist.shape[1] == o
        a = self.tab[o]
        return a if len(a) > 1 else a["mean"]
