"""Métricas en escala original (kWh). Arrays con forma (n_series, n_origins, H)."""
import numpy as np
import pandas as pd


def mase_scale(Y_train: np.ndarray, m=24):
    """Escala de MASE por serie: MAE in-sample del seasonal naive (m) en ENTRENAMIENTO."""
    return np.abs(Y_train[:, m:] - Y_train[:, :-m]).mean(1)


def point_metrics(y, yhat, scale):
    e = yhat - y
    ae = np.abs(e)
    mask = np.abs(y) > 1e-6
    out = dict(
        MAE=ae.mean(),
        RMSE=np.sqrt((e ** 2).mean()),
        WAPE=ae.sum() / np.abs(y).sum(),
        MAPE=(ae[mask] / np.abs(y[mask])).mean() * 100,   # % ; excluye y=0 (división indefinida)
        MASE=(ae.reshape(ae.shape[0], -1).mean(1) / scale).mean(),
    )
    return {k: float(v) for k, v in out.items()}


def per_client(y, yhat, scale, ids):
    ae = np.abs(yhat - y).reshape(y.shape[0], -1)
    se = ((yhat - y) ** 2).reshape(y.shape[0], -1)
    return pd.DataFrame({"unique_id": ids, "MAE": ae.mean(1), "RMSE": np.sqrt(se.mean(1)),
                         "MASE": ae.mean(1) / scale, "mean_kwh": y.reshape(y.shape[0], -1).mean(1)})


def per_step(y, yhat, scale):
    ae = np.abs(yhat - y)
    return pd.DataFrame({"h": np.arange(1, y.shape[2] + 1),
                         "MAE": ae.mean((0, 1)),
                         "MASE": (ae.mean(1) / scale[:, None]).mean(0)})


def pinball(y, q_preds: dict):
    """Pinball loss promedio sobre cuantiles (kWh) y cobertura del intervalo P10-P90."""
    losses = {}
    for q, p in q_preds.items():
        d = y - p
        losses[q] = np.maximum(q * d, (q - 1) * d).mean()
    out = {"pinball": float(np.mean(list(losses.values())))}
    out["pinball_scaled"] = out["pinball"] / float(np.abs(y).mean())
    if 0.1 in q_preds and 0.9 in q_preds:
        out["coverage_80"] = float(((y >= q_preds[0.1]) & (y <= q_preds[0.9])).mean())
    return out
