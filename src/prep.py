"""Atajo para construir el contexto común (mismo panel de clientes para TODOS los modelos)."""
import numpy as np
import pandas as pd
from . import config as C, data as D
from .backtest import Context
from .metrics import mase_scale


def get_context(n_clients=C.N_CLIENTS, seed=C.SEED):
    panel, _ = D.build_panel(n_clients, seed)
    dates = panel.index
    cal_df = D.calendar_features(dates)
    val_pos, test_pos = D.split_index(dates)
    Y = panel.values.T.astype(np.float32)               # (n_series, T)
    scale = mase_scale(Y[:, :val_pos], C.MASE_M)          # escala MASE: solo entrenamiento
    ctx = Context(Y, dates, cal_df.values, np.array(panel.columns), scale, val_pos, test_pos)
    ctx.cal_cols = list(cal_df.columns)
    ctx.panel = panel
    return ctx
