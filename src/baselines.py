"""Baselines estadísticos con StatsForecast (Naive, SeasonalNaive 24/168, MSTL).

Son modelos LOCALES: en cada origen se ajustan solo con la historia previa (ventana de
`input_size` horas), por lo que el 'reentrenamiento' es inherente y no hay fuga.
"""
import time
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, MSTL, AutoETS


class SFForecaster:
    def __init__(self, name, model, ctx, H=24, input_size=24 * 7 * 2, n_jobs=1):
        self.name, self.model, self.ctx, self.H, self.input_size = name, model, ctx, H, input_size
        self.sf = StatsForecast(models=[model], freq="h", n_jobs=n_jobs)
        self.fit_s = 0.0
        self.alias = str(model)

    def predict(self, Y_hist, o):
        win = Y_hist[:, -self.input_size:]
        ds = self.ctx.dates[o - win.shape[1]:o]
        n = win.shape[0]
        df = pd.DataFrame({"unique_id": np.repeat(np.arange(n), win.shape[1]),
                           "ds": np.tile(ds, n), "y": win.ravel()})
        t = time.perf_counter()
        fc = self.sf.forecast(df=df, h=self.H)
        self.fit_s += time.perf_counter() - t
        fc = fc.sort_values(["unique_id", "ds"])
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        return fc[col].values.reshape(n, self.H).astype(np.float32)


def make_baselines(ctx, H=24):
    return [
        SFForecaster("Naive", Naive(), ctx, H, input_size=24),
        SFForecaster("SeasonalNaive24", SeasonalNaive(season_length=24), ctx, H, input_size=48),
        SFForecaster("SeasonalNaive168", SeasonalNaive(season_length=168), ctx, H, input_size=336),
        SFForecaster("MSTL", MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN")),
                     ctx, H, input_size=24 * 7 * 2, n_jobs=1),
    ]
