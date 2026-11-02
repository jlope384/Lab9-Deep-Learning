"""Carga, limpieza y preparación del panel horario (PDF, secciones 1 y 2).

Decisiones clave
----------------
* Unidades: el archivo registra potencia media en kW de cada intervalo de 15 min.
  Energía del intervalo = kW * 0.25 h. Energía horaria (kWh) = suma de los 4 registros / 4.
* Etiquetas: verificado empíricamente con los días de cambio de horario, la marca de tiempo es
  el INICIO del intervalo (el bloque anómalo de marzo/octubre está en 01:00-01:45). Por eso se
  agrupa con floor a la hora: `ds` = inicio de la hora (01:00 = consumo de 01:00 a 02:00).
* Cambios de horario (documentación UCI): todos los días tienen 96 registros.
  - Marzo (día de 23 h): la hora 01:00-02:00 viene en cero para todos los clientes -> se
    imputa interpolando linealmente entre 00:00 y 02:00 de cada cliente.
  - Octubre (día de 25 h): la hora 01:00-02:00 acumula el consumo de dos horas -> se divide
    entre 2. Así la grilla queda regular de 24 h/día (lo que esperan los modelos).
"""
import zipfile, urllib.request
import numpy as np
import pandas as pd
from . import config as C


def download_raw():
    zpath = C.DATA_DIR / "electricity.zip"
    txt = C.DATA_DIR / "LD2011_2014.txt"
    if not txt.exists():
        if not zpath.exists():
            urllib.request.urlretrieve(C.RAW_URL, zpath)
        zipfile.ZipFile(zpath).extract("LD2011_2014.txt", C.DATA_DIR)
    return txt


def load_raw() -> pd.DataFrame:
    cache = C.DATA_DIR / "raw_15min.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raw = pd.read_csv(download_raw(), sep=";", decimal=",", index_col=0, parse_dates=True)
    raw = raw.astype("float32")
    raw.to_parquet(cache)
    return raw


def dst_days(years=(2012, 2013, 2014)):
    """Último domingo de marzo y de octubre (regla UE) para cada año."""
    out = {}
    for y in years:
        mar = pd.Timestamp(f"{y}-03-31")
        oct_ = pd.Timestamp(f"{y}-10-31")
        out[y] = (mar - pd.Timedelta(days=(mar.weekday() + 1) % 7),
                  oct_ - pd.Timedelta(days=(oct_.weekday() + 1) % 7))
    return out


def to_hourly_kwh(raw: pd.DataFrame) -> pd.DataFrame:
    hourly = raw.resample("h").sum(min_count=4) / 4.0   # sum(kW * 0.25 h) -> kWh por hora
    return hourly.astype("float32")


def fix_dst(hourly: pd.DataFrame, log=None) -> pd.DataFrame:
    h = hourly.copy()
    for y, (mar, oct_) in dst_days().items():
        t_mar = mar + pd.Timedelta(hours=1)
        t_oct = oct_ + pd.Timedelta(hours=1)
        if t_mar in h.index:
            before = h.loc[t_mar].abs().sum()
            h.loc[t_mar] = 0.5 * (h.loc[t_mar - pd.Timedelta(hours=1)] + h.loc[t_mar + pd.Timedelta(hours=1)])
            if log is not None:
                log.append(f"{t_mar}: hora en cero (suma panel={before:.1f}) -> interpolada")
        if t_oct in h.index:
            ratio = (h.loc[t_oct].sum() / h.loc[t_oct - pd.Timedelta(hours=1)].sum())
            h.loc[t_oct] = h.loc[t_oct] / 2.0
            if log is not None:
                log.append(f"{t_oct}: hora doble (ratio vs hora previa={ratio:.2f}) -> /2")
    return h


def select_clients(hourly: pd.DataFrame):
    """Clientes con datos completos desde 2012-01-01: consumo > 0 en el primer día de 2012,
    sin NaN y sin tramos de ceros >= 7 días (series que se 'apagan')."""
    p = hourly.loc[C.PERIOD_START:C.PERIOD_END]
    first_day_ok = p.loc[:"2012-01-01 23:00"].sum() > 0
    no_nan = p.notna().all()
    daily = p.resample("D").sum()
    zero_days = (daily == 0)
    # racha máxima de días en cero por cliente
    def max_run(col):
        s = col.values.astype(int)
        best = cur = 0
        for v in s:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best
    runs = zero_days.apply(max_run)
    keep = first_day_ok & no_nan & (runs < 7)
    summary = pd.DataFrame({"first_day_ok": first_day_ok, "no_nan": no_nan,
                            "max_zero_run_days": runs, "keep": keep})
    return p.loc[:, keep], summary


def sample_clients(panel: pd.DataFrame, n=C.N_CLIENTS, seed=C.SEED):
    rng = np.random.default_rng(seed)
    cols = sorted(rng.choice(panel.columns.values, size=min(n, panel.shape[1]), replace=False))
    return panel[cols]


def pt_holidays(index: pd.DatetimeIndex) -> pd.Series:
    import holidays
    pt = holidays.Portugal(years=sorted(set(index.year)))
    return pd.Series([d.date() in pt for d in index], index=index, dtype="int8"), pt


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Covariables conocidas a futuro, codificadas con seno/coseno (continuidad circular:
    23h está cerca de 0h) + indicadores de feriado y fin de semana."""
    hol, _ = pt_holidays(index)
    hour, dow, month = index.hour.values, index.dayofweek.values, index.month.values - 1
    df = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
        "is_holiday": hol.values.astype(np.float32),
        "is_weekend": (dow >= 5).astype(np.float32),
    }, index=index).astype("float32")
    return df


def build_panel(n_clients=C.N_CLIENTS, seed=C.SEED, verbose=True):
    """Pipeline completo crudo -> panel horario limpio. Devuelve (panel_sel, info)."""
    cache = C.DATA_DIR / f"panel_{n_clients}_{seed}.parquet"
    info = {}
    if cache.exists() and (C.DATA_DIR / "panel_all.parquet").exists():
        panel_sel = pd.read_parquet(cache)
        info["panel_all"] = pd.read_parquet(C.DATA_DIR / "panel_all.parquet")
        info["selection"] = pd.read_csv(C.DATA_DIR / "client_selection.csv", index_col=0)
        info["dst_log"] = (C.DATA_DIR / "dst_log.txt").read_text().splitlines()
        return panel_sel, info
    raw = load_raw()
    info["raw_shape"] = raw.shape
    hourly = to_hourly_kwh(raw)
    log = []
    hourly = fix_dst(hourly, log)
    panel_all, summary = select_clients(hourly)
    panel_sel = sample_clients(panel_all, n_clients, seed)
    panel_all.to_parquet(C.DATA_DIR / "panel_all.parquet")
    panel_sel.to_parquet(cache)
    summary.to_csv(C.DATA_DIR / "client_selection.csv")
    (C.DATA_DIR / "dst_log.txt").write_text("\n".join(log))
    info.update(panel_all=panel_all, selection=summary, dst_log=log)
    return panel_sel, info


def split_index(index: pd.DatetimeIndex):
    """Índices enteros (posiciones) de inicio de cada partición en la grilla horaria."""
    pos = {k: index.get_loc(pd.Timestamp(v)) for k, v in
           dict(val=C.VAL_START, test=C.TEST_START).items()}
    return pos["val"], pos["test"]


class Scaler:
    """z-score por serie con estadísticos SOLO del bloque de entrenamiento (fuga #2)."""
    def __init__(self, mode="series"):
        assert mode in ("series", "global", "none")
        self.mode = mode

    def fit(self, Y_train: np.ndarray):  # (n_series, T_train)
        if self.mode == "series":
            self.mu = Y_train.mean(1, keepdims=True)
            self.sd = Y_train.std(1, keepdims=True) + 1e-6
        elif self.mode == "global":
            self.mu = np.full((Y_train.shape[0], 1), Y_train.mean())
            self.sd = np.full((Y_train.shape[0], 1), Y_train.std() + 1e-6)
        else:
            self.mu = np.zeros((Y_train.shape[0], 1)); self.sd = np.ones((Y_train.shape[0], 1))
        self.mu, self.sd = self.mu.astype(np.float32), self.sd.astype(np.float32)
        return self

    def transform(self, Y, idx=None):
        mu, sd = (self.mu, self.sd) if idx is None else (self.mu[idx], self.sd[idx])
        return (Y - mu) / sd

    def inverse(self, Z, idx=None):
        mu, sd = (self.mu, self.sd) if idx is None else (self.mu[idx], self.sd[idx])
        return Z * sd + mu


def to_long(panel: pd.DataFrame) -> pd.DataFrame:
    """Formato Nixtla (unique_id, ds, y)."""
    df = panel.copy(); df.index.name = "ds"
    return df.reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")[["unique_id", "ds", "y"]]
