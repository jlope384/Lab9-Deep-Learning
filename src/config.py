"""Configuración global del laboratorio (rutas, fechas de split, semillas, dispositivo)."""
from pathlib import Path
import os, random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RES_DIR = ROOT / "results"
FIG_DIR = RES_DIR / "figures"
MODEL_DIR = RES_DIR / "models"
PRED_DIR = RES_DIR / "predictions"
for d in (DATA_DIR, RES_DIR, FIG_DIR, MODEL_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_URL = ("https://archive.ics.uci.edu/static/public/321/"
           "electricityloaddiagrams20112014.zip")

# --- Definición del problema (PDF, sección 1) ---
PERIOD_START = "2012-01-01 00:00"
PERIOD_END = "2014-12-31 23:00"
TRAIN_END = "2014-06-30 23:00"
VAL_START, VAL_END = "2014-07-01 00:00", "2014-09-30 23:00"
TEST_START, TEST_END = "2014-10-01 00:00", "2014-12-31 23:00"
XMAS_START, XMAS_END = "2014-12-22 00:00", "2014-12-28 23:00"

L_DEFAULT = 168      # ventana de entrada (horas)
H = 24               # horizonte day-ahead
N_TEST_ORIGINS = 92  # 1-oct a 31-dic 2014
N_CLIENTS = 100      # mínimo permitido por el PDF (restricción de cómputo: CPU)
SEED = 42
MASE_M = 24          # estacionalidad usada en la escala de MASE (seasonal naive diario)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_str():
    import platform
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return f"CPU ({os.cpu_count()} núcleos, {platform.processor() or platform.machine()})"
