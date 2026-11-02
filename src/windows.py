"""Dataset de ventanas deslizantes (PDF sección 2).

Convención: `t` = índice del ORIGEN = primera hora a pronosticar.
  entrada  X  = posiciones [t-L, t)    -> (L, 1 + n_cal): [y_norm, calendario de esas horas]
  futuro   XF = posiciones [t, t+H)    -> (H, n_cal): calendario conocido a futuro
  objetivo y  = posiciones [t, t+H)    -> (H,)
Una ventana pertenece a la partición en la que cae su OBJETIVO completo; la entrada puede venir
de la partición anterior porque esa historia ya es conocida en el instante t (no es fuga).
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def origin_range(T, L, H, t_lo, t_hi):
    """Orígenes válidos t con objetivo dentro de [t_lo, t_hi) y entrada dentro de [0, t)."""
    first = max(t_lo, L)
    last = t_hi - H  # inclusive
    return first, last


class WindowDataset(Dataset):
    def __init__(self, Yn, cal, L, H, t_lo, t_hi, stride=1, series=None, hist_lo=0):
        """Yn: (n_series, T) normalizado; cal: (T, n_cal).
        hist_lo: primera posición permitida para la ENTRADA (para limitar meses de historia)."""
        self.Yn = torch.as_tensor(Yn, dtype=torch.float32)
        self.cal = torch.as_tensor(cal, dtype=torch.float32)
        self.L, self.H = L, H
        first, last = origin_range(Yn.shape[1], L, H, max(t_lo, hist_lo + L), t_hi)
        self.origins = np.arange(first, last + 1, stride)
        self.series = np.arange(Yn.shape[0]) if series is None else np.asarray(series)
        self.n_o = len(self.origins)

    def __len__(self):
        return len(self.series) * self.n_o

    def __getitem__(self, i):
        s = self.series[i // self.n_o]
        t = int(self.origins[i % self.n_o])
        L, H = self.L, self.H
        x = torch.cat([self.Yn[s, t - L:t, None], self.cal[t - L:t]], dim=1)
        return x, self.cal[t:t + H], self.Yn[s, t:t + H], s

    def target_positions(self):
        """Posiciones mínima y máxima que tocan los objetivos (para la auditoría de fugas)."""
        return int(self.origins.min()), int(self.origins.max() + self.H - 1)
