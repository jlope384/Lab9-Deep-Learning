"""Modelos propios en PyTorch. Interfaz común: forward(x, xf, y_teacher=None) -> (B, H)
  x  : (B, L, 1 + n_cal)  [y normalizado, calendario de las horas de entrada]
  xf : (B, H, n_cal)      calendario de las horas a pronosticar (conocido a futuro)
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022): normaliza cada ventana con su
    media/desv. y desnormaliza la salida. Ayuda ante cambios de nivel entre train y test."""
    def __init__(self, affine=True):
        super().__init__()
        self.affine = affine
        if affine:
            self.w = nn.Parameter(torch.ones(1)); self.b = nn.Parameter(torch.zeros(1))

    def norm(self, y):  # y: (B, L)
        self.mu = y.mean(1, keepdim=True).detach()
        self.sd = (y.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        z = (y - self.mu) / self.sd
        return z * self.w + self.b if self.affine else z

    def denorm(self, z):
        if self.affine:
            z = (z - self.b) / (self.w + 1e-8)
        return z * self.sd + self.mu


class _Base(nn.Module):
    def __init__(self, revin=False):
        super().__init__()
        self.revin = RevIN() if revin else None

    def _in(self, x):
        if self.revin is None:
            return x
        y = self.revin.norm(x[..., 0])
        return torch.cat([y[..., None], x[..., 1:]], -1)

    def _out(self, yhat):
        return yhat if self.revin is None else self.revin.denorm(yhat)


class MLP(_Base):
    """MLP multi-salida: aplana historia + calendario futuro y devuelve los 24 pasos."""
    def __init__(self, L, H, n_cal, hidden=512, layers=2, dropout=0.1, revin=False):
        super().__init__(revin)
        d_in = L * (1 + n_cal) + H * n_cal
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]; d = hidden
        mods.append(nn.Linear(d, H))
        self.net = nn.Sequential(*mods)

    def forward(self, x, xf, y_teacher=None):
        x = self._in(x)
        return self._out(self.net(torch.cat([x.flatten(1), xf.flatten(1)], 1)))


class MovingAvg(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k = k; self.pool = nn.AvgPool1d(k, stride=1)

    def forward(self, y):  # (B, L) -> tendencia (B, L), con padding replicado en los bordes
        p = (self.k - 1) // 2
        yp = torch.cat([y[:, :1].repeat(1, p), y, y[:, -1:].repeat(1, self.k - 1 - p)], 1)
        return self.pool(yp[:, None]).squeeze(1)


class DLinear(_Base):
    """Zeng et al. (2023): descompone en tendencia (media móvil) + estacional y aplica una
    capa lineal L->H a cada componente. Solo usa la serie objetivo (sin covariables)."""
    def __init__(self, L, H, n_cal=0, kernel=25, revin=False):
        super().__init__(revin)
        self.ma = MovingAvg(kernel)
        self.lin_t = nn.Linear(L, H); self.lin_s = nn.Linear(L, H)

    def forward(self, x, xf, y_teacher=None):
        y = self._in(x)[..., 0]
        tr = self.ma(y)
        return self._out(self.lin_t(tr) + self.lin_s(y - tr))


class NLinear(_Base):
    """NLinear: resta el último valor (normalización simple), Linear L->H y lo suma de vuelta."""
    def __init__(self, L, H, n_cal=0, revin=False):
        super().__init__(revin)
        self.lin = nn.Linear(L, H)

    def forward(self, x, xf, y_teacher=None):
        y = self._in(x)[..., 0]
        last = y[:, -1:]
        return self._out(self.lin(y - last) + last)


class LSTMMultiOutput(_Base):
    """Estrategia multi-salida: LSTM codifica la historia; una capa densa sobre h_T y el
    calendario futuro devuelve los 24 pasos de una vez."""
    def __init__(self, L, H, n_cal, hidden=64, layers=1, dropout=0.0, revin=False, cell="LSTM"):
        super().__init__(revin)
        R = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = R(1 + n_cal, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden + H * n_cal, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, H))

    def forward(self, x, xf, y_teacher=None):
        out, _ = self.rnn(self._in(x))
        return self._out(self.head(torch.cat([out[:, -1], xf.flatten(1)], 1)))


class LSTMAutoregressive(_Base):
    """Estrategia autorregresiva: modelo de un paso, entrada [y_t, cal_{t+1}] -> y_{t+1}.
    Entrenamiento con teacher forcing (se alimenta el valor real); en inferencia se realimenta
    su propia predicción 24 veces (aquí se acumula el error)."""
    def __init__(self, L, H, n_cal, hidden=64, layers=1, dropout=0.0, revin=False, cell="LSTM"):
        super().__init__(revin)
        R = nn.LSTM if cell == "LSTM" else nn.GRU
        self.H = H
        self.rnn = R(1 + n_cal, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, xf, y_teacher=None):
        x = self._in(x)
        y_hist, cal_hist = x[..., :1], x[..., 1:]
        cal_next = torch.cat([cal_hist[:, 1:], xf[:, :1]], 1)          # calendario de t+1
        if y_teacher is not None and self.training:
            yt = y_teacher
            if self.revin is not None:  # el objetivo se lleva a la misma escala que la entrada
                yt = (y_teacher - self.revin.mu) / self.revin.sd * self.revin.w + self.revin.b
            seq_y = torch.cat([y_hist, yt[:, :-1, None]], 1)                # teacher forcing
            seq_c = torch.cat([cal_next, xf[:, 1:]], 1)
            out, _ = self.rnn(torch.cat([seq_y, seq_c], -1))
            pred = self.head(out[:, -self.H:]).squeeze(-1)
            return self._out(pred)
        out, st = self.rnn(torch.cat([y_hist, cal_next], -1))
        y_prev = self.head(out[:, -1:])                                    # \hat y_{t}
        preds = [y_prev]
        for h in range(1, self.H):
            o, st = self.rnn(torch.cat([y_prev, xf[:, h:h + 1]], -1), st)
            y_prev = self.head(o); preds.append(y_prev)
        return self._out(torch.cat(preds, 1).squeeze(-1))


class Seq2Seq(_Base):
    """Encoder-decoder: el encoder LSTM resume la historia; el decoder LSTM recibe en cada
    paso las covariables de calendario de la hora objetivo (+ opcionalmente la predicción
    previa con teacher forcing) e inicia desde el estado final del encoder."""
    def __init__(self, L, H, n_cal, hidden=64, layers=1, dropout=0.0, revin=False,
                 feed_prev=False, tf_ratio=0.5, cell="LSTM"):
        super().__init__(revin)
        R = nn.LSTM if cell == "LSTM" else nn.GRU
        self.H, self.feed_prev, self.tf_ratio = H, feed_prev, tf_ratio
        d = dropout if layers > 1 else 0
        self.enc = R(1 + n_cal, hidden, layers, batch_first=True, dropout=d)
        self.dec = R(n_cal + (1 if feed_prev else 0), hidden, layers, batch_first=True, dropout=d)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, xf, y_teacher=None):
        x = self._in(x)
        _, st = self.enc(x)
        if not self.feed_prev:                         # decoder no autorregresivo (sin acumulación)
            out, _ = self.dec(xf, st)
            return self._out(self.head(out).squeeze(-1))
        y_prev, preds = x[:, -1:, :1], []
        for h in range(self.H):
            o, st = self.dec(torch.cat([y_prev, xf[:, h:h + 1]], -1), st)
            p = self.head(o); preds.append(p)
            use_tf = self.training and y_teacher is not None and torch.rand(()) < self.tf_ratio
            y_prev = y_teacher[:, h:h + 1, None] if use_tf else p
        return self._out(torch.cat(preds, 1).squeeze(-1))


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


REGISTRY = dict(MLP=MLP, DLinear=DLinear, NLinear=NLinear, LSTM_MO=LSTMMultiOutput,
                LSTM_AR=LSTMAutoregressive, Seq2Seq=Seq2Seq)
