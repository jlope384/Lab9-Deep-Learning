"""Entrenamiento de modelos PyTorch: AdamW + pérdida L1 (alineada con MAE/MASE), early
stopping sobre VALIDACIÓN (nunca prueba), registro por epoch y medición de costo."""
import time, json, copy, resource
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from . import config as C
from .data import Scaler
from .windows import WindowDataset
from .backtest import backtest
from .models_torch import REGISTRY, count_params

torch.set_num_threads(max(1, torch.get_num_threads()))
HOURS_PER_MONTH = 730


class TorchForecaster:
    """Adapta un modelo entrenado a la interfaz predict(Y_hist, o) de backtest()."""
    def __init__(self, name, model, scaler, ctx, L, H, device):
        self.name, self.model, self.scaler, self.ctx = name, model, scaler, ctx
        self.L, self.H, self.device = L, H, device

    @torch.no_grad()
    def predict(self, Y_hist, o, idx=None):
        self.model.eval()
        L, H = self.L, self.H
        yn = self.scaler.transform(Y_hist[:, -L:], idx)
        n = yn.shape[0]
        cal_in = np.broadcast_to(self.ctx.cal[o - L:o], (n, L, self.ctx.cal.shape[1]))
        x = np.concatenate([yn[..., None], cal_in], -1)
        xf = np.broadcast_to(self.ctx.cal[o:o + H], (n, H, self.ctx.cal.shape[1]))
        x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
        xf = torch.as_tensor(np.ascontiguousarray(xf), dtype=torch.float32, device=self.device)
        z = self.model(x, xf).cpu().numpy()
        return self.scaler.inverse(z, idx)


def build_model(arch, L, H, n_cal, norm, **kw):
    cls = REGISTRY[arch]
    return cls(L, H, n_cal, revin=(norm == "revin"), **kw)


def train_model(arch, ctx, L=C.L_DEFAULT, H=C.H, norm="series", hist_months=None, lr=1e-3,
                batch_size=256, epochs=12, samples_per_epoch=40_000, patience=3, seed=C.SEED,
                series=None, name=None, verbose=True, t_hi=None, **model_kw):
    """Entrena con ventanas cuyo OBJETIVO termina antes de `t_hi` (por defecto inicio de
    validación) y hace early stopping con los 92 orígenes diarios de validación."""
    C.set_seed(seed)
    dev = C.get_device()
    t_hi = ctx.val_pos if t_hi is None else t_hi
    scaler = Scaler("global" if norm == "revin" else norm).fit(ctx.Y[:, :ctx.val_pos])  # solo train
    Yn = scaler.transform(ctx.Y)
    hist_lo = 0 if hist_months is None else max(0, t_hi - int(hist_months * HOURS_PER_MONTH))
    ds = WindowDataset(Yn, ctx.cal, L, H, 0, t_hi, stride=1, series=series, hist_lo=hist_lo)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size,
                    sampler=RandomSampler(ds, replacement=True, num_samples=min(samples_per_epoch, len(ds)), generator=g))
    model = build_model(arch, L, H, ctx.cal.shape[1], norm, **model_kw).to(dev)
    p_tot, p_tr = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.L1Loss()
    name = name or arch
    fc = TorchForecaster(name, model, scaler, ctx, L, H, dev)
    idx = None if series is None else np.asarray(series)

    class _Sub:  # forecaster restringido a un subconjunto de series (modelo local)
        def predict(self_, Yh, o):
            return fc.predict(Yh[idx], o, idx) if idx is not None else fc.predict(Yh, o)

    hist, best, best_state, bad = [], np.inf, None, 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train(); te = time.perf_counter(); tl = []
        for x, xf, y, _ in dl:
            x, xf, y = x.to(dev), xf.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x, xf, y_teacher=y), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl.append(loss.item())
        ep_s = time.perf_counter() - te
        if t_hi > ctx.val_pos:   # reentrenamiento con ventana expandible: epochs fijos, sin ES
            hist.append(dict(epoch=ep, train_loss=float(np.mean(tl)), epoch_s=ep_s))
            best_state, best = copy.deepcopy(model.state_dict()), float("nan")
            if verbose: print(f"[{name}] ep{ep} train={hist[-1]['train_loss']:.4f} (refit, {ep_s:.1f}s)", flush=True)
            continue
        vres = _val_backtest(_Sub(), ctx, H, idx)
        vm = vres["metrics"]
        hist.append(dict(epoch=ep, train_loss=float(np.mean(tl)), val_loss=vres["val_loss_norm"],
                         val_MAE=vm["MAE"], val_MASE=vm["MASE"], epoch_s=ep_s))
        if verbose:
            print(f"[{name}] ep{ep} train={hist[-1]['train_loss']:.4f} val_loss={vres['val_loss_norm']:.4f} "
                  f"valMAE={vm['MAE']:.2f} valMASE={vm['MASE']:.4f} ({ep_s:.1f}s)", flush=True)
        if vm["MASE"] < best - 1e-4:
            best, best_state, bad = vm["MASE"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_s = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    peak = (torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else None
    info = dict(arch=arch, L=L, H=H, norm=norm, hist_months=hist_months or 30, lr=lr,
                batch_size=batch_size, epochs_run=len(hist), samples_per_epoch=samples_per_epoch,
                seed=seed, model_kw=model_kw, params_total=p_tot, params_trainable=p_tr,
                train_s=train_s, epoch_s_mean=float(np.mean([h["epoch_s"] for h in hist])),
                best_val_MASE=best, best_epoch=int(np.argmin([h.get("val_MASE", np.inf) for h in hist]) + 1),
                gpu_peak_MB=peak,
                cpu_peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                hardware=C.hardware_str(), history=hist)
    fc_out = _Sub() if idx is not None else fc
    fc_out.name = name
    return fc_out, info


def _val_backtest(fc, ctx, H, idx):
    from .metrics import point_metrics
    origins = ctx.origins("val", H)
    P = np.stack([fc.predict(ctx.Y[:, :o], int(o)) for o in origins], 1)
    Y = np.stack([ctx.Y[:, o:o + H] for o in origins], 1)
    sc = ctx.scale
    if idx is not None:
        Y, sc = Y[idx], sc[idx]
    m = point_metrics(Y, P, sc)
    sd = Y.reshape(Y.shape[0], -1).std(1)[:, None, None] + 1e-6
    return dict(metrics=m, val_loss_norm=float((np.abs(P - Y) / sd).mean()))
