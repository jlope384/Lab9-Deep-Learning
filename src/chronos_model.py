"""Chronos-Bolt zero-shot (sin entrenamiento): BaseChronosPipeline.predict_quantiles."""
import time
import numpy as np
import torch
from . import config as C


class ChronosForecaster:
    def __init__(self, model_id="amazon/chronos-bolt-small", H=24, context=512, batch=100):
        from chronos import BaseChronosPipeline
        self.name = "Chronos-Bolt"
        t = time.perf_counter()
        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=str(C.get_device()),
                                                        torch_dtype=torch.float32)
        self.load_s = time.perf_counter() - t
        self.H, self.context, self.batch, self.model_id = H, context, batch, model_id
        self.params_total = sum(p.numel() for p in self.pipe.model.parameters())

    @torch.no_grad()
    def predict(self, Y_hist, o):
        ctx = torch.tensor(Y_hist[:, -self.context:], dtype=torch.float32)
        q, mean = self.pipe.predict_quantiles(ctx, prediction_length=self.H,
                                              quantile_levels=[0.1, 0.5, 0.9])
        q = q.numpy()
        return {"mean": q[..., 1], 0.1: q[..., 0], 0.5: q[..., 1], 0.9: q[..., 2]}
