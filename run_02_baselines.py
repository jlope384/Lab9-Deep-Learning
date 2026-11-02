import time, sys
from src.prep import get_context
from src.baselines import make_baselines
from src.backtest import backtest
from src import store, config as C
ctx = get_context()
Hs = [24, 168] if len(sys.argv) < 2 or not sys.argv[1].isdigit() else [int(sys.argv[1])]
only = [a for a in sys.argv[1:] if not a.isdigit() and not a.startswith("--")]
for H in Hs:
    for b in make_baselines(ctx, H):
        if only and b.name not in only: continue
        if H == 168 and b.name == "MSTL" and "--mstl168" not in sys.argv: continue
        t = time.time(); res = backtest(b, ctx, "test", H)
        m = store.save(res, dict(family="Baseline", L=b.input_size, strategy="local (directo)",
             covariates="no", params_total=0, params_trainable=0, train_s=0.0,
             hardware=C.hardware_str(), impl="StatsForecast"))
        print(H, b.name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, round(time.time()-t,1), flush=True)
