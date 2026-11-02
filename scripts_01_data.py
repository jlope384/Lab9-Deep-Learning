import time, json
t=time.time()
from src import config as C, data as D
import pandas as pd, numpy as np
panel, info = D.build_panel()
pa = info["panel_all"]
out = {
 "raw_shape": list(info.get("raw_shape", ())),
 "n_clients_after_filter": int(pa.shape[1]),
 "n_clients_used": int(panel.shape[1]),
 "hours_per_series": int(panel.shape[0]),
 "panel_obs_all": int(pa.size), "panel_obs_used": int(panel.size),
 "dst_log": info["dst_log"],
 "excluded_reasons": info["selection"][~info["selection"].keep][["first_day_ok","no_nan","max_zero_run_days"]].describe().to_dict(),
}
print(json.dumps(out, indent=1, default=str)); print("time", time.time()-t)
json.dump(out, open(C.RES_DIR/"data_summary.json","w"), indent=1, default=str)
