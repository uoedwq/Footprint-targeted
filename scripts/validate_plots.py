"""Full-chain leave-one-plot-out validation of the AGB map at the field plots, for one or more QC schemes.

For each calibration plot i: refit the local Gamma calibration without plot i, recompute the footprint AGB, train the
random forest, and predict the 25 m cell of plot i, so that plot i is not used anywhere in its own prediction (the usual
leave-one-plot-out practice in forest remote sensing; footprints near the plot are kept unless EXCL_M is set). Plots not
used for calibration are predicted by the chain built from all calibration plots. For contrast, the full model (all plots, no exclusion) is
also evaluated at every plot.
Usage: python validate_plots.py NE v3_flag [conventional conventional_sar ...]   (MAPCFG=1: same RF configuration as the maps; files *_mapcfg)
Output: out/study/{region}_lopo.csv (per plot), {region}_lopo_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FEATURES, METRIC, STUDY, TRAIN, agb, calibration_plots, gamma_fit, plot_predictors, schemes,
                    stable_forest)

import os
REGION, SCHEMES = sys.argv[1], sys.argv[2:]
MAPCFG = os.environ.get("MAPCFG") == "1"        # same configuration as the final maps (map_agb.py): 400k footprints, 300 trees
N_TRAIN = 400_000 if MAPCFG else 150_000
EXCL = float(os.environ.get("EXCL_M", 0))        # optional exclusion of footprints near the left-out plot (default: none)
RF = dict(n_estimators=300 if MAPCFG else 150, min_samples_leaf=5, max_features=0.33, max_samples=0.5, n_jobs=-1, random_state=0)
TAG = "_mapcfg" if MAPCFG else ""
m = METRIC[REGION]

d0 = pq.read_table(TRAIN / f"{REGION}.parquet").to_pandas()
d0 = d0[stable_forest(d0)].reset_index(drop=True)
pp = plot_predictors(REGION)
cal = calibration_plots(REGION)
xc, yc = cal[f"v1_{m}"].to_numpy(float), cal.agb_mgha.to_numpy(float)


def chain(d, params, exclude_xy=None, seed=0):
    t = d
    if exclude_xy is not None and EXCL > 0:
        t = d[np.hypot(d.x - exclude_xy[0], d.y - exclude_xy[1]) > EXCL]
    t = t.sample(min(N_TRAIN, len(t)), random_state=seed)
    return RandomForestRegressor(**RF).fit(t[FEATURES].to_numpy(np.float32), agb(params, t[m]))


rows, summ = [], []
for scheme in SCHEMES:
    d = d0[schemes(d0, m)[scheme].to_numpy()].reset_index(drop=True)
    full = chain(d, gamma_fit(xc, yc))
    allp = gamma_fit(xc, yc)
    for r in pp.itertuples():
        xp = np.array([[getattr(r, p) for p in FEATURES]], np.float32)
        if r.calibration and r.plot in set(cal["plot"]):
            i = int(np.flatnonzero(cal["plot"].to_numpy() == r.plot)[0])
            params = gamma_fit(np.delete(xc, i), np.delete(yc, i))
        else:
            params = allp
        rf = chain(d, params, (r.x, r.y), seed=len(rows))
        rows.append(dict(region=REGION, scheme=scheme, plot=r.plot, calibration=r.calibration, agb_field=r.agb_mgha,
                         lopo=rf.predict(xp)[0], full_model=full.predict(xp)[0], wc_tree=r.wc_tree, lossyear=r.lossyear))
        print(rows[-1], flush=True)
    s = pd.DataFrame([x for x in rows if x["scheme"] == scheme])
    for subset, g in (("calibration plots", s[s.calibration]), ("all plots", s)):
        for est in ("lopo", "full_model"):
            e = g[est] - g.agb_field
            summ.append(dict(region=REGION, scheme=scheme, plots=subset, estimate=est, n=len(g), bias=e.mean(),
                             rmse=np.sqrt((e ** 2).mean()),
                             r2=1 - (e ** 2).sum() / ((g.agb_field - g.agb_field.mean()) ** 2).sum(),
                             field_mean=g.agb_field.mean()))
out = pd.DataFrame(rows)
out.to_csv(STUDY / f"{REGION}_lopo{TAG}.csv", index=False)
sm = pd.DataFrame(summ)
sm.to_csv(STUDY / f"{REGION}_lopo{TAG}_summary.csv", index=False)
print(sm.round(3).to_string(index=False), flush=True)
