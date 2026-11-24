"""Compare GEDI QC schemes on stable-forest V3 footprints (common.schemes): how many footprints and 1 km cells each
keeps, how repeatable the RH is between footprints <= 10 m apart from different orbits, the regional mean AGB it implies
(local Gamma calibration), and random-forest accuracy under spatial-block vs random cross-validation (the SAR
filter uses the predictors to pick the targets, so random CV is expected to look better than it is).
Also: RH difference between power and coverage beams, night and day, high and low sensitivity at the same place
(footprint pairs <= 10 m apart; all 36 calibration footprints are night-time power-beam shots with sensitivity_a2 > 0.98),
and predictors and Hansen loss at the plot centres (disturbance between GEDI shot and field date).
Output: out/study/qc_schemes.csv, pair_bias.csv, plots_predictors.csv, fig_qc.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FEATURES, METRIC, POWER, STUDY, TRAIN, agb, block_folds, calibration_sets, plot_predictors, schemes,
                    stable_forest)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(18)
N_RF = 100_000


def repeatability(d, m, r=10):
    xy = d[["x", "y"]].to_numpy()
    pr = cKDTree(xy).query_pairs(r, output_type="ndarray")
    g = d.granule.to_numpy()
    pr = pr[g[pr[:, 0]] != g[pr[:, 1]]]
    if len(pr) < 30:
        return dict(pairs=len(pr), rho=np.nan, r=np.nan, sd_robust=np.nan)
    v = d[m].to_numpy(float)
    a1, a2 = v[pr[:, 0]], v[pr[:, 1]]
    return dict(pairs=len(pr), rho=spearmanr(a1, a2).statistic, r=np.corrcoef(np.r_[a1, a2], np.r_[a2, a1])[0, 1],
                sd_robust=1.4826 * np.median(np.abs(a1 - a2)) / np.sqrt(2))


def rf_cv(d, target, folds):
    X = d[FEATURES].to_numpy(np.float32)
    pred = np.empty(len(d))
    for k in np.unique(folds):
        te = folds == k
        rf = RandomForestRegressor(100, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=0)
        rf.fit(X[~te], target[~te])
        pred[te] = rf.predict(X[te])
    e = pred - target
    return dict(r2=1 - (e ** 2).sum() / ((target - target.mean()) ** 2).sum(), rmse=np.sqrt((e ** 2).mean()))


def pair_bias(s, m, r=10):
    """Within V3-flag footprints <= r m apart from different orbits: RH difference between the two kinds of footprint
    for power vs coverage beam, night vs day, sensitivity_a2 > 0.98 vs not."""
    xy = s[["x", "y"]].to_numpy()
    pr = cKDTree(xy).query_pairs(r, output_type="ndarray")
    g = s.granule.to_numpy()
    pr = pr[g[pr[:, 0]] != g[pr[:, 1]]]
    v = s[m].to_numpy(float)
    out = []
    for name, f in (("power minus coverage beam", s.beam.isin(POWER).to_numpy()), ("night minus day", (s.solar_elevation < 0).to_numpy()),
                    ("sensitivity_a2 > 0.98 minus <= 0.98", (s.sensitivity_a2 > 0.98).to_numpy())):
        a, b = pr[:, 0], pr[:, 1]
        mix = f[a] != f[b]
        i, j = np.where(f[a], a, b)[mix], np.where(f[a], b, a)[mix]
        dv = v[i] - v[j]
        out.append(dict(contrast=name, pairs=int(mix.sum()), median_diff_m=np.median(dv), mean_diff_m=dv.mean(),
                        share_first_higher=np.mean(dv > 0)))
    return out


rows, bias_rows = [], []
for region in ["NE", "SW"]:
    m = METRIC[region]
    d = pq.read_table(TRAIN / f"{region}.parquet").to_pandas()
    d = d[stable_forest(d)].reset_index(drop=True)
    params = calibration_sets(region, n_boot=1)["main"]
    d["agb_local"] = agb(params, d[m])
    sch = schemes(d, m)
    bias_rows += [dict(region=region, metric=m, **r) for r in pair_bias(d[sch["v3_flag"].to_numpy()].reset_index(drop=True), m)]
    print(pd.DataFrame(bias_rows).round(3).to_string(index=False), flush=True)
    cell = (np.floor(d.x / 1000).astype(np.int64) * 100000 + np.floor(d.y / 1000).astype(np.int64))
    n_cells = cell[sch["v2_flag"] | sch["v3_flag"] | sch["conventional"]].nunique()
    print(f"{region}: {len(d):,} stable-forest footprints, {n_cells:,} 1 km cells", flush=True)
    for name, mask in sch.items():
        s = d[mask.to_numpy()].reset_index(drop=True)
        rec = dict(region=region, scheme=name, shots=len(s), cells_1km=cell[mask.to_numpy()].nunique() / n_cells,
                   mean_rh=s[m].mean(), mean_agb_local=s.agb_local.mean(),
                   mean_l4a=s.agbd[s.l4a_quality_flag_rel3.eq(1)].mean())
        rec.update(repeatability(s, m))
        sub = s.sample(min(N_RF, len(s)), random_state=1).reset_index(drop=True)
        t = sub.agb_local.to_numpy()
        sp = rf_cv(sub, t, block_folds(sub.x.to_numpy(), sub.y.to_numpy()))
        rn = rf_cv(sub, t, rng.integers(0, 5, len(sub)))
        rec.update(rf_r2_block=sp["r2"], rf_rmse_block=sp["rmse"], rf_r2_random=rn["r2"], rf_rmse_random=rn["rmse"])
        rows.append(rec)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    del d
q = pd.DataFrame(rows)
q.to_csv(STUDY / "qc_schemes.csv", index=False)
pd.DataFrame(bias_rows).to_csv(STUDY / "pair_bias.csv", index=False)

# ------------------------------------------------------------------ predictors at the plot centres
pp = pd.concat([plot_predictors(r) for r in ["NE", "SW"]], ignore_index=True)
pp.to_csv(STUDY / "plots_predictors.csv", index=False)
print(pp[["plot", "region", "wc_tree", "tc2000", "lossyear"]].to_string(index=False), flush=True)

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for k, (col, lab) in enumerate([("shots", "footprints (millions)"), ("rho", "repeatability (Spearman, <=10 m pairs)"),
                                ("rf_r2_block", "RF R2, spatial-block CV")]):
    for j, region in enumerate(["NE", "SW"]):
        r = q[q.region == region]
        v = r[col] / (1e6 if col == "shots" else 1)
        ax[k].barh(np.arange(len(r)) + 0.4 * j, v, height=0.4, label=region)
    ax[k].set_yticks(np.arange(len(r)) + 0.2, r.scheme)
    ax[k].set_xlabel(lab)
ax[0].legend()
for i, region in enumerate(["NE", "SW"]):
    r = q[q.region == region]
    ax[2].scatter(r.rf_r2_random, np.arange(len(r)) + 0.4 * i, marker="|", s=200, color="k")
ax[2].set_title("bars: spatial CV; ticks: random CV", fontsize=9)
fig.tight_layout(); fig.savefig(STUDY / "fig_qc.png", dpi=200); plt.close(fig)
