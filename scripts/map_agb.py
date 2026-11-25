"""Random-forest AGB map (2021, 25 m) of one region and its uncertainty.

Training: stable-forest V3 footprints passing the chosen QC scheme (common.schemes), AGB from the local Gamma GLM calibration of the region
(common.calibration_sets), at most N_TRAIN random footprints, predictors common.FEATURES (22 + neighbourhood context).
Accuracy: 5-fold spatial-block CV (20 km blocks) and random CV against footprint AGB, and against 1 km and 5 km cell means of held-out footprints.
Pixel 90% intervals: split-conformal by predicted-AGB decile, from the spatial-CV residuals (Mondrian conformal). They describe the scatter of footprint
AGB (local calibration) around the map; the uncertainty of the calibration enters the regional means below.
Map: forest = WorldCover 2021 tree fraction >= 50% inside the AOI; tiles out/maps/{aoi}/tiles/*.tif (int16, AGB x 10: mean, lower, upper); 1 km means;
COG mosaic.
Regional totals and calibration uncertainty: a random-forest prediction is a weighted mean of training targets; the weights are fixed by the trees and by
the bootstrap sample (with multiplicities) that each tree was trained on. While the map is predicted, the number of forest cells in every leaf is counted
over ALL forest cells, which gives the weight of every training footprint in the regional mean. The regional mean under other targets (5,000
plot-bootstrap Gamma fits, five allometries, the L4A V3 curve) is then exact for the same trees; the forest is also retrained with the L4A V3 targets.
Usage: python map_agb.py NE v3_strict          (QUICK=1: smoke test with a small model and a few tiles in _quick/)
Output: out/study/{region}_*.csv, {region}_weights.npz, out/maps/{region}/
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FEATURES, FOREST_WC, MAPS, METRIC, PRED, STUDY, TRAIN, agb, block_folds, calibration_sets, predictor_vrt, schemes, stable_forest,
                    tile_feature_arrays)
PREDICTORS = FEATURES                      # 22 predictors + neighbourhood context (common.CTX)

REGION, SCHEME = sys.argv[1], sys.argv[2]
SHP = {"NE": "Studyarea_Northeast_500km_forGEE", "SW": "Studyarea_Southwest_500km_forGEE"}
ROOT = Path(__file__).resolve().parents[1]
NP = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
QUICK = os.environ.get("QUICK") == "1"                     # smoke test: small model, few tiles, nothing overwritten
N_TRAIN, N_BOOT = (20_000, 50) if QUICK else (400_000, 5000)
RF = dict(n_estimators=20 if QUICK else 300, min_samples_leaf=5, max_features=0.33, max_samples=0.5, n_jobs=NP, random_state=0)
if QUICK:
    STUDY_IN, STUDY, MAPS = STUDY, STUDY / "_quick", MAPS / "_quick"
    STUDY.mkdir(parents=True, exist_ok=True)
PX, TILE = 25, 640
rng = np.random.default_rng(19)
m = METRIC[REGION]
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)


def metrics(y, p):
    e = p - y
    return dict(n=len(y), bias=e.mean(), rmse=np.sqrt((e ** 2).mean()), r2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                mean_obs=y.mean())


# ------------------------------------------------------------------ training data
d = pq.read_table(TRAIN / f"{REGION}.parquet").to_pandas()
d = d[stable_forest(d)].reset_index(drop=True)
d = d[schemes(d, m)[SCHEME].to_numpy()].reset_index(drop=True)
cal = calibration_sets(REGION, n_boot=N_BOOT)
d["target"] = agb(cal["main"], d[m])
tr = d.sample(min(N_TRAIN, len(d)), random_state=1).reset_index(drop=True)
n_footprints = len(d)
del d
X = tr[PREDICTORS].to_numpy(np.float32)
y = tr.target.to_numpy()
log(f"{REGION} {SCHEME}: {n_footprints:,} footprints, training on {len(tr):,}; {len(cal['plots'])} calibration plots, a={cal['main'][0]:.3f} b={cal['main'][1]:.3f}")

# ------------------------------------------------------------------ cross-validation and conformal table
folds = block_folds(tr.x.to_numpy(), tr.y.to_numpy())
oof = np.empty(len(tr))
for k in range(5):
    te = folds == k
    oof[te] = RandomForestRegressor(**RF).fit(X[~te], y[~te]).predict(X[te])
rnd = rng.integers(0, 5, len(tr))
oof_r = np.empty(len(tr))
for k in range(5):
    te = rnd == k
    oof_r[te] = RandomForestRegressor(**RF).fit(X[~te], y[~te]).predict(X[te])


def agg_metrics(size, min_n=3):
    """Spatial-CV accuracy for cell means of held-out footprints (cells with >= min_n footprints)."""
    k = np.floor(tr.x.to_numpy() / size).astype(np.int64) * 10 ** 6 + np.floor(tr.y.to_numpy() / size).astype(np.int64)
    g = pd.DataFrame(dict(k=k, o=y, p=oof)).groupby("k").agg(o=("o", "mean"), p=("p", "mean"), n=("o", "size"))
    g = g[g.n >= min_n]
    return dict(cv=f"spatial blocks 20 km, {size // 1000} km cell means", **metrics(g.o.to_numpy(), g.p.to_numpy()))


cv = pd.DataFrame([dict(cv="spatial blocks 20 km", **metrics(y, oof)), dict(cv="random", **metrics(y, oof_r)),
                   agg_metrics(1000), agg_metrics(5000)])
cv["slope_obs_on_pred"] = np.polyfit(oof, y, 1)[0]
cv.insert(0, "region", REGION)
cv.to_csv(STUDY / f"{REGION}_cv.csv", index=False)
log("\n" + cv.round(3).to_string(index=False))

edges = np.unique(np.quantile(oof, np.linspace(0, 1, 11)))
kb = np.clip(np.searchsorted(edges, oof, side="right") - 1, 0, len(edges) - 2)
res = y - oof
conf = pd.DataFrame([dict(bin=b, lo_edge=edges[b], hi_edge=edges[b + 1], n=(kb == b).sum(),
                          q05=np.quantile(res[kb == b], 0.05), q95=np.quantile(res[kb == b], 0.95)) for b in range(len(edges) - 1)])
# coverage check: table from folds 0-2, coverage on folds 3-4 (all held-out footprints pooled)
hit, nval = 0, 0
for b in range(len(edges) - 1):
    s_cal, s_val = (kb == b) & (folds < 3), (kb == b) & (folds >= 3)
    if s_cal.sum() > 20 and s_val.sum() > 20:
        lo, hi = np.quantile(res[s_cal], [0.05, 0.95])
        hit += ((res[s_val] >= lo) & (res[s_val] <= hi)).sum(); nval += s_val.sum()
conf["coverage_check"] = hit / nval
conf.to_csv(STUDY / f"{REGION}_conformal.csv", index=False)
log(f"conformal 90% coverage on held-out folds: {hit / nval:.4f}")

# ------------------------------------------------------------------ final model, and a forest trained on the L4A V3 curve
rf = RandomForestRegressor(**RF).fit(X, y)
imp = pd.DataFrame(dict(predictor=PREDICTORS, importance=rf.feature_importances_)).sort_values("importance", ascending=False)
imp.to_csv(STUDY / f"{REGION}_importance.csv", index=False)
log("trained final model; top predictors: " + ", ".join(imp.predictor.head(6)))
q = tr[(tr.l4a_quality_flag_rel3 == 1) & tr.agbd.gt(0) & tr[m].gt(1)]
b_l4, a_l4 = np.polyfit(np.log(q[m]), np.log(q.agbd), 1)
cf_l4 = q.agbd.mean() / np.exp(a_l4 + b_l4 * np.log(q[m])).mean()
l4a_curve = lambda rh: np.exp(np.log(cf_l4) + a_l4 + b_l4 * np.log(np.clip(np.asarray(rh, float), 1, None)))
rf2 = RandomForestRegressor(**RF).fit(X, l4a_curve(tr[m]))
T, T2 = [e.tree_ for e in rf.estimators_], [e.tree_ for e in rf2.estimators_]
V, V2 = [t.value[:, 0, 0].copy() for t in T], [t.value[:, 0, 0].copy() for t in T2]
sizes = np.array([t.node_count for t in T])
off = np.r_[0, np.cumsum(sizes)]


def interval(p):
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return np.clip(p + conf.q05.to_numpy()[b], 0, None), p + conf.q95.to_numpy()[b]


# ------------------------------------------------------------------ map
tl = pd.read_csv(PRED / REGION / "tiles.csv")
if QUICK:
    tl = tl.iloc[::max(1, len(tl) // 12)].reset_index(drop=True)
crs = tl.crs.iloc[0]
poly = gpd.read_file(ROOT / f"0_data/3_vector/{SHP[REGION]}.shp").to_crs(crs).union_all()
out = MAPS / REGION / "tiles"
out.mkdir(parents=True, exist_ok=True)
X0, Y0 = tl.x0.min(), tl.y0.max()
nr, nc = tl.row.max() + 1, tl.col.max() + 1
vrt_path = str(predictor_vrt(REGION))
TMP = tempfile.mkdtemp()


def work(w):
    """One process: predicts and writes its tiles; returns the sums for the 1 km grid and saves the number of forest cells in every leaf."""
    vrt = rasterio.open(vrt_path)
    km = np.zeros((4, nr * 16, nc * 16))                                      # sum of AGB, forest cells, cells inside the AOI, sum of interval width
    acc = np.zeros(off[-1], np.int32)
    tot, tot2, npx = 0.0, 0.0, 0
    for i in range(w, len(tl), NP):
        t = tl.iloc[i]
        fa = tile_feature_arrays(vrt, t.x0, t.y0)
        a = np.stack([fa[p] for p in PREDICTORS])
        inside = geometry_mask([poly], out_shape=(TILE, TILE), transform=from_origin(t.x0, t.y0, PX, PX), invert=True)
        forest = inside & np.all(np.isfinite(a), axis=0) & (fa["wc_tree"] >= FOREST_WC)
        mean, width = np.full((TILE, TILE), np.nan), np.full((TILE, TILE), np.nan)
        if forest.any():
            Xp = np.ascontiguousarray(a[:, forest].T, dtype=np.float32)
            p, p2 = np.zeros(len(Xp)), np.zeros(len(Xp))
            for j in range(len(T)):
                leaf = T[j].apply(Xp)
                acc[off[j]:off[j + 1]] += np.bincount(leaf, minlength=sizes[j]).astype(np.int32)
                p += V[j][leaf]
                p2 += V2[j][T2[j].apply(Xp)]
            p /= len(T); p2 /= len(T)
            mean[forest] = p
            lo, hi = interval(p)
            width[forest] = hi - lo
            outa = np.full((3, TILE, TILE), -1, np.int16)
            outa[0][forest], outa[1][forest], outa[2][forest] = np.round(p * 10), np.round(lo * 10), np.round(np.clip(hi, 0, 3276) * 10)
            prof = dict(driver="GTiff", width=TILE, height=TILE, crs=crs, transform=from_origin(t.x0, t.y0, PX, PX), compress="deflate", predictor=2, tiled=True,
                        blockxsize=256, blockysize=256, count=3, dtype="int16", nodata=-1)
            with rasterio.open(out / t.file, "w", **prof) as dst:
                dst.write(outa)
                dst.descriptions = ("agb_x10", "agb_p05_x10", "agb_p95_x10")
            tot += p.sum(); tot2 += p2.sum(); npx += len(p)
        s = (slice(t.row * 16, t.row * 16 + 16), slice(t.col * 16, t.col * 16 + 16))            # 1 km aggregation (40 x 40 pixels)
        blk = mean.reshape(16, 40, 16, 40)
        km[0][s] += np.nansum(blk, axis=(1, 3))
        km[1][s] += np.isfinite(blk).sum(axis=(1, 3))
        km[2][s] += inside.reshape(16, 40, 16, 40).sum(axis=(1, 3))
        km[3][s] += np.nansum(width.reshape(16, 40, 16, 40), axis=(1, 3))
    np.save(f"{TMP}/acc_{w}.npy", acc)
    return km, tot, tot2, npx


with mp.get_context("fork").Pool(NP) as pool:
    parts = pool.map(work, range(NP), chunksize=1)
km = sum(p[0] for p in parts)
tot, tot2, npx = (sum(p[k] for p in parts) for k in (1, 2, 3))
N = np.zeros(off[-1], np.int64)
for w in range(NP):
    N += np.load(f"{TMP}/acc_{w}.npy")
    os.remove(f"{TMP}/acc_{w}.npy")
with np.errstate(invalid="ignore"):
    km_mean = np.where(km[1] > 0, km[0] / km[1], np.nan).astype(np.float32)
    km_width = np.where(km[1] > 0, km[3] / km[1], np.nan).astype(np.float32)
kprof = dict(driver="GTiff", height=km_mean.shape[0], width=km_mean.shape[1], count=4, dtype="float32", crs=crs,
             transform=from_origin(X0, Y0, 1000, 1000), nodata=np.nan, compress="deflate")
with rasterio.open(MAPS / REGION / f"{REGION}_agb_1km.tif", "w", **kprof) as dst:
    dst.write(np.stack([km_mean, (km[1] / 1600).astype(np.float32), (km[2] / 1600).astype(np.float32), km_width]))
    dst.descriptions = ("agb_mean_forest", "forest_fraction", "aoi_fraction", "pi90_width_mean")
vrt = MAPS / REGION / f"{REGION}_agb_25m.vrt"
files = sorted(str(p) for p in out.glob("t_*.tif"))
bin_ = Path(sys.executable).parent
subprocess.run([str(bin_ / "gdalbuildvrt"), "-q", str(vrt)] + files, check=True)
subprocess.run([str(bin_ / "gdal_translate"), "-q", "-of", "COG", "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS",
                str(vrt), str(MAPS / REGION / f"{REGION}_agb_25m_cog.tif")], check=True)
summary = dict(aoi=REGION, forest_ha=npx * 0.0625, mean_agb=tot / npx, total_tg=tot * 0.0625 / 1e6)
log(f"{REGION} map: {summary}")

# ------------------------------------------------------------------ weight of every training footprint in the regional mean, other calibrations
Lt = rf.apply(X).astype(np.int32)
W = np.zeros(len(X))
for j, s in enumerate(rf.estimators_samples_):
    c = np.bincount(s, minlength=len(X)).astype(float)                        # how often tree j drew each footprint
    cl = np.bincount(Lt[:, j], weights=c, minlength=sizes[j])
    W += c * np.where(cl > 0, N[off[j]:off[j + 1]] / np.maximum(cl, 1e-9), 0)[Lt[:, j]]
W /= len(T) * npx
assert abs(W.sum() - 1) < 1e-9 and abs(W @ y / summary["mean_agb"] - 1) < 1e-9, (W.sum(), W @ y, summary["mean_agb"])
rh, inv = np.unique(tr[m].to_numpy(float), return_inverse=True)
wrh = np.bincount(inv, weights=W)                                             # weight of every RH value: any target that is a function of RH
mean_of = lambda f: float(wrh @ f(rh))
base = mean_of(lambda t: agb(cal["main"], t))
assert abs(base / summary["mean_agb"] - 1) < 1e-9
rows = [dict(set="main (local Gamma GLM)", mean_agb=base), dict(set="L4A V3 curve", mean_agb=mean_of(l4a_curve))]
rows += [dict(set=f"allometry {k}", mean_agb=mean_of(lambda t: agb(v, t))) for k, v in cal["allometry"].items()]
boot = np.array([mean_of(lambda t: agb(p, t)) for p in cal["boot"]])
rows += [dict(set=f"plot bootstrap p{q_:g}", mean_agb=np.percentile(boot, q_)) for q_ in (2.5, 50, 97.5)]
rows.append(dict(set="L4A V3 curve, retrained forest", mean_agb=tot2 / npx))
tab = pd.DataFrame(rows)
tab.insert(0, "region", REGION)
tab["ratio"] = tab.mean_agb / base
tab["total_tg"] = summary["total_tg"] * tab.ratio
tab["forest_mha"] = summary["forest_ha"] / 1e6
tab.to_csv(STUDY / f"{REGION}_totals.csv", index=False)
log("\n" + tab.round(4).to_string(index=False))
np.savez_compressed(STUDY / f"{REGION}_weights.npz", rh=rh, w=wrh)

json.dump(dict(region=REGION, scheme=SCHEME, n_footprints=n_footprints, n_train=len(tr), n_calibration_plots=len(cal["plots"]), calibration=list(cal["main"]), n_boot=N_BOOT,
               l4a_curve=[float(cf_l4 * np.exp(a_l4)), float(b_l4)], edges=edges.tolist(), summary=summary), open(STUDY / f"{REGION}_run.json", "w"), indent=1, default=float)
log("done")
