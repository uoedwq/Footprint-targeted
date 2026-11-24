"""Random-forest AGB maps (2021, 25 m) for one region, its uncertainty, and transfer to its test area.

Training: stable-forest V3 footprints passing the chosen QC scheme (common.schemes), AGB from the local Gamma GLM
calibration of the region (common.calibration_sets), at most N_TRAIN random footprints, 22 predictors (common.PREDICTORS).
Accuracy: 5-fold spatial-block CV (20 km blocks) and random CV against footprint AGB, and against 1 km and 5 km cell
means of held-out footprints. Predictors: common.FEATURES (22 + neighbourhood context).
Pixel 90% intervals: split-conformal by predicted-AGB decile, from the spatial-CV residuals (Mondrian conformal).
Map: forest = WorldCover 2021 tree fraction >= 50% inside the AOI; tiles out/maps/{aoi}/tiles/*.tif (int16, AGB x 10:
mean, lower, upper); 1 km means; COG mosaic.
Regional totals and calibration uncertainty: a random forest prediction is a weighted mean of training targets with
weights fixed by the trees, so other calibrations are propagated by recomputing the leaf means with new targets
(plot-bootstrap Gamma fits, five allometries, the L4A V3 curve) on a 200k forest-pixel sample.
Area of applicability (Meyer & Pebesma 2021) on pixel samples of the region and the test area.
Usage: python map_agb.py NE v3_flag
Output: out/study/{region}_*.csv, out/maps/{region}/, out/maps/{test}/
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FEATURES, MAPS, METRIC, PRED, STUDY, TRAIN, agb, block_folds, calibration_sets, predictor_vrt, schemes,
                    stable_forest, tile_feature_arrays, FOREST_WC)
PREDICTORS = FEATURES                      # 22 predictors + neighbourhood context (common.CTX)

REGION, SCHEME = sys.argv[1], sys.argv[2]
TEST = REGION + "test"
SHP = {"NE": "Studyarea_Northeast_500km_forGEE", "SW": "Studyarea_Southwest_500km_forGEE",
       "NEtest": "Studyarea_Northeast_100km_testNE_ForGEE", "SWtest": "Studyarea_Southwest_100km_testSW2_ForGEE"}
ROOT = Path(__file__).resolve().parents[1]
import os
QUICK = os.environ.get("QUICK") == "1"                     # smoke test: small model, few tiles, nothing overwritten
N_TRAIN, N_PIX = (20_000, 5_000) if QUICK else (400_000, 200_000)
RF = dict(n_estimators=20 if QUICK else 300, min_samples_leaf=5, max_features=0.33, max_samples=0.5, n_jobs=-1,
          random_state=0)
if QUICK:
    STUDY, MAPS = STUDY / "_quick", MAPS / "_quick"
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
cal = calibration_sets(REGION, n_boot=200)
d["target"] = agb(cal["main"], d[m])
tr = d.sample(min(N_TRAIN, len(d)), random_state=1).reset_index(drop=True)
X = tr[PREDICTORS].to_numpy(np.float32)
y = tr.target.to_numpy()
log(f"{REGION} {SCHEME}: {len(d):,} footprints, training on {len(tr):,}; calibration a={cal['main'][0]:.3f} b={cal['main'][1]:.3f}")

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
log(cv.round(3).to_string(index=False))

edges = np.unique(np.quantile(oof, np.linspace(0, 1, 11)))
kb = np.clip(np.searchsorted(edges, oof, side="right") - 1, 0, len(edges) - 2)
res = y - oof
conf = pd.DataFrame([dict(bin=b, lo_edge=edges[b], hi_edge=edges[b + 1], n=(kb == b).sum(),
                          q05=np.quantile(res[kb == b], 0.05), q95=np.quantile(res[kb == b], 0.95)) for b in range(len(edges) - 1)])
# coverage check: table from folds 0-2, coverage on folds 3-4
cal_f = folds < 3
cov = []
for b in range(len(edges) - 1):
    s_cal, s_val = (kb == b) & cal_f, (kb == b) & ~cal_f
    if s_cal.sum() > 20 and s_val.sum() > 20:
        lo, hi = np.quantile(res[s_cal], [0.05, 0.95])
        cov.append(((res[s_val] >= lo) & (res[s_val] <= hi)).mean())
conf["coverage_check"] = np.mean(cov)
conf.to_csv(STUDY / f"{REGION}_conformal.csv", index=False)
log(f"conformal 90% coverage on held-out folds: {np.mean(cov):.3f}")

# ------------------------------------------------------------------ final model
rf = RandomForestRegressor(**RF).fit(X, y)
imp = pd.DataFrame(dict(predictor=PREDICTORS, importance=rf.feature_importances_)).sort_values("importance", ascending=False)
imp.to_csv(STUDY / f"{REGION}_importance.csv", index=False)
log("trained final model; top predictors: " + ", ".join(imp.predictor.head(6)))


def interval(p):
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return np.clip(p + conf.q05.to_numpy()[b], 0, None), p + conf.q95.to_numpy()[b]


# ------------------------------------------------------------------ maps
def predict_area(aoi, model, sample_n):
    tl = pd.read_csv(PRED / aoi / "tiles.csv")
    crs = tl.crs.iloc[0]
    poly = gpd.read_file(ROOT / f"0_data/3_vector/{SHP[aoi]}.shp").to_crs(crs).union_all()
    out = MAPS / aoi / "tiles"
    out.mkdir(parents=True, exist_ok=True)
    X0, Y0 = tl.x0.min(), tl.y0.max()
    nr, nc = tl.row.max() + 1, tl.col.max() + 1
    km_sum = np.zeros((nr * 16, nc * 16)); km_n = np.zeros_like(km_sum); km_cells = np.zeros_like(km_sum)
    km_w = np.zeros_like(km_sum)
    tot, npx, samples = 0.0, 0, []
    vrt = rasterio.open(predictor_vrt(aoi))
    for i, t in enumerate(tl.itertuples()):
        if QUICK and i % max(1, len(tl) // 12):
            continue
        fa = tile_feature_arrays(vrt, t.x0, t.y0)
        a = np.stack([fa[p] for p in PREDICTORS])
        prof = dict(driver="GTiff", width=TILE, height=TILE, crs=crs, transform=from_origin(t.x0, t.y0, PX, PX), compress="deflate",
                    predictor=2, tiled=True, blockxsize=256, blockysize=256)
        inside = geometry_mask([poly], out_shape=(TILE, TILE), transform=from_origin(t.x0, t.y0, PX, PX), invert=True)
        valid = np.all(np.isfinite(a), axis=0)
        forest = inside & valid & (fa["wc_tree"] >= FOREST_WC)
        mean = np.full((TILE, TILE), np.nan)
        width = np.full((TILE, TILE), np.nan)
        if forest.any():
            Xp = a[:, forest].T.astype(np.float32)
            p = model.predict(Xp)
            mean[forest] = p
            lo, hi = interval(p)
            width[forest] = hi - lo
            outa = np.full((3, TILE, TILE), -1, np.int16)
            outa[0][forest], outa[1][forest], outa[2][forest] = np.round(p * 10), np.round(lo * 10), np.round(np.clip(hi, 0, 3276) * 10)
            prof2 = prof | dict(count=3, dtype="int16", nodata=-1)
            with rasterio.open(out / t.file, "w", **prof2) as dst:
                dst.write(outa)
                dst.descriptions = ("agb_x10", "agb_p05_x10", "agb_p95_x10")
            tot += p.sum()
            npx += forest.sum()
            k = min(sample_n // len(tl) + 1, forest.sum())
            pick = rng.choice(np.flatnonzero(forest.ravel()), k, replace=False)
            samples.append(a.reshape(len(PREDICTORS), -1)[:, pick].T)
        # 1 km aggregation (40 x 40 pixels)
        blk = mean.reshape(16, 40, 16, 40)
        r0, c0 = t.row * 16, t.col * 16
        km_sum[r0:r0 + 16, c0:c0 + 16] += np.nansum(blk, axis=(1, 3))
        km_n[r0:r0 + 16, c0:c0 + 16] += np.isfinite(blk).sum(axis=(1, 3))
        km_w[r0:r0 + 16, c0:c0 + 16] += np.nansum(width.reshape(16, 40, 16, 40), axis=(1, 3))
        km_cells[r0:r0 + 16, c0:c0 + 16] += inside.reshape(16, 40, 16, 40).sum(axis=(1, 3))
        if (i + 1) % 100 == 0:
            log(f"{aoi}: {i + 1}/{len(tl)} tiles")
    with np.errstate(invalid="ignore"):
        km_mean = np.where(km_n > 0, km_sum / km_n, np.nan).astype(np.float32)
        km_width = np.where(km_n > 0, km_w / km_n, np.nan).astype(np.float32)
    kprof = dict(driver="GTiff", height=km_mean.shape[0], width=km_mean.shape[1], count=4, dtype="float32", crs=crs,
                 transform=from_origin(X0, Y0, 1000, 1000), nodata=np.nan, compress="deflate")
    with rasterio.open(MAPS / aoi / f"{aoi}_agb_1km.tif", "w", **kprof) as dst:
        dst.write(np.stack([km_mean, (km_n / 1600).astype(np.float32), (km_cells / 1600).astype(np.float32), km_width]))
        dst.descriptions = ("agb_mean_forest", "forest_fraction", "aoi_fraction", "pi90_width_mean")
    vrt = MAPS / aoi / f"{aoi}_agb_25m.vrt"
    files = sorted(str(p) for p in out.glob("t_*.tif"))
    bin_ = Path(sys.executable).parent
    subprocess.run([str(bin_ / "gdalbuildvrt"), "-q", str(vrt)] + files, check=True)
    subprocess.run([str(bin_ / "gdal_translate"), "-q", "-of", "COG", "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
                    "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS", str(vrt), str(MAPS / aoi / f"{aoi}_agb_25m_cog.tif")],
                   check=True)
    return dict(aoi=aoi, forest_ha=npx * 0.0625, mean_agb=tot / max(npx, 1), total_tg=tot * 0.0625 / 1e6), np.vstack(samples)


summary, pix = predict_area(REGION, rf, N_PIX)
log(f"{REGION} map: {summary}")
HAVE_TEST = (PRED / TEST / "tiles.csv").exists() and (TRAIN / f"{TEST}.parquet").exists()
if HAVE_TEST:
    summary_t, pix_t = predict_area(TEST, rf, 50_000)
    log(f"{TEST} map: {summary_t}")
else:
    summary_t, pix_t = {}, pix[:10]
    log(f"{TEST}: predictors or footprints missing, test area skipped")

# ------------------------------------------------------------------ calibration uncertainty by leaf re-weighting
Lt = rf.apply(X).astype(np.int32)
Lp = rf.apply(pix.astype(np.float32)).astype(np.int32)
sizes = [e.tree_.node_count for e in rf.estimators_]


def reweighted_mean(target):
    acc = np.zeros(len(Lp))
    for j in range(len(sizes)):
        s = np.bincount(Lt[:, j], weights=target, minlength=sizes[j])
        c = np.bincount(Lt[:, j], minlength=sizes[j])
        acc += (s / np.maximum(c, 1))[Lp[:, j]]
    return (acc / len(sizes)).mean()


base = reweighted_mean(y)
q = tr[(tr.l4a_quality_flag_rel3 == 1) & tr.agbd.gt(0) & tr[m].gt(1)]
b, a = np.polyfit(np.log(q[m]), np.log(q.agbd), 1)
cf = q.agbd.mean() / np.exp(a + b * np.log(q[m])).mean()
alts = {"L4A V3 curve": np.exp(np.log(cf) + a + b * np.log(np.clip(tr[m].to_numpy(float), 1, None)))}
alts.update({f"allometry {k}": agb(v, tr[m]) for k, v in cal["allometry"].items()})
rows = [dict(set="main (local Gamma GLM)", ratio=1.0)]
rows += [dict(set=k, ratio=reweighted_mean(v) / base) for k, v in alts.items()]
boot = np.array([reweighted_mean(agb(p, tr[m])) / base for p in cal["boot"]])
rows += [dict(set="plot bootstrap p2.5", ratio=np.percentile(boot, 2.5)), dict(set="plot bootstrap p50", ratio=np.median(boot)),
         dict(set="plot bootstrap p97.5", ratio=np.percentile(boot, 97.5))]
tot = pd.DataFrame(rows)
tot["mean_agb"] = summary["mean_agb"] * tot.ratio
tot["total_tg"] = summary["total_tg"] * tot.ratio
tot.insert(0, "region", REGION)
tot["forest_mha"] = summary["forest_ha"] / 1e6
tot["direct_vs_reweighted"] = rf.predict(pix.astype(np.float32)).mean() / base
tot.to_csv(STUDY / f"{REGION}_totals.csv", index=False)
log(tot.round(3).to_string(index=False))

# ------------------------------------------------------------------ test area: accuracy against its own GEDI footprints
if HAVE_TEST:
    dt = pq.read_table(TRAIN / f"{TEST}.parquet").to_pandas()
    dt = dt[stable_forest(dt)].reset_index(drop=True)
    dt = dt[schemes(dt, m)[SCHEME].to_numpy()].reset_index(drop=True)
    yt = agb(cal["main"], dt[m])
    pt = rf.predict(dt[PREDICTORS].to_numpy(np.float32))
    tt = pd.DataFrame([dict(region=REGION, area=TEST, **metrics(yt, pt))])
else:
    tt = pd.DataFrame([dict(region=REGION, area=TEST)])


# ------------------------------------------------------------------ area of applicability
def aoa(Xtr, folds_tr, weights, *Xnew):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd * weights
    sub = rng.choice(len(Z), min(20_000, len(Z)), replace=False)
    dbar = np.mean(cKDTree(Z[sub]).query(Z[rng.choice(len(Z), 2000, replace=False)], k=2)[0][:, 1]) * 1.0
    # training DI with the nearest training point from another fold
    di_tr = np.empty(len(sub))
    for k in np.unique(folds_tr):
        a_ = sub[folds_tr[sub] == k]
        b_ = sub[folds_tr[sub] != k]
        di_tr[folds_tr[sub] == k] = cKDTree(Z[b_]).query(Z[a_])[0] / dbar
    q1, q3 = np.percentile(di_tr, [25, 75])
    thr = q3 + 1.5 * (q3 - q1)
    tree = cKDTree(Z[sub])
    return thr, [tree.query((xn - mu) / sd * weights)[0] / dbar for xn in Xnew]


thr, (di_reg, di_test) = aoa(X, folds, rf.feature_importances_, pix.astype(np.float32), pix_t.astype(np.float32))
tt["test_area_share_outside_aoa"] = (di_test > thr).mean()
tt["region_share_outside_aoa"] = (di_reg > thr).mean()
tt["aoa_threshold"] = thr
tt["test_mean_agb_map"] = summary_t.get("mean_agb", np.nan)
tt["test_forest_mha"] = summary_t.get("forest_ha", np.nan) / 1e6
tt.to_csv(STUDY / f"{REGION}_test_area.csv", index=False)
log(tt.round(3).to_string(index=False))

json.dump(dict(region=REGION, scheme=SCHEME, n_footprints=len(d), n_train=len(tr), calibration=list(cal["main"]),
               edges=edges.tolist(), summary=summary, summary_test=summary_t), open(STUDY / f"{REGION}_run.json", "w"),
          indent=1, default=float)
log("done")
