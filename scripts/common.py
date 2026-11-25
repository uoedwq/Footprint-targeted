"""Shared definitions: calibration (local Gamma GLM), QC schemes, predictors, forest
mask, spatial folds."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "out/study"
TRAIN = ROOT / "in/data/training"
PRED = ROOT / "in/data/predictors"
MAPS = ROOT / "out/maps"
METRIC = {"NE": "rh98", "SW": "rh80", "NEtest": "rh98", "SWtest": "rh80"}
HOME = {"NEtest": "NE", "SWtest": "SW", "NE": "NE", "SW": "SW"}
POWER = ["BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]
PREDICTORS = ["vv", "vh", "vh_vv", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12", "ndvi", "ndmi",
              "hh", "hv", "hv_hh", "elev", "slope", "north", "east"]
ALLOM = ["agb_luo2018_mgha", "agb_ne_alt1_mgha", "agb_ne_alt2_mgha", "agb_sw_alt_mgha", "agb_chave14_wd055_mgha"]
FOREST_WC = 50            # WorldCover 2021 tree fraction of the 25 m cell (%) for "forest"
STABLE_FROM = 17          # footprints in cells with Hansen loss in 2017..2025 are not used for training


# ------------------------------------------------------------------ calibration
def gamma_fit(x, y):
    """Gamma GLM with log link, log E[AGB] = a + b log RH; returns (a, b)."""
    lx = np.log(x)
    b0, a0 = np.polyfit(lx, np.log(y), 1)
    f = lambda p: np.sum(y * np.exp(-(p[0] + p[1] * lx)) + p[0] + p[1] * lx)
    return minimize(f, [a0, b0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-11, maxiter=4000)).x


def agb(params, rh):
    a, b = params
    return np.exp(a + b * np.log(np.clip(np.asarray(rh, float), 1.0, None)))       # RH <= 1 m treated as 1 m


def calibration_plots(region):
    plots = pd.read_csv(STUDY / "plots_loo.csv").merge(pd.read_csv(STUDY / "field_plots.csv")[["plot"] + ALLOM], on="plot")
    m = METRIC[region]
    return plots[(plots.region == region) & plots.calibration & plots[f"v1_{m}"].gt(0)].reset_index(drop=True)


def calibration_sets(region, n_boot=200, seed=1):
    """Main parameters, plot-bootstrap parameters and the five allometries (all Gamma GLM on the RH metric of the region; RH of the targeted V1 shot)."""
    cal = calibration_plots(region)
    m = METRIC[region]
    x, y = cal[f"v1_{m}"].to_numpy(float), cal.agb_mgha.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = []
    while len(boot) < n_boot:
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            boot.append(gamma_fit(x[i], y[i]))
    allo = {k: gamma_fit(x, cal[k].to_numpy(float)) for k in ALLOM}
    return dict(main=gamma_fit(x, y), boot=np.array(boot), allometry=allo, plots=cal)


# ------------------------------------------------------------------ QC schemes
def rel3_with(df, tropical_a2=0.98):
    """l2a_quality_flag_rel3 recomputed (GEDI02 User Guide V3), with a configurable a2 threshold for tropical forest."""
    land = ~df["worldcover_class"].isin([0, 80])
    tropical = (df["pft_class"] == 2) & df["region_class"].isin([4, 5, 6])
    thr = np.where(tropical, tropical_a2, 0.95)
    return ((df["l2_algrunflag"] == 1) & (df["rx_maxamp"] > 8.0 * df["sd_corrected"]) & (df["stale_return_flag"] == 0)
            & (df["surface_flag"] == 1) & ((~land) | ((df["elev_highestreturn"] - df["digital_elevation_model"]) < 150))
            & np.where(land, df["sensitivity_a2"] > thr, df["sensitivity_a2"] > 0.5))


def sar_consistent(df, metric, tol_db=3.0):
    """SAR-consistency filter: within +-3 dB of the binned-median SAR vs RH curve for PALSAR-2 HH, HV and S1 VV, VH."""
    ok = np.ones(len(df), bool)
    rh = df[metric].to_numpy(float)
    bins = np.arange(0, 51, 1.0)
    k = np.clip(np.digitize(rh, bins) - 1, 0, len(bins) - 2)
    for b in ("hh", "hv", "vv", "vh"):
        raw = df[b].to_numpy(float)
        valid = np.isfinite(raw) & (raw != -32768)
        v = raw / 100
        med = pd.Series(v[valid]).groupby(k[valid]).median()
        curve = np.interp(np.arange(len(bins) - 1), med.index.to_numpy(), med.to_numpy())
        ok &= valid & (np.abs(v - curve[k]) <= tol_db)
    return ok


def schemes(df, metric):
    """Boolean masks of the QC schemes compared in the article (all exclude leaf-off shots)."""
    night = df.solar_elevation < 0
    power = df.beam.isin(POWER)
    leafon = df.leaf_off_flag != 1
    rel2 = df.quality_flag_rel2 == 1
    rel3 = df.quality_flag_rel3 == 1
    out = {
        "conventional": rel2 & (df.sensitivity > 0.98) & night & power & leafon,
        "v2_flag": rel2 & leafon,
        "v3_flag": rel3 & leafon,
        "v3_flag_a2_095": pd.Series(rel3_with(df, 0.95), index=df.index) & leafon,
        "v3_strict": rel3 & (df.sensitivity_a2 > 0.98) & night & power & leafon,
    }
    out["conventional_sar"] = out["conventional"] & sar_consistent(df, metric)
    return out


# ------------------------------------------------------------------ masks, folds
def stable_forest(df):
    ly = df.lossyear.to_numpy()
    feats = [c for c in PREDICTORS + ["ndvi_m5", "ndvi_m11", "hv_m5", "hv_m11", "vh_m5", "vh_m11", "hv_sd3", "ndvi_sd3", "tpi11"]
             if c in df]
    ok = np.isfinite(df[feats].to_numpy(np.float32)).all(axis=1) & (df[feats].to_numpy(np.float32) != -32768).all(axis=1)
    return (df.wc_tree.to_numpy() >= FOREST_WC) & ((ly == 0) | (ly < STABLE_FROM)) & ok


def block_folds(x, y, block_m=20000, k=5, seed=7):
    """Spatial cross-validation folds: 20 km blocks assigned at random to k folds."""
    bid = (np.floor(x / block_m).astype(np.int64) * 100000 + np.floor(y / block_m).astype(np.int64))
    u = np.unique(bid)
    f = np.random.default_rng(seed).integers(0, k, len(u))
    return f[np.searchsorted(u, bid)]


def plot_predictors(region):
    """FEATURES (+ wc_tree, tc2000, lossyear) of the 25 m cell at each plot centre of a region."""
    from pyproj import Transformer
    plots = pd.read_csv(STUDY / "field_plots.csv")[["plot", "region", "lon", "lat", "calibration", "agb_mgha"]]
    tl = pd.read_csv(PRED / region / "tiles.csv")
    p = plots[plots.region == region].copy().reset_index(drop=True)
    p["x"], p["y"] = Transformer.from_crs("EPSG:4326", tl.crs.iloc[0], always_xy=True).transform(p.lon.to_numpy(), p.lat.to_numpy())
    return pd.concat([p, features_at(region, p.x, p.y)], axis=1)


# ------------------------------------------------------------------ neighbourhood context
CTX = ["ndvi_m5", "ndvi_m11", "hv_m5", "hv_m11", "vh_m5", "vh_m11", "hv_sd3", "ndvi_sd3", "tpi11"]
FEATURES = PREDICTORS + CTX          # 22 predictors + context: means over 5 x 5 (125 m) and 11 x 11 (275 m) cells,
PAD, PX, TILE = 5, 25, 640           # local s.d. over 3 x 3 (75 m), topographic position (elevation minus its 275 m mean)


def predictor_vrt(region):
    import subprocess
    import sys as _sys
    vrt = PRED / region / "stack.vrt"
    files = sorted(str(p) for p in (PRED / region).glob("t_*.tif"))
    if not vrt.exists() or vrt.stat().st_mtime < max(Path(f).stat().st_mtime for f in files):
        subprocess.run([str(Path(_sys.executable).parent / "gdalbuildvrt"), "-q", str(vrt)] + files, check=True)
    return vrt


def _masked_mean(a, size):
    from scipy.ndimage import uniform_filter
    v = np.isfinite(a)
    num = uniform_filter(np.where(v, a, 0.0), size, mode="nearest")
    den = uniform_filter(v.astype(float), size, mode="nearest")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def tile_feature_arrays(src, x0, y0):
    """All bands of one 640 x 640 tile (upper-left x0, y0) read from the region's VRT with a 5-cell margin, plus the
    context features; returns {name: float32 array 640 x 640} with NaN for missing values."""
    from rasterio.windows import Window
    names = list(src.descriptions)
    c0 = int(round((x0 - src.transform.c) / PX)) - PAD
    r0 = int(round((src.transform.f - y0) / PX)) - PAD
    a = src.read(window=Window(c0, r0, TILE + 2 * PAD, TILE + 2 * PAD), boundless=True, fill_value=-32768).astype(np.float32)
    a[a == -32768] = np.nan
    g = {n: a[i] for i, n in enumerate(names)}
    ctx = {"ndvi_m5": _masked_mean(g["ndvi"], 5), "ndvi_m11": _masked_mean(g["ndvi"], 11), "hv_m5": _masked_mean(g["hv"], 5),
           "hv_m11": _masked_mean(g["hv"], 11), "vh_m5": _masked_mean(g["vh"], 5), "vh_m11": _masked_mean(g["vh"], 11)}
    for k, b in (("hv_sd3", "hv"), ("ndvi_sd3", "ndvi")):
        m, m2 = _masked_mean(g[b], 3), _masked_mean(g[b] ** 2, 3)
        ctx[k] = np.sqrt(np.clip(m2 - m ** 2, 0, None))
    ctx["tpi11"] = g["elev"] - _masked_mean(g["elev"], 11)
    s = slice(PAD, PAD + TILE)
    out = {n: g[n][s, s] for n in names}
    out.update({k: v[s, s].astype(np.float32) for k, v in ctx.items()})
    return out


def features_at(region, x, y):
    """FEATURES (+ wc_tree, tc2000, lossyear) for points (projected x, y) of a region; NaN where missing."""
    import rasterio
    tl = pd.read_csv(PRED / region / "tiles.csv")
    X0, Y0 = tl.x0.min(), tl.y0.max()
    x, y = np.asarray(x, float), np.asarray(y, float)
    col = np.floor((x - X0) / (PX * TILE)).astype(int)
    row = np.floor((Y0 - y) / (PX * TILE)).astype(int)
    cols = FEATURES + ["wc_tree", "tc2000", "lossyear"]
    out = np.full((len(x), len(cols)), np.nan, np.float32)
    groups = pd.Series(row * 10000 + col).groupby(row * 10000 + col).indices
    with rasterio.open(predictor_vrt(region)) as src:
        for t in tl.itertuples():
            idx = groups.get(t.row * 10000 + t.col)
            if idx is None:
                continue
            f = tile_feature_arrays(src, t.x0, t.y0)
            px = np.clip(((x[idx] - t.x0) // PX).astype(int), 0, TILE - 1)
            py = np.clip(((t.y0 - y[idx]) // PX).astype(int), 0, TILE - 1)
            out[idx] = np.stack([f[c][py, px] for c in cols]).T
    return pd.DataFrame(out, columns=cols)


# ------------------------------------------------------------------ GEDI L4B Version 2.1 (mission weeks 19-223, 2019-04-18 to 2023-03-16; doi 10.3334/ORNLDAAC/2299)
L4B_FILE = ROOT / "0_data/9_GEDI_L4B/GEDI_L4B_Gridded_Biomass_V2_1_2299/data/GEDI04_B_MW019MW223_02_002_02_R01000M_MU.tif"


def l4b_v21(x, y, crs):
    """AGBD (Mg/ha) of the 1 km L4B cell that contains each point (x, y in crs); NaN where L4B has no estimate."""
    import rasterio
    from pyproj import Transformer
    shape = np.shape(x)
    X, Y = Transformer.from_crs(crs, "EPSG:6933", always_xy=True).transform(np.asarray(x, float).ravel(), np.asarray(y, float).ravel())
    with rasterio.open(L4B_FILE) as s:
        c = np.floor((X - s.transform.c) / s.transform.a).astype(int)
        r = np.floor((Y - s.transform.f) / s.transform.e).astype(int)
        a = s.read(1, window=rasterio.windows.Window(c.min(), r.min(), c.max() - c.min() + 1, r.max() - r.min() + 1)).astype(float)
        nd = s.nodata
    if nd is not None:
        a[a == nd] = np.nan
    a[a < 0] = np.nan
    return a[r - r.min(), c - c.min()].reshape(shape)


def l4b_v21_grid(transform, width, height, crs):
    """L4B Version 2.1 at the cell centres of a grid."""
    xx, yy = np.meshgrid(transform.c + (np.arange(width) + 0.5) * transform.a, transform.f + (np.arange(height) + 0.5) * transform.e)
    return l4b_v21(xx, yy, crs)
