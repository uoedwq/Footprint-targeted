"""RH -> AGB calibration from the footprint-targeted plots — model comparison, L4A-prior hybrid, and what each choice
does to the regional mean AGB of GEDI V3 forest footprints.

Calibration plots, RH metric per region (NE RH98, SW RH80; targeted V1 shot; the RH of the same footprints in V3 is compared in plot_checks.py):
  M1 power law fitted by NLS on the AGB scale 
  M2 log-log OLS with Snowdon ratio back-transformation
  M3 Gamma GLM with log link (constant coefficient of variation)
  M4 L4A V3 x local ratio (one parameter; plots whose L4A V3 value passes the L4A quality flag)
  H(n0) hybrid: log-log regression with a Zellner g-prior centred on the L4A V3 curve of the region, so the posterior is
        (n * local + n0 * L4A) / (n + n0); n0 = how many plots the L4A prior is worth. n0 = 0 is M2, n0 -> inf is the
        L4A curve. The L4A curve is a log-log fit of L4A V3 AGBD (quality flag 1) on the RH metric of the region over the region's
        forest footprints.
Leave-one-out (LOO) metrics with plot-bootstrap 95% intervals; nested LOO for choosing n0.
Regional application: V3 L2A forest footprints of NE/SW with the conventional filter (quality_flag 1, degrade 0,
sensitivity > 0.98, night, power beams), Landsat tree cover >= 30%, not leaf-off, and L4A V3 quality flag 1 (so every
model and the L4A product are averaged over the same footprints; unweighted footprint means, not map means). Regional mean footprint AGB per model;
sample-size uncertainty by plot bootstrap; allometry uncertainty by refitting with the five allometries.
Growth between GEDI shot and field date: RH change of footprint pairs <= 25 m apart from different orbits vs time apart.
Output: out/study/loo.csv, hybrid.csv, regional.csv, allometry.csv, growth.csv, fig_curves.png,
        fig_regional.png
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyproj import Transformer
from scipy.optimize import curve_fit, minimize
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "out/study"
V3DIR = ROOT / "in/data/gedi_v3/merged"
METRIC = {"NE": "rh98", "SW": "rh80"}
UTM = {"NE": "EPSG:32652", "SW": "EPSG:32647"}
POWER = ["BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]
FIELD_DATE = {"NE": pd.Timestamp("2021-10-31"), "SW": pd.Timestamp("2021-12-15")}   # NE Oct-Nov 2021, SW Dec 2021
ALLOM = ["agb_luo2018_mgha", "agb_ne_alt1_mgha", "agb_ne_alt2_mgha", "agb_sw_alt_mgha", "agb_chave14_wd055_mgha"]
N0 = [0, 3, 6, 12, 24, 48, np.inf]
B = 5000                                                                    # plot bootstraps (all intervals of this script)
rng = np.random.default_rng(15)


# ------------------------------------------------------------------ calibration models: fit(x, y) -> predict(x)
def m1_nls(x, y):
    b0, la0 = np.polyfit(np.log(x), np.log(np.clip(y, 0.5, None)), 1)
    a, b = curve_fit(lambda t, a, b: a * t ** b, x, y, p0=[np.exp(la0), b0], maxfev=20000)[0]
    return lambda t: a * t ** b


def loglog(x, y):
    """OLS of log y on log x -> (intercept, slope), Snowdon ratio correction."""
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    cf = y.mean() / np.exp(a + b * np.log(x)).mean()
    return np.array([a, b]), cf


def m2_loglog(x, y):
    (a, b), cf = loglog(x, y)
    return lambda t: cf * np.exp(a + b * np.log(t))


def m3_gamma(x, y):
    """Gamma GLM, log link: minimise the Gamma deviance (mean parameters do not depend on the shape)."""
    (a0, b0), _ = loglog(x, y)
    lx = np.log(x)
    f = lambda p: np.sum(y * np.exp(-(p[0] + p[1] * lx)) + p[0] + p[1] * lx)
    a, b = minimize(f, [a0, b0], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000)).x
    return lambda t: np.exp(a + b * np.log(t))


def hybrid(prior, n0):
    """g-prior posterior mean of the log-log parameters: weighted mean of local OLS and the L4A curve."""
    beta0, cf0 = prior

    def fit(x, y):
        beta, cf = loglog(x, y)
        n = len(x)
        if np.isinf(n0):
            beta, cf = beta0, cf0
        else:
            beta = (n * beta + n0 * beta0) / (n + n0)
            cf = y.mean() / np.exp(beta[0] + beta[1] * np.log(x)).mean()        # level re-anchored on the plots
        return lambda t: cf * np.exp(beta[0] + beta[1] * np.log(t))
    return fit


def metrics(y, p):
    e = p - y
    return dict(bias=e.mean(), rmse=np.sqrt((e ** 2).mean()), r2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def loo_pred(fit, x, y):
    return np.array([fit(np.delete(x, i), np.delete(y, i))(x[i:i + 1])[0] for i in range(len(x))])


def loo_with_ci(fit, x, y):
    p = loo_pred(fit, x, y)
    out = metrics(y, p)
    boot = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if np.ptp(y[i]) > 0:
            boot.append(metrics(y[i], p[i]))
    boot = pd.DataFrame(boot)
    for k in ("bias", "rmse", "r2"):
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(boot[k], [2.5, 97.5])
    out.update(n=len(y), field_mean=y.mean(), rel_rmse=out["rmse"] / y.mean())
    return out, p


# ------------------------------------------------------------------ regional footprints
COLS = ["shot_number", "lon", "lat", "beam", "granule", "delta_time", "quality_flag_rel2", "degrade_flag", "sensitivity",
        "solar_elevation", "landsat_treecover", "leaf_off_flag", "rh98", "rh80"]


def load_region(aoi):
    t = ds.dataset([str(f) for f in sorted(V3DIR.glob(f"L2A_{aoi}_*.parquet"))]).to_table(
        columns=COLS, filter=(pc.field("quality_flag_rel2") == 1) & (pc.field("degrade_flag") == 0)
        & (pc.field("sensitivity") >= 0.95) & (pc.field("landsat_treecover") >= 30) & (pc.field("leaf_off_flag") != 1))
    d = t.to_pandas()
    d["conventional"] = (d.sensitivity > 0.98) & (d.solar_elevation < 0) & d.beam.isin(POWER)
    d["granule"] = pd.factorize(d.granule)[0].astype(np.int32)
    ids = d.loc[d.conventional, "shot_number"].to_numpy()
    l4 = ds.dataset([str(f) for f in sorted(V3DIR.glob(f"L4A_{aoi}_*.parquet"))]).to_table(
        columns=["shot_number", "agbd", "l4a_quality_flag_rel3"], filter=pc.field("shot_number").isin(ids)).to_pandas()
    d = d.merge(l4, on="shot_number", how="left")
    x, y = Transformer.from_crs("EPSG:4326", UTM[aoi], always_xy=True).transform(d.lon.to_numpy(), d.lat.to_numpy())
    d["x"], d["y"] = x, y
    return d.drop(columns=["lon", "lat", "beam", "quality_flag_rel2", "degrade_flag", "solar_elevation", "leaf_off_flag"])


def growth(d, m):
    """Median RH change per year between footprints <= 25 m apart from different orbits (later minus earlier)."""
    xy = d[["x", "y"]].to_numpy()
    pr = cKDTree(xy).query_pairs(25, output_type="ndarray")
    g = d.granule.to_numpy()
    pr = pr[g[pr[:, 0]] != g[pr[:, 1]]]
    t, v = d.delta_time.to_numpy(), d[m].to_numpy(float)
    later = t[pr[:, 0]] > t[pr[:, 1]]
    i, j = np.where(later, pr[:, 0], pr[:, 1]), np.where(later, pr[:, 1], pr[:, 0])
    dt = (t[i] - t[j]) / (365.25 * 86400)
    dv = v[i] - v[j]
    ok = (dt >= 0.5) & (np.abs(dv) <= 10)                                  # drop near-simultaneous pairs and gross changes
    bins = np.arange(0.5, 7.5, 1.0)
    k = np.digitize(dt[ok], bins) - 1
    tab = pd.DataFrame(dict(dt=dt[ok], dv=dv[ok], k=k)).groupby("k").agg(dt=("dt", "median"), dv=("dv", "median"),
                                                                           pairs=("dv", "size")).reset_index(drop=True)
    tab = tab[tab.pairs >= 50]
    slope = np.polyfit(tab.dt, tab.dv, 1, w=np.sqrt(tab.pairs))[0]
    return slope, tab, int(ok.sum()), float(np.mean(np.abs(dv[dt >= 0.5]) > 10))


# ------------------------------------------------------------------ main
plots = pd.read_csv(STUDY / "plots_loo.csv").merge(pd.read_csv(STUDY / "field_plots.csv")[["plot"] + ALLOM], on="plot")
loo_rows, hyb_rows, reg_rows, allo_rows, gro_rows, curves = [], [], [], [], [], {}
for region in ["NE", "SW"]:
    m = METRIC[region]
    cal = plots[(plots.region == region) & plots.calibration & plots[f"v1_{m}"].gt(0)].reset_index(drop=True)
    x, y = cal[f"v1_{m}"].to_numpy(float), cal.agb_mgha.to_numpy(float)

    d = load_region(region)
    fp = d[d.conventional & d.l4a_quality_flag_rel3.eq(1) & d.agbd.notna()]       # same footprints for every model and L4A
    xr = np.clip(fp[m].to_numpy(float), 1.0, None)                         # RH <= 1 m: treat as 1 m (power laws need > 0)
    q = fp[fp.agbd.gt(0) & fp[m].gt(1)]
    prior = loglog(q[m].to_numpy(float), q.agbd.to_numpy(float))         # L4A V3 curve in the RH metric of the region
    print(f"{region}: {len(d):,} forest leaf-on shots, {d.conventional.sum():,} conventional filter, {len(fp):,} also L4A flag 1; "
          f"L4A curve a={prior[0][0]:.3f} b={prior[0][1]:.3f} cf={prior[1]:.3f}", flush=True)

    # growth between shot and field date
    slope, tab, npairs, share_big = growth(d, m)
    gap = (FIELD_DATE[region] - pd.to_datetime(cal.date)).dt.days.to_numpy() / 365.25
    b_loc = loglog(x, y)[0][1]
    gro_rows.append(dict(region=region, metric=m, rh_change_m_per_yr=slope, pairs=npairs, share_change_gt10m=share_big,
                         gap_yr_median=np.nanmedian(gap), plots_rh_median=np.median(x),
                         agb_change_pct=100 * b_loc * slope * np.nanmedian(gap) / np.median(x)))
    tab.assign(region=region).to_csv(STUDY / f"growth_bins_{region}.csv", index=False)

    models = {"M1 power law, NLS": m1_nls, "M2 log-log, Snowdon": m2_loglog, "M3 Gamma GLM": m3_gamma}
    for n0 in N0:
        models[f"H n0={n0}"] = hybrid(prior, n0)
    preds = {}
    for name, fit in models.items():
        res, p = loo_with_ci(fit, x, y)
        preds[name] = p
        f_all = fit(x, y)
        reg_mean = f_all(xr).mean()
        loo_rows.append(dict(region=region, model=name, **res, regional_mean=reg_mean))
        curves[(region, name)] = f_all
    # M4: L4A V3 x local ratio, on plots with an L4A value
    ok = cal.v3_l4_l4a_quality_flag_rel3.eq(1).to_numpy()                 # plots whose L4A value passes its flag
    l4p, yy = cal.v3_l4_agbd.to_numpy(float)[ok], y[ok]
    fit4 = lambda a, b: (lambda t: t * b.sum() / a.sum())
    p4 = np.array([fit4(np.delete(l4p, i), np.delete(yy, i))(l4p[i]) for i in range(len(yy))])
    res4 = metrics(yy, p4) | dict(n=len(yy), field_mean=yy.mean())
    ratio = yy.sum() / l4p.sum()
    l4_reg = fp.agbd
    loo_rows.append(dict(region=region, model="M4 L4A V3 x ratio", **res4, rel_rmse=res4["rmse"] / yy.mean(),
                         regional_mean=ratio * l4_reg.mean()))
    reg_rows.append(dict(region=region, model="extrapolation", regional_mean=np.nan, boot_lo=np.nan, boot_hi=np.nan, boot_n=0,
                         plots_max_metric=x.max(), share_beyond_plots=np.mean(xr > x.max()),
                         regional_metric_p50=np.median(xr), regional_metric_p90=np.percentile(xr, 90),
                         plots_metric_p50=np.median(x), n_footprints=len(xr)))
    loo_rows.append(dict(region=region, model="L4A V3 product", **(metrics(yy, l4p) | dict(n=len(yy), field_mean=yy.mean())),
                         rel_rmse=metrics(yy, l4p)["rmse"] / yy.mean(), regional_mean=l4_reg.mean()))

    # nested LOO: choose n0 on the training plots only, predict the held-out plot
    nest = []
    for i in range(len(x)):
        xt, yt = np.delete(x, i), np.delete(y, i)
        best = min(N0, key=lambda n0: np.mean((loo_pred(hybrid(prior, n0), xt, yt) - yt) ** 2))
        nest.append((hybrid(prior, best)(xt, yt)(x[i:i + 1])[0], best))
    pn = np.array([v for v, _ in nest])
    hyb_rows.append(dict(region=region, model="H nested n0", **metrics(y, pn), n=len(y),
                         n0_chosen=" ".join(str(b) for _, b in nest)))

    # sample-size (plot bootstrap) and allometry uncertainty of the regional mean
    ux, inv = np.unique(xr, return_inverse=True)
    wx = np.bincount(inv) / len(xr)                                          # mean over the footprints = weighted mean over the RH values
    for name in ["M1 power law, NLS", "M3 Gamma GLM", "H n0=12"]:
        means = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            if np.ptp(x[i]) > 0:
                try:
                    means.append(wx @ models[name](x[i], y[i])(ux))
                except RuntimeError:
                    pass
        reg_rows.append(dict(region=region, model=name, regional_mean=curves[(region, name)](xr).mean(),
                             boot_lo=np.percentile(means, 2.5), boot_hi=np.percentile(means, 97.5), boot_n=len(means)))
    reg_rows.append(dict(region=region, model="L4A V3 product", regional_mean=l4_reg.mean(), boot_lo=np.nan, boot_hi=np.nan,
                         boot_n=0))
    for al in ALLOM:
        ya = cal[al].to_numpy(float)
        allo_rows.append(dict(region=region, allometry=al, field_mean=ya.mean(),
                              regional_mean=m3_gamma(x, ya)(xr).mean(), l4a_ratio=ya[ok].sum() / l4p.sum()))
    curves[(region, "data")] = (x, y, fp[m].to_numpy(float), fp.agbd.to_numpy(float))
    del d, fp

loo = pd.DataFrame(loo_rows)
loo.to_csv(STUDY / "loo.csv", index=False)
pd.DataFrame(hyb_rows).to_csv(STUDY / "hybrid.csv", index=False)
reg = pd.DataFrame(reg_rows)
reg.to_csv(STUDY / "regional.csv", index=False)
pd.DataFrame(allo_rows).to_csv(STUDY / "allometry.csv", index=False)
gro = pd.DataFrame(gro_rows)
gro.to_csv(STUDY / "growth.csv", index=False)
pd.set_option("display.width", 220)
print(loo[["region", "model", "n", "bias", "rmse", "rel_rmse", "r2", "r2_lo", "r2_hi", "regional_mean"]].round(3).to_string(index=False))
print(pd.DataFrame(hyb_rows).round(3).to_string(index=False))
print(reg.round(2).to_string(index=False))
print(pd.DataFrame(allo_rows).round(3).to_string(index=False))
print(gro.round(3).to_string(index=False), flush=True)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for i, region in enumerate(["NE", "SW"]):
    x, y, xr, l4 = curves[(region, "data")]
    m = METRIC[region]
    hi = np.nanpercentile(xr, 99)
    bins = np.arange(1, hi + 1, 1.0)
    k = np.digitize(xr, bins)
    med = pd.Series(l4).groupby(k).median()
    q25, q75 = pd.Series(l4).groupby(k).quantile(0.25), pd.Series(l4).groupby(k).quantile(0.75)
    mid = med.index.to_numpy() + 0.5                                       # bin k covers [k, k+1) m
    ax[i].fill_between(mid, q25.to_numpy(), q75.to_numpy(), color="0.88", label="L4A V3 footprints (IQR)")
    ax[i].plot(mid, med.to_numpy(), color="0.45", lw=1.2, label="L4A V3 footprints (median)")
    t = np.linspace(1, hi, 200)
    for name, c, ls in (("M1 power law, NLS", "#56B4E9", "-"), ("M3 Gamma GLM", "#0072B2", "-"), ("H n0=12", "#E69F00", "--"),
                        ("H n0=inf", "#000000", ":")):
        ax[i].plot(t, curves[(region, name)](t), color=c, ls=ls, label=name.replace("H n0=inf", "L4A curve (n0=inf)"))
    ax[i].scatter(x, y, s=22, color="k", zorder=4, label="calibration plots")
    ax[i].axvline(x.max(), color="k", lw=0.6, ls=":")
    ax[i].set_xlabel(f"GEDI {m.upper()} (m)")
    ax[i].set_title(f"{region}: RH to AGB, plots end at {m.upper()} = {x.max():.0f} m", fontsize=10)
    ax[i].set_ylim(0, None)
ax[0].set_ylabel("AGB (Mg/ha)")
ax[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(STUDY / "fig_curves.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for i, region in enumerate(["NE", "SW"]):
    r = loo[(loo.region == region) & ~loo.model.str.startswith("H n0=3") & ~loo.model.str.startswith("H n0=6")
            & ~loo.model.str.startswith("H n0=48")]
    labels = r.model.replace({"H n0=inf": "L4A curve (H n0=inf)"})
    ax[i].barh(labels, r.regional_mean, color=["#E69F00" if "L4A" in s or "inf" in s else "#0072B2" for s in r.model])
    b = reg[(reg.region == region) & reg.boot_lo.notna()]
    for _, rr in b.iterrows():
        yy = list(r.model).index(rr.model)
        ax[i].plot([rr.boot_lo, rr.boot_hi], [yy, yy], color="k", lw=1.5)
    ax[i].set_title(f"{region}: mean AGB of forest footprints", fontsize=10)
    ax[i].set_xlabel("Mg/ha")
    ax[i].invert_yaxis()
fig.tight_layout(); fig.savefig(STUDY / "fig_regional.png", dpi=200); plt.close(fig)
