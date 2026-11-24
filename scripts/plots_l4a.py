"""What the field plots show about GEDI biomass, and how much of the RH -> AGB scatter comes from the plot-footprint offset.

Part 1 (plots): local power law (NE RH98, SW RH80; NLS, targeted shot) with leave-one-out
predictions, compared with L4A V2 and V3 on the same plots (bootstrap 95% intervals); leaf-off status of each targeted
shot (V3 L2A land_cover leaf_off_flag) and the LOO residuals by status. Also L4A V3 rescaled by the plots' ratio
sum(field)/sum(L4A) estimated without the predicted plot (one-parameter local calibration of the official product).
Part 2 (GEDI pairs): correlation of RH between two V3 footprints from different orbits as a function of their separation
(short-lag correlogram; Pearson, Spearman and a robust per-shot SD from the median pair difference), per region, for two filters: the conventional filter (quality_flag 1, degrade 0, sensitivity > 0.98,
night, power beams) and a relaxed one (quality_flag 1, degrade 0, sensitivity >= 0.95, all beams). Forest = Landsat tree
cover >= 30%, leaf-off shots excluded. Used to estimate how much the plots' V3 offsets (median 12.7 m NE, 23.4 m SW)
lower the plot-footprint correlation.
Output: out/study/plots_loo.csv, calibration_vs_l4a.csv, leafoff.csv, pair_correlation.csv,
        offset_attenuation.csv, filter_counts.csv, fig_pairs.png, fig_plots.png
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "out/study"
V3DIR = ROOT / "in/data/gedi_v3/merged"
METRIC = {"NE": "rh98", "SW": "rh80"}                       # RH metric per region
UTM = {"NE": "EPSG:32652", "SW": "EPSG:32647"}
POWER = ["BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]
BINS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50])
R_MAX = BINS[-1]
B = 5000
rng = np.random.default_rng(14)


def power(x, a, b):
    return a * np.power(x, b)


def fit_power(x, y):
    b0, la0 = np.polyfit(np.log(x), np.log(np.clip(y, 0.5, None)), 1)
    return curve_fit(power, x, y, p0=[np.exp(la0), b0], maxfev=20000)[0]


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = yhat - y
    return dict(rmse=np.sqrt((e ** 2).mean()), bias=e.mean(), r2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                r2_pearson=np.corrcoef(y, yhat)[0, 1] ** 2)


def with_ci(y, yhat):
    """Point metrics plus percentile bootstrap intervals (plots resampled with replacement)."""
    out = metrics(y, yhat)
    n = len(y)
    boot = pd.DataFrame([metrics(y[i], yhat[i]) for i in (rng.integers(0, n, n) for _ in range(B)) if np.ptp(y[i]) > 0])
    for k in ("rmse", "bias", "r2"):
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(boot[k], [2.5, 97.5])
    out.update(n=n, field_mean=y.mean(), rel_rmse=out["rmse"] / y.mean())
    return out


# ------------------------------------------------------------------ part 1: plots
plots = pd.read_csv(STUDY / "field_plots.csv")
v3 = pd.read_csv(STUDY / "plots_v3.csv", dtype={"v2_shot_number": str})
keep = ["plot", "v2_shot_number", "v3_plot_dist_m", "v3_delta_time", "v2_agbd_l4a", "v2_l4_quality", "v3_l4_agbd",
        "v3_l4_l4a_quality_flag_rel3", "v3_l4_predict_stratum"]
df = plots[["plot", "region", "calibration", "agb_mgha", "v1_rh98", "v1_rh80", "v1_sensitivity", "v1_solar_elevation",
            "v1_beam"]].merge(v3[keep], on="plot", how="left")
ids = [int(x) for x in df.v2_shot_number.dropna()]
lo = ds.dataset([str(f) for f in sorted(V3DIR.glob("L2A_[NS][EW]_*.parquet"))]).to_table(
    filter=pc.field("shot_number").isin(ids), columns=["shot_number", "leaf_off_flag", "phenology_phase", "landsat_treecover",
                                                       "pft_class"]).to_pandas()
lo["v2_shot_number"] = lo.shot_number.astype(str)
df = df.merge(lo.drop(columns="shot_number"), on="v2_shot_number", how="left")
df["date"] = (pd.Timestamp("2018-01-01") + pd.to_timedelta(df.v3_delta_time, unit="s")).dt.date

rows = []
for region in ["NE", "SW"]:
    m = METRIC[region]
    cal = df[(df.region == region) & df.calibration & df[f"v1_{m}"].gt(0)].copy()
    x, y = cal[f"v1_{m}"].to_numpy(float), cal.agb_mgha.to_numpy(float)
    cal["loo"] = [power(x[i], *fit_power(np.delete(x, i), np.delete(y, i))) for i in range(len(x))]
    df.loc[cal.index, "loo"] = cal.loo
    sets = {"all calibration plots": cal,
            "plots with an L4A V3 value": cal[cal.v3_l4_agbd.notna()],
            "plots passing the L4A V3 flag": cal[cal.v3_l4_l4a_quality_flag_rel3.eq(1)]}
    for name, s in sets.items():
        est = {"local power law (LOO)": s.loo}
        if name != "all calibration plots":
            est["L4A V3"] = s.v3_l4_agbd
            l4, yy = s.v3_l4_agbd.to_numpy(float), s.agb_mgha.to_numpy(float)
            est["L4A V3 x local ratio (LOO)"] = pd.Series([l4[i] * np.delete(yy, i).sum() / np.delete(l4, i).sum()
                                                           for i in range(len(s))], index=s.index)
            if name == "plots passing the L4A V3 flag":
                est["L4A V2"] = s.v2_agbd_l4a
        for e, v in est.items():
            ok = v.notna().to_numpy()
            rows.append(dict(region=region, plots=name, estimate=e) | with_ci(s.agb_mgha.to_numpy(float)[ok], v.to_numpy(float)[ok]))
cmp_ = pd.DataFrame(rows)
cmp_.to_csv(STUDY / "calibration_vs_l4a.csv", index=False)
df.to_csv(STUDY / "plots_loo.csv", index=False)
print(cmp_[["region", "plots", "estimate", "n", "field_mean", "bias", "rmse", "rel_rmse", "r2", "r2_pearson", "r2_lo", "r2_hi"]]
      .round(3).to_string(index=False), flush=True)

lf = []
for region in ["NE", "SW"]:
    c = df[(df.region == region) & df.loo.notna()].assign(logres=lambda d: np.log(d.agb_mgha / d.loo))
    for flag, g in c.groupby(c.leaf_off_flag.fillna(-1).astype(int)):
        other = c[c.leaf_off_flag.fillna(-1).astype(int) != flag]
        p = mannwhitneyu(g.logres, other.logres).pvalue if len(g) > 1 and len(other) > 1 else np.nan
        lf.append(dict(region=region, leaf_off_flag=flag, n=len(g), plots=" ".join(g["plot"]),
                       median_obs_over_pred=np.exp(g.logres.median()), p_vs_rest=p))
lf = pd.DataFrame(lf)
lf.to_csv(STUDY / "leafoff.csv", index=False)
print(lf.to_string(index=False), flush=True)


# ------------------------------------------------------------------ part 2: footprint pairs
COLS = ["lon", "lat", "beam", "granule", "delta_time", "quality_flag_rel2", "degrade_flag", "sensitivity", "solar_elevation",
        "landsat_treecover", "leaf_off_flag", "rh98", "rh80"]


def load(aoi):
    t = ds.dataset([str(f) for f in sorted(V3DIR.glob(f"L2A_{aoi}_*.parquet"))]).to_table(
        columns=COLS, filter=(pc.field("quality_flag_rel2") == 1) & (pc.field("degrade_flag") == 0)
        & (pc.field("sensitivity") >= 0.95) & (pc.field("landsat_treecover") >= 30) & (pc.field("leaf_off_flag") != 1))
    d = t.to_pandas()
    d["granule"] = pd.factorize(d.granule)[0].astype(np.int32)
    d["conventional"] = (d.sensitivity > 0.98) & (d.solar_elevation < 0) & d.beam.isin(POWER)
    x, y = Transformer.from_crs("EPSG:4326", UTM[aoi], always_xy=True).transform(d.lon.to_numpy(), d.lat.to_numpy())
    d["x"], d["y"] = x, y
    return d.drop(columns=["lon", "lat", "beam", "quality_flag_rel2", "degrade_flag", "solar_elevation", "leaf_off_flag"])


def pair_stats(d, region, filt):
    xy = d[["x", "y"]].to_numpy()
    pr = cKDTree(xy).query_pairs(R_MAX, output_type="ndarray")
    pr = pr[d.granule.to_numpy()[pr[:, 0]] != d.granule.to_numpy()[pr[:, 1]]]          # different orbits only
    dist = np.hypot(*(xy[pr[:, 0]] - xy[pr[:, 1]]).T)
    dt_yr = np.abs(d.delta_time.to_numpy()[pr[:, 0]] - d.delta_time.to_numpy()[pr[:, 1]]) / (365.25 * 86400)
    b = np.digitize(dist, BINS) - 1
    out = []
    for m in ("rh98", "rh80"):
        v = d[m].to_numpy(float)
        a1, a2 = v[pr[:, 0]], v[pr[:, 1]]
        for k in range(len(BINS) - 1):
            for span, sel in (("all", b == k), ("<=2 yr", (b == k) & (dt_yr <= 2))):
                u1, u2 = np.r_[a1[sel], a2[sel]], np.r_[a2[sel], a1[sel]]          # symmetric pairs
                dif = np.abs(a1[sel] - a2[sel])
                ok = sel.sum() > 30
                out.append(dict(region=region, filter=filt, metric=m, span=span, d_lo=BINS[k], d_hi=BINS[k + 1],
                                pairs=int(sel.sum()), r=np.corrcoef(u1, u2)[0, 1] if ok else np.nan,
                                rho=spearmanr(a1[sel], a2[sel]).statistic if ok else np.nan,
                                semivar=0.5 * np.mean(dif ** 2) if ok else np.nan, var=np.var(u1) if ok else np.nan,
                                sd_shot_robust=1.4826 * np.median(dif) / np.sqrt(2) if ok else np.nan,   # per-shot SD from MAD
                                share_diff_gt10m=np.mean(dif > 10) if ok else np.nan))
    return out


pairs, counts = [], []
for region in ["NE", "SW"]:
    d = load(region)
    for filt, sub in (("conventional", d[d.conventional]), ("relaxed", d)):
        counts.append(dict(region=region, filter=filt, forest_leafon_shots=len(sub)))
        pairs += pair_stats(sub.reset_index(drop=True), region, filt)
        print(region, filt, f"{len(sub):,} shots", flush=True)
    del d
pairs = pd.DataFrame(pairs)
pairs.to_csv(STUDY / "pair_correlation.csv", index=False)
pd.DataFrame(counts).to_csv(STUDY / "filter_counts.csv", index=False)
print(pairs[pairs.span.eq("all")].round(3).to_string(index=False), flush=True)

# attenuation at each calibration plot's V3 offset, relative to the 0-5 m bin
att = []
for region in ["NE", "SW"]:
    m = METRIC[region]
    c = df[(df.region == region) & df.loo.notna() & df.v3_plot_dist_m.notna()]
    for filt in ("conventional", "relaxed"):
        p = pairs[(pairs.region == region) & (pairs["filter"] == filt) & (pairs.metric == m) & (pairs.span == "all")]
        mid, r = ((p.d_lo + p.d_hi) / 2).to_numpy(), p.r.to_numpy()
        k = (np.interp(c.v3_plot_dist_m, mid, r) / r[0]) ** 2
        rho = p.rho.to_numpy()
        att.append(dict(region=region, filter=filt, metric=m, n_plots=len(c), median_offset_m=c.v3_plot_dist_m.median(),
                        r_0_5m=r[0], r_at_offsets_mean=np.interp(c.v3_plot_dist_m, mid, r).mean(), r2_factor_mean=k.mean(),
                        rho_0_5m=rho[0], rho2_factor_mean=((np.interp(c.v3_plot_dist_m, mid, rho) / rho[0]) ** 2).mean(),
                        plots_sd_metric=c[f"v1_{m}"].std(), shot_sd_0_5m=np.sqrt(p.semivar.iloc[0]),
                        shot_sd_robust_0_5m=p.sd_shot_robust.iloc[0]))
att = pd.DataFrame(att)
att.to_csv(STUDY / "offset_attenuation.csv", index=False)
print(att.round(3).to_string(index=False), flush=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for i, region in enumerate(["NE", "SW"]):
    m = METRIC[region]
    for filt, lw in (("relaxed", 1.8), ("conventional", 0.9)):
        p = pairs[(pairs.region == region) & (pairs["filter"] == filt) & (pairs.metric == m) & (pairs.span == "all")]
        mid = (p.d_lo + p.d_hi) / 2
        ax[i].plot(mid, p.r, "-o", lw=lw, ms=3 * lw, color="#0072B2", label=f"Pearson, {filt} filter")
        ax[i].plot(mid, p.rho, "--s", lw=lw, ms=3 * lw, color="#E69F00", label=f"Spearman, {filt} filter")
    off = df[(df.region == region) & df.loo.notna()].v3_plot_dist_m
    ax[i].axvspan(off.quantile(0.25), off.quantile(0.75), color="0.88", label="plot-footprint offsets (IQR)")
    ax[i].axvline(off.median(), color="0.4", lw=1)
    ax[i].set_title(f"{region}: {m.upper()} of two V3 footprints from different orbits", fontsize=10)
    ax[i].set_xlabel("distance between footprint centres (m)")
    ax[i].set_ylim(0, 0.7)
ax[0].set_ylabel("correlation")
ax[1].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(STUDY / "fig_pairs.png", dpi=200); plt.close(fig)

# field AGB vs local LOO prediction and L4A V3 at the same plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for i, region in enumerate(["NE", "SW"]):
    c = df[(df.region == region) & df.loo.notna()]
    ok = c.v3_l4_l4a_quality_flag_rel3.eq(1)
    ax[i].scatter(c.agb_mgha, c.loo, s=30, color="#0072B2", label="local power law (leave-one-out)", zorder=3)
    ax[i].scatter(c.agb_mgha[ok], c.v3_l4_agbd[ok], s=34, marker="^", color="#E69F00", label="L4A V3, passes flag", zorder=3)
    ax[i].scatter(c.agb_mgha[~ok], c.v3_l4_agbd[~ok], s=34, marker="^", facecolors="none", edgecolors="#E69F00",
                  label="L4A V3, fails flag (leaf-off)", zorder=3)
    hi = 1.08 * np.nanmax(np.r_[c.agb_mgha, c.loo, c.v3_l4_agbd])
    ax[i].plot([0, hi], [0, hi], color="0.5", lw=1)
    ax[i].set_xlim(0, hi); ax[i].set_ylim(0, hi)
    ax[i].set_title(f"{region}: {len(c)} calibration plots", fontsize=10)
    ax[i].set_xlabel("field AGB (Mg/ha)")
ax[0].set_ylabel("estimated AGB (Mg/ha)")
ax[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(STUDY / "fig_plots.png", dpi=200); plt.close(fig)
