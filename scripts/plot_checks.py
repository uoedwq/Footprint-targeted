"""Checks at the calibration plots that are quoted in the text (run after the maps and the leave-one-plot-out validation):
  a  correlation of field AGB with every RH metric, and the metric chosen again in every leave-one-out fold (Gamma GLM);
  b  calibration with the plots within a given distance of their Version 3 footprint;
  c  calibration with the Version 3 RH of the same footprints instead of the Version 1 RH;
  d  coverage of the 90% prediction intervals of the maps at the plots (leave-one-plot-out);
  e  one rescaling factor for L4A Version 3;
  f  regional mean AGB under the calibrations of b (<= 25 m: plot and footprint overlap) and c (exact weights of map_agb.py, same trees).
Output: out/study/metrics.csv, nested_metric.csv, distance.csv, rh_versions.csv, intervals.csv, near_totals.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import METRIC, STUDY, agb, gamma_fit

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
RH = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98]
p = pd.read_csv(STUDY / "field_plots.csv")[["plot", "region", "calibration", "agb_mgha"] + [f"v1_rh{k}" for k in RH]]
v = pd.read_csv(STUDY / "plots_v3.csv")[["plot", "v3_rh98", "v3_rh80", "v3_plot_dist_m"]]
h = p[p.calibration].merge(v, on="plot")
assert len(h) == 36 and h.notna().all().all()


def stat(y, e):
    x = np.asarray(e) - np.asarray(y)
    return dict(n=len(x), bias=x.mean(), rmse=np.sqrt((x ** 2).mean()), r2=1 - (x ** 2).sum() / ((y - y.mean()) ** 2).sum())


def loo(x, y):
    x = np.clip(x, 1, None)
    return np.array([agb(gamma_fit(np.delete(x, i), np.delete(y, i)), x[i:i + 1])[0] for i in range(len(x))])


# ------------------------------------------------------------------ a  metrics
per, nest = [], []
for r in ("NE", "SW"):
    g = h[h.region == r]
    y = g.agb_mgha.to_numpy(float)
    X = {k: g[f"v1_rh{k}"].to_numpy(float) for k in RH}
    for k in RH:
        per.append(dict(region=r, metric=f"rh{k}", pearson=np.corrcoef(X[k], y)[0, 1], spearman=pd.Series(X[k]).corr(pd.Series(y), method="spearman"), clipped=int((X[k] < 1).sum()))
                   | {f"loo_{a}": b for a, b in stat(y, loo(X[k], y)).items()})
    pred, chosen = [], []
    for i in range(len(y)):
        yt = np.delete(y, i)
        best = max(RH, key=lambda k: np.corrcoef(np.delete(X[k], i), yt)[0, 1])
        x = np.clip(X[best], 1, None)
        pred.append(agb(gamma_fit(np.delete(x, i), yt), x[i:i + 1])[0]); chosen.append(best)
    nest.append(dict(region=r) | stat(y, np.array(pred)) | dict(chosen=" ".join(f"RH{k}x{chosen.count(k)}" for k in sorted(set(chosen)))))
per, nest = pd.DataFrame(per), pd.DataFrame(nest)
per.to_csv(STUDY / "metrics.csv", index=False); nest.to_csv(STUDY / "nested_metric.csv", index=False)
print(per.round(3).to_string(index=False)); print(nest.round(3).to_string(index=False))

# ------------------------------------------------------------------ b, c  distance to the Version 3 footprint; Version 3 RH
rows, ver, PAR = [], [], {}
for r in ("NE", "SW"):
    m = METRIC[r]
    for lab, g, c in [(f"<= {q} m", h[(h.region == r) & (h.v3_plot_dist_m <= q)], f"v1_{m}") for q in (15, 20, 25, 30, 40)] + [
            ("all calibration plots", h[h.region == r], f"v1_{m}"), ("Version 3 RH", h[h.region == r], f"v3_{m}")]:
        x, y = np.clip(g[c].to_numpy(float), 1, None), g.agb_mgha.to_numpy(float)
        PAR[(r, lab)] = (gamma_fit(x, y), x, y)
        a, b = PAR[(r, lab)][0]
        rows.append(dict(region=r, plots=lab, a=np.exp(a), b=b) | stat(y, loo(x, y)))
    g = h[h.region == r]
    e = (g[f"v3_{m}"] - g[f"v1_{m}"]).abs()
    ver.append(dict(region=r, n=len(g), same_within_5cm=int((e <= 0.05).sum()), within_06m=int((e <= 0.61).sum()), max_diff=e.max(), plot_of_max=g.loc[e.idxmax(), "plot"]))
dist, ver = pd.DataFrame(rows), pd.DataFrame(ver)
dist.to_csv(STUDY / "distance.csv", index=False); ver.to_csv(STUDY / "rh_versions.csv", index=False)
print(dist.round(3).to_string(index=False)); print(ver.round(2).to_string(index=False))

# ------------------------------------------------------------------ d  90% prediction intervals at the plots
m = pd.concat([pd.read_csv(STUDY / f"{r}_lopo_mapcfg.csv") for r in ("NE", "SW")])
m = m[(m.scheme == "v3_strict") & m.calibration]
rows = []
for r in ("NE", "SW"):
    c = pd.read_csv(STUDY / f"{r}_conformal.csv")
    edges = np.r_[c.lo_edge, c.hi_edge.iloc[-1]]
    g = m[m.region == r]
    k = np.clip(np.searchsorted(edges, g.lopo, side="right") - 1, 0, len(edges) - 2)
    lo, hi = np.clip(g.lopo.to_numpy() + c.q05.to_numpy()[k], 0, None), g.lopo.to_numpy() + c.q95.to_numpy()[k]
    rows.append(pd.DataFrame(dict(plot=g["plot"], region=r, calibration=g.calibration, field=g.agb_field, map=g.lopo, lo=lo, hi=hi, inside=(g.agb_field >= lo) & (g.agb_field <= hi))))
e = pd.concat(rows)
e.to_csv(STUDY / "intervals.csv", index=False)
for lab, g in (("all", e), ("NE", e[e.region == "NE"]), ("SW", e[e.region == "SW"])):
    k, n = int(g.inside.sum()), len(g)
    print(f"{lab}: {k}/{n} inside ({k / n:.0%}, 95% {beta.ppf(.025, k, n - k + 1):.2f}-{beta.ppf(.975, k + 1, n - k) if k < n else 1:.2f}); outside: {g[~g.inside]['plot'].tolist()}")

# ------------------------------------------------------------------ e  one factor
t = pd.read_csv(STUDY / "loo.csv")
print(t[t.model.isin(["M3 Gamma GLM", "M4 L4A V3 x ratio", "L4A V3 product"])][["region", "model", "n", "bias", "rmse", "r2", "regional_mean"]].round(2).to_string(index=False))

# ------------------------------------------------------------------ f  regional mean under the calibrations of b and c
rows = []
for r in ("NE", "SW"):
    z = np.load(STUDY / f"{r}_weights.npz")
    tot = pd.read_csv(STUDY / f"{r}_totals.csv").set_index("set")
    l4, main = tot.loc["L4A V3 curve", "mean_agb"], tot.loc["main (local Gamma GLM)", "mean_agb"]
    for lab in ("<= 25 m", "Version 3 RH"):
        par, x, y = PAR[(r, lab)]
        mean = float(z["w"] @ agb(par, z["rh"]))
        rows.append(dict(region=r, plots=lab, n=len(x), mean_agb=mean, rel_main=mean / main - 1, l4a=l4, l4a_rel=l4 / mean - 1, l4a_rel_main=l4 / main - 1))
near = pd.DataFrame(rows)
near.to_csv(STUDY / "near_totals.csv", index=False)
print(near.round(3).to_string(index=False))
