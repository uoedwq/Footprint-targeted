"""Compare our 2021 maps with GEDI L4B V2.1 (1 km) and ESA CCI Biomass V6 2021 (100 m), at 1 km and at the plots.

1 km: L4B Version 2.1 (file of the ORNL DAAC, common.l4b_v21) and CCI Biomass (Earth Engine, averaged from 100 m) on the exact 1 km grid of our map
(out/maps/{aoi}/{aoi}_agb_1km.tif). Comparison in cells that lie inside the AOI and are >= 80% forest in our mask, where the
products' all-land means approximate a forest mean.
Plots: L4B 1 km cell and CCI 100 m pixel at each plot centre, against field AGB.
Output: out/study/products_1km.csv, products_plots.csv, fig_products.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MAPS, STUDY, l4b_v21, l4b_v21_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ee
ee.Initialize(project="YOUR-EE-PROJECT", opt_url="https://earthengine-highvolume.googleapis.com")
CCI = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB") \
    .filter(ee.Filter.eq("system:index", "CCI_BIOMASS_100m_AGB_2021_V6")).first().select("AGB")


def fetch(img, crs, transform, w, h, chunk=300):
    out = np.full((h, w), np.nan, np.float32)
    for r0 in range(0, h, chunk):
        for c0 in range(0, w, chunk):
            hh, ww = min(chunk, h - r0), min(chunk, w - c0)
            req = {"expression": img.unmask(-9999).toFloat(), "fileFormat": "GEO_TIFF",
                   "grid": {"dimensions": {"width": ww, "height": hh},
                            "affineTransform": {"scaleX": transform.a, "shearX": 0, "translateX": transform.c + c0 * transform.a,
                                                "shearY": 0, "scaleY": transform.e, "translateY": transform.f + r0 * transform.e},
                            "crsCode": crs}}
            with MemoryFile(ee.data.computePixels(req)) as mf, mf.open() as src:
                a = src.read(1)
            out[r0:r0 + hh, c0:c0 + ww] = np.where(a == -9999, np.nan, a)
    return out


rows, pix = [], {}
for aoi in ["NE", "SW"]:
    with rasterio.open(MAPS / aoi / f"{aoi}_agb_1km.tif") as src:
        ours, ffrac, afrac = src.read()[:3].astype(float)
        crs, tr, w, h = src.crs.to_string(), src.transform, src.width, src.height
    crs_ee = crs if crs.startswith("EPSG") else "EPSG:" + str(src.crs.to_epsg())
    l4b = l4b_v21_grid(tr, w, h, crs_ee)
    cci = fetch(CCI.reduceResolution(ee.Reducer.mean(), maxPixels=512).reproject(crs=crs_ee, scale=1000), crs_ee, tr, w, h, 150)
    sel = (afrac >= 0.999) & (ffrac >= 0.8) & np.isfinite(ours)
    pix[aoi] = (ours[sel], l4b[sel], cci[sel])
    for name, v in (("GEDI L4B V2.1", l4b), ("ESA CCI V6 2021", cci)):
        ok = sel & np.isfinite(v)
        e = v[ok] - ours[ok]
        q = lambda z: np.percentile(z, [5, 50, 95])
        rows.append(dict(region=aoi, product=name, cells=ok.sum(), ours_mean=ours[ok].mean(), product_mean=v[ok].mean(),
                         ours_p05=q(ours[ok])[0], ours_p50=q(ours[ok])[1], ours_p95=q(ours[ok])[2], ours_sd=ours[ok].std(),
                         prod_p05=q(v[ok])[0], prod_p50=q(v[ok])[1], prod_p95=q(v[ok])[2], prod_sd=v[ok].std(),
                         diff_mean=e.mean(), rel_diff=e.mean() / ours[ok].mean(), r=np.corrcoef(ours[ok], v[ok])[0, 1],
                         rmsd=np.sqrt((e ** 2).mean())))
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(STUDY / "products_1km.csv", index=False)

# plots
pl = pd.read_csv(STUDY / "field_plots.csv")[["plot", "region", "lon", "lat", "calibration", "agb_mgha"]]
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"plot": r.plot}) for r in pl.itertuples()])
v2 = CCI.rename("cci").reduceRegions(fc, ee.Reducer.first(), 100).getInfo()["features"]
pv = pd.DataFrame([dict(plot=f["properties"]["plot"], cci=f["properties"].get("first")) for f in v2])
pl["l4b"] = np.nan
for region in ("NE", "SW"):
    s = pl.region == region
    pl.loc[s, "l4b"] = l4b_v21(pl.lon[s], pl.lat[s], "EPSG:4326")
pl = pl.merge(pv, on="plot")
pl.to_csv(STUDY / "products_plots.csv", index=False)
summ = []
for region, g in pl.groupby("region"):
    for k in ("l4b", "cci"):
        ok = g[k].notna()
        e = g[k][ok] - g.agb_mgha[ok]
        summ.append(dict(region=region, product=k, n=ok.sum(), bias=e.mean(), rmse=np.sqrt((e ** 2).mean()),
                         r2=1 - (e ** 2).sum() / ((g.agb_mgha[ok] - g.agb_mgha[ok].mean()) ** 2).sum()))
pd.DataFrame(summ).to_csv(STUDY / "products_plots_summary.csv", index=False)
print(pd.DataFrame(summ).round(2).to_string(index=False), flush=True)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
for i, aoi in enumerate(["NE", "SW"]):
    o, l, c = pix[aoi]
    for j, (v, name) in enumerate(((l, "GEDI L4B V2.1"), (c, "ESA CCI 2021"))):
        a = ax[2 * i + j]
        ok = np.isfinite(v)
        a.hexbin(o[ok], v[ok], gridsize=50, mincnt=1, bins="log", cmap="viridis")
        hi = np.nanpercentile(np.r_[o, v[ok]], 99.5)
        a.plot([0, hi], [0, hi], color="k", lw=1)
        a.set_xlim(0, hi); a.set_ylim(0, hi)
        a.set_xlabel("this study, 1 km forest mean (Mg/ha)"); a.set_ylabel(f"{name} (Mg/ha)")
        a.set_title(f"{aoi}: {ok.sum():,} cells >= 80% forest", fontsize=9)
fig.tight_layout(); fig.savefig(STUDY / "fig_products.png", dpi=200); plt.close(fig)
