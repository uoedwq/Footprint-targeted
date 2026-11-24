"""PALSAR-2 predictors HH, HV, HV-HH: cells without 2021 data are filled from the 2022 mosaic (variant A, used in the article).

Rule of variant B (tested, not used):
  2021 yearly mosaic where it has data and, in the northeast, was acquired in April-October (the strips of 30 November 2021
  and later were acquired over frozen forest: HV about 4 dB below 2019, 2020, 2022 and 2023 at the same cells);
  all other cells: 2022 mosaic (complete; northeast 19 April - 26 May 2022, the season of most of the 2021 mosaic), then 2020.
A single year is used for the fill, not a mean of two, so the speckle and the local s.d. of HV are the same everywhere.
Every year is resampled bilinearly from its native grid (single images, no composite before the resampling).
Variant A (used) keeps the 2021 values of the frozen strips and fills the cells without data only.

  python palsar_fill.py test                    one filled, one frozen and one untouched tile: checks
  python palsar_fill.py run NE|SW B|A OUTDIR    writes the repaired tiles to OUTDIR/{region}; other tiles are symlinks
Output also: OUTDIR/{region}_repaired_tiles.csv
"""
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "in/data/predictors"
CACHE = ROOT / "out/study/fig_v4/cache"
PX, TILE = 25, 640
BANDS = ["hh", "hv", "hv_hh"]
FROZEN_FROM = "2021-11-01"                                 # northeast only
col = None


def init():
    global col
    ee.Initialize(project="YOUR-EE-PROJECT", opt_url="https://earthengine-highvolume.googleapis.com")
    col = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")


def year(y):
    return ee.Image(col.filterDate(f"{y}-01-01", f"{y + 1}-01-01").first())


def g3(y):
    """HH, HV, HV-HH (dB x 100) of one year on the output grid, exactly as predictors.py samples the 2021 mosaic (bilinear
    from the native grid; verified identical to the stored cells). Years are requested one by one and combined in numpy:
    unmask() of one year with another inside Earth Engine changed the sampling of the first."""
    dn = year(y).select(["HH", "HV"])
    g = dn.updateMask(dn.gt(0)).pow(2).log10().multiply(10).subtract(83)
    g = g.addBands(g.select("HV").subtract(g.select("HH")).rename("HV_HH")).multiply(100).resample("bilinear")
    return g.round().unmask(-32768, False).clamp(-32768, 32767).toInt16()


def day(y):
    """Acquisition date (days since 1970) of the year's mosaic, nearest cell; -9999 = no data."""
    im = year(y)
    return im.select("epoch").double().divide(86400000).updateMask(im.select("HV").gt(0)).unmask(-9999, False).toFloat()


def layers(region, variant, t):
    """(3, 640, 640) HH, HV, HV-HH and the year each cell comes from."""
    a = get(g3(2021), t)
    ok = (a != -32768).all(axis=0)
    if region == "NE" and variant == "B":
        d = get(day(2021), t)[0]
        ok &= (d >= pd.Timestamp("2021-04-01").value / 8.64e13) & (d < pd.Timestamp(FROZEN_FROM).value / 8.64e13)
    src = np.where(ok, 2021, 0).astype(np.int16)
    out = np.where(ok, a, -32768).astype(np.int16)
    for y in (2022, 2020):
        if (src == 0).any():
            b = get(g3(y), t)
            k = (src == 0) & (b != -32768).all(axis=0)
            out[:, k] = b[:, k]
            src[k] = y
    return out, src, a


def get(img, t):
    req = {"expression": img, "fileFormat": "GEO_TIFF",
           "grid": {"dimensions": {"width": TILE, "height": TILE},
                    "affineTransform": {"scaleX": PX, "shearX": 0, "translateX": t.x0, "shearY": 0, "scaleY": -PX, "translateY": t.y0}, "crsCode": t.crs}}
    msg = ""
    for k in range(6):
        try:
            with MemoryFile(ee.data.computePixels(req)) as mf, mf.open() as s:
                return s.read()
        except Exception as e:                             # noqa: BLE001
            msg = str(e)[:150]
            time.sleep(15 * (k + 1))
    raise RuntimeError(f"{t.file}: {msg}")


def affected(region, variant):
    """Tiles with a cell without 2021 data or (northeast, B) acquired in the frozen season (500 m survey, plus the gap list)."""
    z = np.load(CACHE / f"palsar_survey_{region}.npz")
    ep = z["ep2021"]
    bad = ~np.isfinite(ep)
    if region == "NE" and variant == "B":
        bad |= (ep >= pd.Timestamp(FROZEN_FROM).value / 8.64e13) | (ep < pd.Timestamp("2021-04-01").value / 8.64e13)
    tl = pd.read_csv(PRED / region / "tiles.csv")
    gaps = set(pd.read_csv(ROOT / "out/study/tiles_with_gaps.csv").query("aoi == @region").file)
    k = []
    for t in tl.itertuples():
        r0, c0 = int((float(z["y0"]) - t.y0) / 500), int((t.x0 - float(z["x0"])) / 500)
        k.append(bad[max(r0 - 1, 0):r0 + 33, max(c0 - 1, 0):c0 + 33].any() or t.file in gaps)
    return tl[np.array(k)]


def repair(region, variant, t, dst):
    new, src, _ = layers(region, variant, t)
    with rasterio.open(PRED / region / t.file) as s:
        a, prof, names = s.read(), s.profile, list(s.descriptions)
    old = a[names.index("hv")].copy()
    for j, b in enumerate(BANDS):
        a[names.index(b)] = new[j]
    tmp = dst.with_suffix(".part")
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(a)
        d.descriptions = tuple(names)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    tmp.rename(dst)
    k = src == 2021
    return dict(region=region, variant=variant, file=t.file, from_2021=k.mean(), from_2022=(src == 2022).mean(), from_2020=(src == 2020).mean(),
                missing_after=(src == 0).mean(), kept_2021_unchanged=(np.abs(new[1].astype(int) - old.astype(int))[k] <= 1).mean() if k.any() else np.nan,
                hv_mean_new=new[1][src > 0].mean() / 100, hv_mean_old=old[old != -32768].mean() / 100)


def run(region, variant, out):
    out = Path(out) / region
    out.mkdir(parents=True, exist_ok=True)
    tl = pd.read_csv(PRED / region / "tiles.csv")
    (out / "tiles.csv").write_text((PRED / region / "tiles.csv").read_text())
    aff = affected(region, variant)
    for f in tl.file:
        p = out / f
        if not p.exists() and not p.is_symlink():
            p.symlink_to(PRED / region / f)
    print(region, variant, "tiles", len(tl), "to repair", len(aff), flush=True)
    rows = []
    with cf.ThreadPoolExecutor(int(os.environ.get("THREADS", 8))) as ex:
        fut = [ex.submit(repair, region, variant, t, out / t.file) for t in aff.itertuples()]
        for k, f in enumerate(cf.as_completed(fut), 1):
            rows.append(f.result())
            if k % 25 == 0:
                print(region, variant, k, "of", len(fut), flush=True)
    d = pd.DataFrame(rows).sort_values("file")
    d.to_csv(out.parent / f"{region}_repaired_tiles.csv", index=False)
    print(d.describe().round(3).to_string())
    assert d.missing_after.max() == 0, "cells still missing"


def test():
    sd = lambda a: np.median(np.lib.stride_tricks.sliding_window_view(a, (3, 3)).std(axis=(2, 3)))
    for region, f, what in (("NE", "t_032_022.tif", "no data in 2021"), ("NE", "t_033_027.tif", "frozen strip (BH)"), ("NE", "t_015_024.tif", "untouched"),
                            ("SW", "t_031_029.tif", "no data in 2021"), ("SW", "t_012_012.tif", "untouched")):
        tl = pd.read_csv(PRED / region / "tiles.csv")
        t = next(tl[tl.file == f].itertuples())
        with rasterio.open(PRED / region / f) as s:
            old = s.read(list(s.descriptions).index("hv") + 1).astype(float) / 100
        new, src, _ = layers(region, "B", t)
        hv, k = new[1] / 100.0, src == 2021
        print(f"{region} {f} ({what}): cells from 2021/2022/2020 {k.mean():.2f}/{(src == 2022).mean():.2f}/{(src == 2020).mean():.2f}; stored HV mean {old.mean():.2f}, "
              f"sd(3x3) {sd(old):.2f} | new HV mean {hv.mean():.2f}, sd(3x3) {sd(hv):.2f}; max |new - stored| where 2021 kept: "
              f"{np.abs(hv - old)[k].max() if k.any() else float('nan'):.2f} dB", flush=True)


if __name__ == "__main__":
    init()
    if sys.argv[1] == "test":
        test()
    else:
        run(*sys.argv[2:5])
