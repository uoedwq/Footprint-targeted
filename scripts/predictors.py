"""2021 predictor stack at 25 m for NE, SW, NEtest, SWtest, downloaded from Earth Engine
(ee.data.computePixels on the high-volume endpoint; no Google Drive). One int16 GeoTIFF (deflate) per 640 x 640 px tile
(16 km) on a UTM grid aligned to 25 m; only tiles intersecting the AOI polygon. Resumable: existing tiles are skipped.

Bands (int16, nodata -32768):
  vv, vh, vh_vv          Sentinel-1 IW GRD, median of 2021 (dB x 100)
  b2 .. b12, ndvi, ndmi  Sentinel-2 L2A harmonized, Cloud Score+ cs_cdf >= 0.6, median (NE Jun-Sep 2021 leaf-on,
                         SW whole 2021); reflectance x 10000, indices x 10000
  hh, hv, hv_hh          ALOS-2 PALSAR-2 yearly mosaic 2021, gamma0 = 10 log10(DN^2) - 83 (dB x 100)
  elev, slope, north, east   Copernicus GLO-30 DEM (m; deg x 100; cos/sin aspect x 10000)
  wc_tree                ESA WorldCover 2021 tree-cover fraction of the 25 m cell (x 100)
  tc2000, lossyear       Hansen GFC v1.13 tree cover 2000 (%), loss year (1..25 = 2001..2025)
S1/S2 composites and the WorldCover tree mask are averaged over a 3 x 3 window of 10 m pixels (30 m) and sampled at
25 m; PALSAR-2 and DEM layers are resampled bilinearly. (A composite has a 1-degree default projection, so it is given
a 10 m projection first; reduceResolution exceeded the Earth Engine memory limit.)

  python predictors.py test NE        # one tile near the centre, timing and band summary
  python predictors.py run NE SW NEtest SWtest
Output: in/data/predictors/{aoi}/t_{row}_{col}.tif, in/data/predictors/{aoi}/tiles.csv
"""
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "in/data/predictors"
PX, TILE = 25, 640
CRS = {"NE": "EPSG:32652", "SW": "EPSG:32647", "NEtest": "EPSG:32651", "SWtest": "EPSG:32648"}
SHP = {"NE": "Studyarea_Northeast_500km_forGEE", "SW": "Studyarea_Southwest_500km_forGEE",
       "NEtest": "Studyarea_Northeast_100km_testNE_ForGEE", "SWtest": "Studyarea_Southwest_100km_testSW2_ForGEE"}
S2_SEASON = {"NE": ("2021-06-01", "2021-10-01"), "NEtest": ("2021-06-01", "2021-10-01"),
             "SW": ("2021-01-01", "2022-01-01"), "SWtest": ("2021-01-01", "2022-01-01")}
S2B = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BANDS = ["vv", "vh", "vh_vv"] + [b.lower() for b in S2B] + ["ndvi", "ndmi", "hh", "hv", "hv_hh", "elev", "slope", "north",
                                                            "east", "wc_tree", "tc2000", "lossyear"]
THREADS = int(os.environ.get("THREADS", 16))


def ee_init():
    import ee
    ee.Initialize(project="YOUR-EE-PROJECT", opt_url="https://earthengine-highvolume.googleapis.com")
    return ee


def tiles(aoi):
    poly = gpd.read_file(ROOT / f"0_data/3_vector/{SHP[aoi]}.shp").to_crs(CRS[aoi]).union_all()
    x0, y0, x1, y1 = poly.bounds
    x0, y1 = np.floor(x0 / PX) * PX, np.ceil(y1 / PX) * PX
    ncol, nrow = int(np.ceil((x1 - x0) / (PX * TILE))), int(np.ceil((y1 - y0) / (PX * TILE)))
    rows = []
    for r in range(nrow):
        for c in range(ncol):
            tx, ty = x0 + c * PX * TILE, y1 - r * PX * TILE
            if box(tx, ty - PX * TILE, tx + PX * TILE, ty).intersects(poly):
                rows.append(dict(row=r, col=c, x0=tx, y0=ty, file=f"t_{r:03d}_{c:03d}.tif"))
    return pd.DataFrame(rows), poly


def image(ee, aoi, x0, y0):
    crs = CRS[aoi]
    g = ee.Geometry.Rectangle([x0, y0 - PX * TILE, x0 + PX * TILE, y0], ee.Projection(crs), False)
    gb = g.buffer(500, 1)
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(gb).filterDate("2021-01-01", "2022-01-01")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]).median())
    s1 = s1.addBands(s1.select("VH").subtract(s1.select("VV")).rename("VH_VV")).multiply(100)
    s1 = s1.setDefaultProjection(crs, None, 10).focalMean(1, "square", "pixels")      # 30 m window mean
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(gb).filterDate(*S2_SEASON[aoi])
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90)).linkCollection(cs, ["cs_cdf"])
          .map(lambda im: im.updateMask(im.select("cs_cdf").gte(0.6))).select(S2B).median())
    s2 = s2.addBands(s2.normalizedDifference(["B8", "B4"]).multiply(10000).rename("NDVI")) \
           .addBands(s2.normalizedDifference(["B8", "B11"]).multiply(10000).rename("NDMI"))
    s2 = s2.setDefaultProjection(crs, None, 10).focalMean(1, "square", "pixels")
    pal = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH").filterDate("2021-01-01", "2022-01-01").first()
    dn = pal.select(["HH", "HV"])
    g0 = dn.updateMask(dn.gt(0)).pow(2).log10().multiply(10).subtract(83)
    g0 = g0.addBands(g0.select("HV").subtract(g0.select("HH")).rename("HV_HH")).multiply(100)
    dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(gb).select("DEM").mosaic()
           .setDefaultProjection(ee.Projection("EPSG:4326").atScale(30)))
    asp = ee.Terrain.aspect(dem).multiply(np.pi / 180)
    terr = ee.Image.cat(dem.rename("elev"), ee.Terrain.slope(dem).multiply(100).rename("slope"),
                        asp.cos().multiply(10000).rename("north"), asp.sin().multiply(10000).rename("east"))
    cont = ee.Image.cat(g0, terr).resample("bilinear")             # native ~25-30 m layers; composites aggregated above
    wcm = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    wc = wcm.eq(10).focalMean(1, "square", "pixels").multiply(100).rename("wc_tree")   # tree fraction, 30 m window
    han = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    han = ee.Image.cat(han.select("treecover2000"), han.select("lossyear").unmask(0))
    img = ee.Image.cat(s1, s2, cont, wc, han).rename(BANDS).round().unmask(-32768, False).clamp(-32768, 32767).toInt16()
    return img


def pixels(ee, aoi, x0, y0, n, img):
    """One computePixels request of n x n cells with upper-left corner (x0, y0); returns (array, profile)."""
    req = {"expression": img, "fileFormat": "GEO_TIFF",
           "grid": {"dimensions": {"width": n, "height": n},
                    "affineTransform": {"scaleX": PX, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -PX,
                                        "translateY": y0}, "crsCode": CRS[aoi]}}
    with MemoryFile(ee.data.computePixels(req)) as mf, mf.open() as src:
        return src.read(), src.profile


def fetch(ee, aoi, t, dest, retries=6):
    """Whole tile in one request; if Earth Engine runs out of memory (cloudy areas with many Sentinel-2 scenes), the
    tile is assembled from 2 x 2 and then 4 x 4 sub-requests."""
    img = image(ee, aoi, t.x0, t.y0)
    split, msg = 1, ""
    for k in range(retries):
        try:
            n = TILE // split
            a = np.empty((len(BANDS), TILE, TILE), np.int16)
            for i in range(split):
                for j in range(split):
                    a[:, i * n:(i + 1) * n, j * n:(j + 1) * n], prof = pixels(ee, aoi, t.x0 + j * n * PX, t.y0 - i * n * PX, n, img)
            prof = prof | dict(width=TILE, height=TILE, transform=from_origin(t.x0, t.y0, PX, PX), compress="deflate",
                               predictor=2, tiled=True, blockxsize=256, blockysize=256, nodata=-32768)
            tmp = dest.with_suffix(".part")
            with rasterio.open(tmp, "w", **prof) as dst:
                dst.write(a)
                dst.descriptions = tuple(BANDS)
            tmp.rename(dest)
            return True
        except Exception as e:                      # quota, timeout, transient server errors, memory limit
            msg = str(e)[:200]
            if "memory" in msg.lower() and split < 4:
                split *= 2
                continue
            time.sleep(min(300, 15 * 2 ** k))
    print(f"FAILED {aoi} {t.file}: {msg}", flush=True)
    return False


def run(aois):
    ee = ee_init()
    for aoi in aois:
        d = OUT / aoi
        d.mkdir(parents=True, exist_ok=True)
        tl, _ = tiles(aoi)
        tl.assign(crs=CRS[aoi]).to_csv(d / "tiles.csv", index=False)
        todo = [t for t in tl.itertuples() if not (d / t.file).exists()]
        print(f"{aoi}: {len(tl)} tiles, {len(todo)} to do", flush=True)
        t0, done, failed = time.time(), 0, 0
        with cf.ThreadPoolExecutor(THREADS) as ex:
            futs = [ex.submit(fetch, ee, aoi, t, d / t.file) for t in todo]
            for f in cf.as_completed(futs):
                ok = f.result()
                done += ok
                failed += not ok
                if (done + failed) % 50 == 0:
                    print(f"{aoi}: {done + failed}/{len(todo)} ({failed} failed), {(time.time() - t0) / 60:.0f} min", flush=True)
        print(f"{aoi}: finished {done}, failed {failed}, {(time.time() - t0) / 60:.0f} min", flush=True)


def test(aoi):
    ee = ee_init()
    tl, _ = tiles(aoi)
    t = tl.iloc[len(tl) // 2]
    if os.environ.get("TEST_XY"):                       # projected x,y inside the wanted tile
        x, y = map(float, os.environ["TEST_XY"].split(","))
        t = tl[(tl.x0 <= x) & (x < tl.x0 + PX * TILE) & (tl.y0 >= y) & (y > tl.y0 - PX * TILE)].iloc[0]
    dest = Path(os.environ.get("TEST_DIR", "/tmp")) / f"test_{aoi}.tif"
    t0 = time.time()
    ok = fetch(ee, aoi, t, dest, retries=4)
    print(f"{aoi}: {len(tl)} tiles; test tile {t.file} ok={ok} in {time.time() - t0:.0f} s, {dest.stat().st_size / 1e6:.1f} MB" if ok else "failed")
    if ok:
        with rasterio.open(dest) as src:
            a = src.read().astype(float)
            for i, b in enumerate(BANDS):
                v = a[i][a[i] != -32768]
                print(f"  {b:8s} valid {v.size / a[i].size:6.1%}  p5 {np.percentile(v, 5) if v.size else np.nan:9.1f}  "
                      f"p50 {np.median(v) if v.size else np.nan:9.1f}  p95 {np.percentile(v, 95) if v.size else np.nan:9.1f}")


if __name__ == "__main__":
    {"test": lambda a: test(a[0]), "run": run}[sys.argv[1]](sys.argv[2:])
