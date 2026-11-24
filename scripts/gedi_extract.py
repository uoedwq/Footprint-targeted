"""Extract every GEDI V3 shot inside the study areas from NASA (L2A for RH metrics, L4A V3 for AGBD).

Areas: the two 500 km regions and the two 100 km test areas (0_data/3_vector/*_forGEE.shp), 2019-04 .. present.
Kept shots: L2A l2_algrunflag == 1 or l2a_quality_flag_rel2 == 1; L4A l2_algrunflag == 1 (all quality flags kept
so the QC experiments can compare V2-style, V3 and custom filters).

Modes per granule:
  stream    read only the needed datasets over HTTPS (large block cache)
  download  copy the whole granule to node-local scratch, read, delete
Usage:
  python gedi_extract.py list                         # search CMR once, write granule lists
  python gedi_extract.py run L2A --shard 0/8 --workers 6 --mode download
  python gedi_extract.py bench L2A                    # time both modes on one granule
Output: in/data/gedi_v3/{L2A,L4A}/parts/<granule>.parquet ; done-lists for resuming.
"""
import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import earthaccess
import h5py
import numpy as np
import pandas as pd
import pyogrio
import shapely

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "in/data/gedi_v3"
AOIS = {"NE": "0_data/3_vector/Studyarea_Northeast_500km_forGEE.shp",
        "SW": "0_data/3_vector/Studyarea_Southwest_500km_forGEE.shp",
        "NEtest": "0_data/3_vector/Studyarea_Northeast_100km_testNE_ForGEE.shp",
        "SWtest": "0_data/3_vector/Studyarea_Southwest_100km_testSW2_ForGEE.shp"}
POLY = {k: pyogrio.read_dataframe(ROOT / v).geometry.iloc[0] for k, v in AOIS.items()}
for p in POLY.values():
    shapely.prepare(p)
RH_KEEP = list(range(0, 101, 5)) + [98, 99]
PRODUCTS = {
    "L2A": dict(short_name="GEDI02_A", version="003", fields={
        "shot_number": "shot_number", "delta_time": "delta_time", "degrade_flag": "degrade_flag",
        "quality_flag_rel2": "l2a_quality_flag_rel2", "quality_flag_rel3": "l2a_quality_flag_rel3",
        "l2_algrunflag": "l2_algrunflag", "sensitivity": "sensitivity", "sensitivity_a2": "geolocation/sensitivity_a2",
        "solar_elevation": "solar_elevation", "selected_algorithm": "selected_algorithm",
        "num_detectedmodes": "num_detectedmodes", "elev_lowestmode": "elev_lowestmode",
        "elev_highestreturn": "elev_highestreturn", "digital_elevation_model": "digital_elevation_model",
        "surface_flag": "surface_flag", "elevation_bias_flag": "elevation_bias_flag",
        "stale_return_flag": "geolocation/stale_return_flag", "rx_maxamp": "rx_assess/rx_maxamp",
        "sd_corrected": "rx_assess/sd_corrected", "local_beam_elevation": "geolocation/local_beam_elevation",
        "local_beam_azimuth": "geolocation/local_beam_azimuth", "pft_class": "land_cover_data/pft_class",
        "region_class": "land_cover_data/region_class", "worldcover_class": "land_cover_data/worldcover_class",
        "landsat_treecover": "land_cover_data/landsat_treecover", "leaf_off_flag": "land_cover_data/leaf_off_flag",
        "urban_proportion": "land_cover_data/urban_proportion", "phenology_phase": "land_cover_data/phenology_phase"},
        keep=lambda d: (d["l2_algrunflag"] == 1) | (d["quality_flag_rel2"] == 1)),
    "L4A": dict(short_name="GEDI_L4A_AGB_Density_V3_2508", version=None, fields={
        "shot_number": "shot_number", "delta_time": "delta_time", "agbd": "agbd", "agbd_se": "agbd_se",
        "agbd_pi_lower": "agbd_pi_lower", "agbd_pi_upper": "agbd_pi_upper", "agbd_t": "agbd_t", "agbd_t_se": "agbd_t_se",
        "l2a_quality_flag_rel3": "l2a_quality_flag_rel3", "l4a_quality_flag_rel3": "l4a_quality_flag_rel3",
        "l2_algrunflag": "l2_algrunflag", "predict_stratum": "predict_stratum", "predictor_limit_flag": "predictor_limit_flag",
        "response_limit_flag": "response_limit_flag", "selected_algorithm": "selected_algorithm",
        "degrade_flag": "degrade_flag", "sensitivity": "sensitivity", "pft_class": "land_cover_data/pft_class",
        "region_class": "land_cover_data/region_class"},
        keep=lambda d: d["l2_algrunflag"] == 1),
}
lock = threading.Lock()


def granule_list(prod):
    f = BASE / "meta" / f"{prod}_granules.json"
    if f.exists():
        return json.loads(f.read_text())
    spec = PRODUCTS[prod]
    seen = {}
    for aoi, poly in POLY.items():
        kw = dict(short_name=spec["short_name"], bounding_box=tuple(poly.bounds), temporal=("2019-04-01", "2026-12-31"))
        if spec["version"]:
            kw["version"] = spec["version"]
        for g in earthaccess.search_data(**kw):
            ur = g["umm"]["GranuleUR"]
            links = [u for u in g.data_links() if u.endswith(".h5")]
            seen.setdefault(ur, dict(url=links[0], mb=g.size, aois=[]))["aois"].append(aoi)
        print(f"{prod} {aoi}: cumulative {len(seen)} granules", flush=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(seen))
    return seen


def extract(h, prod, aois, name):
    spec, frames = PRODUCTS[prod], []
    for beam in sorted(k for k in h.keys() if k.startswith("BEAM")):
        b = h[beam]
        lat, lon = b["lat_lowestmode"][:], b["lon_lowestmode"][:]
        inside = np.zeros(lat.size, bool)
        which = np.full(lat.size, "", object)
        for aoi in aois:
            x0, y0, x1, y1 = POLY[aoi].bounds
            cand = np.flatnonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
            if cand.size:
                ok = shapely.contains_xy(POLY[aoi], lon[cand], lat[cand])
                inside[cand[ok]] = True
                which[cand[ok]] = aoi
        idx = np.flatnonzero(inside)
        if not idx.size:
            continue
        i0, i1 = idx.min(), idx.max() + 1
        rec = {"granule": name, "beam": beam, "aoi": which[idx], "lon": lon[idx], "lat": lat[idx]}
        for out, path in spec["fields"].items():
            if path in b:
                rec[out] = b[path][i0:i1][idx - i0]
        if prod == "L2A":
            rh = b["rh"][i0:i1][idx - i0]
            for k in RH_KEEP:
                rec[f"rh{k}"] = rh[:, k].astype("float32")
        df = pd.DataFrame(rec)
        frames.append(df[spec["keep"](df)])
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df["shot_number"] = df["shot_number"].astype("uint64")
        for c in df.columns:
            if df[c].dtype == "float64" and c not in ("lon", "lat", "delta_time"):
                df[c] = df[c].astype("float32")
    return df


def process(prod, name, info, mode, scratch, donef):
    out = BASE / prod / "parts" / f"{name}.parquet"
    for attempt in range(5):
        try:
            if mode == "download":
                d = Path(tempfile.mkdtemp(dir=scratch))
                try:
                    path = earthaccess.download([info["url"]], local_path=str(d), show_progress=False)[0]
                    with h5py.File(path, "r") as h:
                        df = extract(h, prod, info["aois"], name)
                finally:
                    shutil.rmtree(d, ignore_errors=True)
            else:
                fs = earthaccess.get_fsspec_https_session()
                with fs.open(info["url"], cache_type="blockcache", block_size=32 * 2 ** 20) as fh, h5py.File(fh, "r") as h:
                    df = extract(h, prod, info["aois"], name)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            with lock, open(donef, "a") as fh:            # one done-file per shard: no cross-node appends
                fh.write(name + "\n")
            return len(df)
        except Exception as e:
            print(f"  {name}: attempt {attempt + 1}: {type(e).__name__}: {str(e)[:150]}", flush=True)
            time.sleep(30 * (attempt + 1))
    return -1


def run(prod, shard, workers, mode):
    grans = granule_list(prod)
    k, n = map(int, shard.split("/"))
    names = sorted(grans)[k::n]
    donef = BASE / prod / f"done_{k}of{n}.txt"
    donef.parent.mkdir(parents=True, exist_ok=True)
    done = set().union(*[set(f.read_text().split()) for f in (BASE / prod).glob("done*.txt")])
    todo = [g for g in names if g not in done]
    scratch = os.environ.get("SNIC_TMP") or os.environ.get("TMPDIR") or "/tmp"
    print(f"{prod} shard {shard}: {len(todo)} of {len(names)} granules to do, mode={mode}, scratch={scratch}", flush=True)
    t0, nshots, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, prod, g, grans[g], mode, scratch, donef): g for g in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            (failed.append(futs[f]) if r < 0 else None)
            nshots += max(r, 0)
            if i % 10 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"[{i}/{len(futs)}] shots {nshots:,}, failed {len(failed)}, {el / 60:.1f} min, {el / i:.1f} s/granule", flush=True)
    print("failed:", failed, flush=True)


def bench(prod):
    grans = granule_list(prod)
    name = sorted(grans)[len(grans) // 2]
    scratch = os.environ.get("SNIC_TMP") or os.environ.get("TMPDIR") or "/tmp"
    for mode in ["stream", "download"]:
        t0 = time.time()
        d = Path(tempfile.mkdtemp(dir=scratch))
        try:
            if mode == "download":
                p = earthaccess.download([grans[name]["url"]], local_path=str(d), show_progress=False)[0]
                with h5py.File(p, "r") as h:
                    df = extract(h, prod, grans[name]["aois"], name)
            else:
                fs = earthaccess.get_fsspec_https_session()
                with fs.open(grans[name]["url"], cache_type="blockcache", block_size=32 * 2 ** 20) as fh, h5py.File(fh, "r") as h:
                    df = extract(h, prod, grans[name]["aois"], name)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        print(f"bench {prod} {mode}: {name} ({grans[name]['mb']:.0f} MB) -> {len(df):,} shots in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "bench"])
    ap.add_argument("product", nargs="?", default="L2A", choices=list(PRODUCTS))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mode", default="download", choices=["download", "stream"])
    a = ap.parse_args()
    earthaccess.login(strategy="netrc")
    if a.cmd == "list":
        for p in PRODUCTS:
            g = granule_list(p)
            print(f"{p}: {len(g)} granules, {sum(v['mb'] for v in g.values()) / 1e6:.2f} TB")
    elif a.cmd == "bench":
        bench(a.product)
    else:
        run(a.product, a.shard, a.workers, a.mode)
