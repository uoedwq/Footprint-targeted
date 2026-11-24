"""Footprint table for training and QC experiments — GEDI V3 L2A (+ L4A AGBD) with the 2021 predictors of the
25 m cell containing each footprint, plus
neighbourhood context (common.CTX). Missing values are NaN.

Kept: degrade_flag == 0, quality_flag_rel2 == 1 or quality_flag_rel3 == 1, Landsat tree cover >= 10% (loose; the forest
mask is applied later from the predictor stack: WorldCover 2021 tree fraction and Hansen loss year). All years.
Usage: python training_table.py NE SW NEtest SWtest
Output: in/data/training/{aoi}.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features_at

ROOT = Path(__file__).resolve().parents[1]
V3DIR = ROOT / "in/data/gedi_v3/merged"
PRED = ROOT / "in/data/predictors"
OUT = ROOT / "in/data/training"
PX, TILE = 25, 640
L2COLS = ["shot_number", "lon", "lat", "delta_time", "year", "beam", "granule", "quality_flag_rel2", "quality_flag_rel3",
          "degrade_flag", "sensitivity", "sensitivity_a2", "solar_elevation", "leaf_off_flag", "landsat_treecover",
          "pft_class", "region_class", "worldcover_class", "l2_algrunflag", "rx_maxamp", "sd_corrected",
          "stale_return_flag", "surface_flag", "elev_highestreturn", "elev_lowestmode", "digital_elevation_model",
          "num_detectedmodes", "rh50", "rh60", "rh80", "rh95", "rh98"]


def load(aoi):
    flt = (pc.field("degrade_flag") == 0) & ((pc.field("quality_flag_rel2") == 1) | (pc.field("quality_flag_rel3") == 1)) \
        & (pc.field("landsat_treecover") >= 10)
    parts = []
    for f in sorted(V3DIR.glob(f"L2A_{aoi}_*.parquet")):
        t = pq.read_table(f, columns=L2COLS, filters=flt).to_pandas()
        l4f = V3DIR / f.name.replace("L2A_", "L4A_")
        if l4f.exists():
            l4 = pq.read_table(l4f, columns=["shot_number", "agbd", "l4a_quality_flag_rel3"]).to_pandas()
            t = t.merge(l4, on="shot_number", how="left")
        else:
            t["agbd"], t["l4a_quality_flag_rel3"] = np.nan, np.nan
        parts.append(t)
        print(f"{aoi} {f.name}: {len(t):,}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["granule"] = pd.factorize(d.granule)[0].astype(np.int32)
    return d


def extract(aoi, d):
    """Predictors and neighbourhood context of the 25 m cell of each footprint (common.features_at)."""
    crs = pd.read_csv(PRED / aoi / "tiles.csv").crs.iloc[0]
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(d.lon.to_numpy(), d.lat.to_numpy())
    d = d.reset_index(drop=True)
    d["x"], d["y"] = x, y
    f = features_at(aoi, x, y)
    print(f"{aoi}: predictors for {int(np.isfinite(f.vv).sum()):,} of {len(d):,} footprints", flush=True)
    return pd.concat([d, f], axis=1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for aoi in sys.argv[1:]:
        d = extract(aoi, load(aoi))
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False), OUT / f"{aoi}.parquet", compression="zstd")
        print(f"{aoi}: wrote {len(d):,} rows", flush=True)
