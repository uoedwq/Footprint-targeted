"""Merge the per-granule GEDI V3 parts into one Parquet file per product, area and year.

Empty parts (granule crossed only the bounding box) are skipped; `year` is added from delta_time (GEDI epoch
2018-01-01 UTC). Usage: python gedi_merge.py L2A L4A
Output: in/data/gedi_v3/merged/{L2A,L4A}_{aoi}_{year}.parquet ; out/study/v3_counts_{L2A,L4A}.csv
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "in/data/gedi_v3"
EPOCH = np.datetime64("2018-01-01T00:00:00", "us")
FLUSH_ROWS = 1_000_000                                   # rows buffered per output file before a row group is written


def year_of(delta_time):
    t = EPOCH + (np.asarray(delta_time) * 1e6).astype("timedelta64[us]")
    return t.astype("datetime64[Y]").astype(int) + 1970


def summarise(prod, t):
    """Counts per area and year, with the shares passing each quality definition."""
    df = t.to_pandas()
    if prod == "L2A":
        # NB: relaxed filter, not the conventional one (which also needs sensitivity > 0.98, night, power beams; see plots_l4a.py)
        df["v2_usual"] = (df.quality_flag_rel2 == 1) & (df.degrade_flag == 0) & (df.sensitivity >= 0.95)
        df["rel2"] = df.quality_flag_rel2 == 1
        df["rel3"] = df.quality_flag_rel3 == 1
        cols = ["rel2", "rel3", "v2_usual"]
    else:
        df["l2a_rel3"] = df.l2a_quality_flag_rel3 == 1
        df["l4a_rel3"] = df.l4a_quality_flag_rel3 == 1
        cols = ["l2a_rel3", "l4a_rel3"]
    g = df.groupby(["aoi", "year"])
    out = g.size().rename("shots").to_frame().join(g["granule"].nunique().rename("granules")).join(g[cols].sum())
    return out


def merge(prod):
    parts = sorted((BASE / prod / "parts").glob("*.parquet"))
    outdir = BASE / "merged"
    outdir.mkdir(parents=True, exist_ok=True)
    writers, buf, counts, schema, n_empty = {}, defaultdict(list), [], None, 0

    def flush(key):
        if key not in writers:
            writers[key] = pq.ParquetWriter(outdir / f"{prod}_{key[0]}_{key[1]}.parquet", schema, compression="zstd")
        writers[key].write_table(pa.concat_tables(buf.pop(key)))

    for i, f in enumerate(parts, 1):
        pf = pq.ParquetFile(f)
        if pf.metadata.num_rows == 0:
            n_empty += 1
            continue
        t = pf.read()
        t = t.replace_schema_metadata(None).append_column("year", pa.array(year_of(t["delta_time"]), pa.int16()))
        if schema is None:
            schema = t.schema
        assert t.schema.equals(schema), f.name
        for aoi in pc.unique(t["aoi"]).to_pylist():
            ta = t.filter(pc.equal(t["aoi"], aoi))
            for yr in pc.unique(ta["year"]).to_pylist():
                key = (aoi, yr)
                buf[key].append(ta.filter(pc.equal(ta["year"], yr)))
                if sum(len(x) for x in buf[key]) >= FLUSH_ROWS:
                    flush(key)
        counts.append(summarise(prod, t))
        if i % 500 == 0:
            print(f"{prod}: {i}/{len(parts)} parts", flush=True)
    for key in list(buf):
        flush(key)
    for w in writers.values():
        w.close()
    c = pd.concat(counts).groupby(level=[0, 1]).sum().reset_index()
    c.to_csv(ROOT / f"out/study/v3_counts_{prod}.csv", index=False)
    print(f"{prod}: {len(parts)} parts ({n_empty} empty) -> {len(writers)} files, {c.shots.sum():,} shots", flush=True)
    print(c.to_string(index=False), flush=True)


if __name__ == "__main__":
    for prod in sys.argv[1:] or ["L2A", "L4A"]:
        merge(prod)
