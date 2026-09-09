#!/usr/bin/env python
"""Export a subset of a cached Eindhoven parquet shard to CSV.

Cowork sessions cannot reach Zenodo (and PyPI is blocked by org egress policy, so
pyarrow cannot be installed there). This script runs on the machine that already has
the cache and the project venv, and writes a compact CSV that any environment can read.

Usage (from the repo root):

    reactmetric/.venv/bin/python reactmetric/scripts/export_eindhoven_csv.py

Writes reactmetric/.data/eindhoven_<n>tracks.csv in the bundled-sample schema
(time_ms, object_identifier, x_position_mm, y_position_mm), which loads through the
same path as reactmetric.datasets.load_eindhoven_sample().
"""
from __future__ import annotations

import argparse
import os

import pyarrow.parquet as pq

CACHE = os.path.expanduser("~/.cache/reactmetric/eindhoven")
DEFAULT_SRC = os.path.join(CACHE, "Eindhoven_centraal_trajectories_01_10_day02.parquet")
COLS = ["time_ms", "object_identifier", "x_position_mm", "y_position_mm"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="parquet shard to read")
    ap.add_argument("--tracks", type=int, default=3000, help="number of tracks to keep")
    ap.add_argument("--min-length", type=int, default=40, help="drop shorter tracks")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(here, ".data", f"eindhoven_{args.tracks}tracks.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    pf = pq.ParquetFile(args.src)
    missing = set(COLS) - set(pf.schema.names)
    if missing:
        raise SystemExit(f"{args.src} is missing columns: {sorted(missing)}")

    keep: set[int] = set()
    counts: dict[int, int] = {}
    buf: dict[int, list] = {}
    done = False
    for batch in pf.iter_batches(batch_size=200_000, columns=COLS):
        d = batch.to_pydict()
        for t, tid, x, y in zip(d["time_ms"], d["object_identifier"],
                                d["x_position_mm"], d["y_position_mm"]):
            tid = int(tid)
            if tid not in keep:
                if len(keep) >= args.tracks:
                    continue
                keep.add(tid)
            buf.setdefault(tid, []).append((t, tid, x, y))
            counts[tid] = counts.get(tid, 0) + 1
        if len(keep) >= args.tracks and min(counts.values()) >= args.min_length:
            done = True
        if done and len(buf) >= args.tracks:
            break

    kept = {k: v for k, v in buf.items() if len(v) >= args.min_length}
    n_rows = sum(len(v) for v in kept.values())
    with open(out, "w", newline="") as f:
        f.write(",".join(COLS) + "\n")
        for tid in sorted(kept):
            for row in sorted(kept[tid], key=lambda r: r[0]):
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

    size_mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}")
    print(f"  {len(kept)} tracks, {n_rows} rows, {size_mb:.1f} MB")
    print(f"  median track length {sorted(len(v) for v in kept.values())[len(kept)//2]} frames")


if __name__ == "__main__":
    main()
