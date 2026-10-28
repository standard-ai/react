"""Shared setup for the integration examples: data, the scorer, and a short report."""

from __future__ import annotations

import argparse
import os

import reactmetric as rm


def parse_args(doc: str):
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=int, default=400,
                    help="Eindhoven tracks to load (first use downloads a ~860 MB Zenodo shard)")
    ap.add_argument("--sample", action="store_true", help="use the bundled 36-track sample instead")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="integration_out")
    return ap.parse_args()


def load(args):
    data = (rm.datasets.load_eindhoven_sample() if args.sample
            else rm.datasets.load_eindhoven(download=True, max_tracks=args.tracks))
    calib, test = data.split(fraction=0.3, seed=0)
    print(f"{len(calib)} calibration / {len(test)} test tracks")
    return calib, test


def fit_scorer(calib, args):
    """REACT at lead 10, the paper's headline level, with and without the 0.1 m target."""
    return rm.ReactScorer.fit(calib, info=(4.2,), leads=(10,), distortions=(0.1,),
                              n_jobs=args.jobs)


def report(result, models):
    cols = ["model", "events", "anticipation", "km_react", "censored", "relapse3", "mean_s_mixture"]
    for d in (0.0, 0.1):
        print(f"\n{rm.viz.react_label(10, d, 4.2)}")
        tb = result.table(lead=10, info=4.2, distortion=d)
        tb = tb[tb["model"].isin(list(models) + ["oracle_B"])][cols]
        print(tb.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
