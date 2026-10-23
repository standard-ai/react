"""Command-line interface: `reactmetric evaluate ...`.

Runs the one-call benchmark from files:

    reactmetric evaluate --data tracks.parquet --forecasts my_preds.parquet \\
        --info 3 4.2 6 --leads 5 10 --distortions 0.1 --baselines ConstantVelocity IMM \\
        --out react_results.json --records react_records.csv --plots out/
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _load_data(path: str, id_col: str, time_col: Optional[str], coords, dt: float):
    import pandas as pd

    from .core.trajectory import TrajectorySet

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return TrajectorySet.from_dataframe(df, id=id_col, coords=coords, time=time_col, dt=dt)


def _load_forecasts(path: str, args):
    from .io import Forecasts

    if path.endswith(".npz"):
        return Forecasts.from_npz(path)
    import pandas as pd

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return Forecasts.from_dataframe(
        df, id=args.id, issue=args.issue_col, lead=args.lead_col,
        mean=args.mean_cols, cov=args.cov_cols,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="reactmetric")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="Run a REACT benchmark from files.")
    ev.add_argument("--data", required=True, help="trajectories parquet/csv (Argoverse-2 schema)")
    ev.add_argument("--forecasts", required=True, help="forecasts parquet/csv/npz")
    ev.add_argument("--id", default="track_id")
    ev.add_argument("--time-col", default=None)
    ev.add_argument("--coords", nargs="+", default=["position_x", "position_y"])
    ev.add_argument("--dt", type=float, default=0.1)
    ev.add_argument("--issue-col", default="issue_frame")
    ev.add_argument("--lead-col", default="lead")
    ev.add_argument("--mean-cols", nargs="+", default=["mu_x", "mu_y"])
    ev.add_argument("--cov-cols", nargs="+", default=["sigma_x", "sigma_y", "rho"])
    ev.add_argument("--calibration-fraction", type=float, default=0.3)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--info", type=float, nargs="+", default=[3.0, 4.2, 6.0],
                    help="assumed information levels I_traj (nats)")
    ev.add_argument("--leads", type=int, nargs="+", default=[1, 5, 10, 15])
    ev.add_argument("--distortions", type=float, nargs="*", default=[0.1, 0.3],
                    help="target distances d (m); d = infinity is always included")
    ev.add_argument("--baselines", nargs="*", default=["ConstantVelocity", "CTRV", "IMM"])
    ev.add_argument("--jobs", type=int, default=1)
    ev.add_argument("--out", default="react_results.json", help="summary JSON")
    ev.add_argument("--records", default=None, help="per-event records CSV")
    ev.add_argument("--plots", default=None, help="directory to write plots")

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        from .benchmark import benchmark

        data = _load_data(args.data, args.id, args.time_col, args.coords, args.dt)
        forecasts = _load_forecasts(args.forecasts, args)
        calib, test = data.split(fraction=args.calibration_fraction, seed=args.seed)

        result = benchmark(
            {"model": forecasts}, test, calib, baselines=args.baselines, info=args.info,
            leads=args.leads, distortions=args.distortions, n_jobs=args.jobs, seed=args.seed,
        )
        result.to_json(args.out)
        print(f"Wrote {args.out}")
        if args.records:
            result.to_csv(args.records)
            print(f"Wrote {args.records}")
        if args.plots:
            from .viz import km_curves

            os.makedirs(args.plots, exist_ok=True)
            for lead in args.leads:
                path = os.path.join(args.plots, f"km_lead{lead}.png")
                km_curves(result, lead=lead, info=args.info[len(args.info) // 2], save=path)
            print(f"Wrote plots to {args.plots}/")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
