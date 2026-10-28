"""Forecasts produced elsewhere (another framework, another language) and saved to a file.

Write one row per (track, issue frame, lead) with the bivariate Gaussian parameters
(mu_x, mu_y, sigma_x, sigma_y, rho), where the issue frame is the last observed frame and
lead h forecasts frame issue + h. ``rm.Forecasts.from_dataframe`` reads it back; the command
line does the same in one call.

The "external" model here is a damped-velocity extrapolator written in plain numpy.

    python examples/integrations/forecasts_from_file.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
from _common import fit_scorer, load, parse_args, report

import reactmetric as rm


def damped_velocity_rows(traj, horizon=15, history=5, damping=0.9, sigma0=0.03, growth=0.06):
    """The external model: velocity from the last `history` frames, decaying by `damping`
    per frame; isotropic spread sigma0 + growth * lead (metres)."""
    pos, dt, rows = traj.positions, traj.dt, []
    for t in range(history, len(pos) - 1):          # t: issue frame = last observed frame
        v = (pos[t] - pos[t - history]) / (history * dt)
        step = np.cumsum(damping ** np.arange(1, horizon + 1))[:, None] * v * dt
        for h in range(1, horizon + 1):
            s = sigma0 + growth * h * dt
            rows.append((traj.id, t, h, *(pos[t] + step[h - 1]), s, s, 0.0))
    return rows


def main():
    args = parse_args(__doc__)
    calib, test = load(args)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "damped_velocity.csv")
    cols = ["track_id", "issue_frame", "lead", "mu_x", "mu_y", "sigma_x", "sigma_y", "rho"]
    rows = [r for t in test.ids for r in damped_velocity_rows(test[t])]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    print(f"wrote {len(rows):,} forecast rows to {path}")

    forecasts = rm.Forecasts.from_dataframe(pd.read_csv(path), id="track_id",
                                            issue="issue_frame", lead="lead",
                                            mean=["mu_x", "mu_y"],
                                            cov=["sigma_x", "sigma_y", "rho"])
    scorer = fit_scorer(calib, args)
    models = {"DampedVelocity (file)": forecasts,
              "ConstantVelocity": rm.baselines.ConstantVelocity.fit(calib)}
    result = scorer.evaluate(models, test, n_jobs=args.jobs)
    report(result, models)

    # the same from the command line: tracks and forecasts as files, REACT as JSON
    tracks_csv = os.path.join(args.out, "tracks.csv")
    data = rm.TrajectorySet({**{t: calib[t] for t in calib.ids}, **{t: test[t] for t in test.ids}})
    pd.DataFrame([{"track_id": t, "frame": i, "position_x": x, "position_y": y}
                  for t in data.ids for i, (x, y) in enumerate(data[t].positions)]
                 ).to_csv(tracks_csv, index=False)
    cmd = [sys.executable, "-m", "reactmetric.cli", "evaluate", "--data", tracks_csv,
           "--time-col", "frame", "--forecasts", path, "--info", "4.2", "--leads", "10",
           "--distortions", "0.1", "--baselines", "ConstantVelocity",
           "--jobs", str(args.jobs), "--out", os.path.join(args.out, "cli_summary.json")]
    print("\n$ reactmetric evaluate --data tracks.csv --forecasts damped_velocity.csv ...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
