"""`reactmetric evaluate` end to end on files (one lead, one level)."""

import json

import numpy as np
import pytest

from reactmetric import baselines
from reactmetric.cli import main

pd = pytest.importorskip("pandas")


def _frames(n=24, seed=0):
    rng = np.random.default_rng(seed)
    rows, fc = [], []
    cv = baselines.ConstantVelocity()
    from reactmetric.core.trajectory import Trajectory

    for i in range(n):
        v = np.r_[np.full(200, rng.uniform(0.9, 1.4)), np.zeros(150), np.full(200, 1.1)]
        ang = rng.uniform(0, 2 * np.pi)
        pos = np.cumsum(v[:, None] * 0.1 * np.array([np.cos(ang), np.sin(ang)]), 0)
        pos = pos + 0.01 * rng.standard_normal(pos.shape)
        rows += [{"track_id": i, "position_x": x, "position_y": y} for x, y in pos]
        p = cv.predict(Trajectory(id=i, positions=pos, dt=0.1))
        for k, iss in enumerate(p.issue_frames):
            for h in range(p.horizon):
                S = p.covs[k, h]
                sx, sy = np.sqrt(S[0, 0]), np.sqrt(S[1, 1])
                fc.append({"track_id": i, "issue_frame": int(iss), "lead": h + 1,
                           "mu_x": p.means[k, h, 0], "mu_y": p.means[k, h, 1],
                           "sigma_x": sx, "sigma_y": sy, "rho": S[0, 1] / (sx * sy)})
    return pd.DataFrame(rows), pd.DataFrame(fc)


def test_evaluate_from_files(tmp_path):
    data, fc = _frames()
    data.to_csv(tmp_path / "tracks.csv", index=False)
    fc.to_csv(tmp_path / "fc.csv", index=False)
    rc = main([
        "evaluate", "--data", str(tmp_path / "tracks.csv"), "--forecasts", str(tmp_path / "fc.csv"),
        "--info", "4.2", "--leads", "5", "--distortions", "--baselines", "ConstantVelocity",
        "--out", str(tmp_path / "out.json"), "--records", str(tmp_path / "rec.csv"),
    ])
    assert rc == 0
    out = json.load(open(tmp_path / "out.json"))
    assert {row["model"] for row in out["summary"]} == {"model", "ConstantVelocity", "oracle_B"}
    assert (tmp_path / "rec.csv").exists()
