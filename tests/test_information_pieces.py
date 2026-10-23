"""Pieces of the paper's formulation that have closed forms or known answers."""

import numpy as np

from reactmetric import information as ri
from reactmetric.core.trajectory import Trajectory, TrajectorySet
from reactmetric.reference import fit_climatology


def test_distortion_margin_closed_form():
    # a 2D Gaussian with RMS error d has entropy 1 + log 2pi + log(d^2 / 2); beating it needs 0
    d = 0.1
    H_d = 1.0 + np.log(2.0 * np.pi) + np.log(d * d / 2.0)
    assert ri.distortion_margin(np.array([H_d]), d)[0] == 0.0
    assert np.isclose(ri.distortion_margin(np.array([H_d + 1.3]), d)[0], 1.3)
    assert ri.distortion_margin(np.array([H_d - 2.0]), d)[0] == 0.0  # clamped


def test_noise_floor_recovers_sensor_noise():
    rng = np.random.default_rng(0)
    sigma = 0.012
    tracks = {
        i: Trajectory(id=i, positions=rng.uniform(0, 5, 2) + sigma * rng.standard_normal((400, 2)),
                      dt=0.1)
        for i in range(10)
    }
    est, n, used = ri.estimate_noise_floor(TrajectorySet(tracks))
    assert used == 10 and n > 3000
    assert abs(est - sigma) / sigma < 0.05


def test_oracle_outcome():
    assert ri.oracle_outcome(3, 80)["status"] == "recovered"
    out = ri.oracle_outcome(None, 80)
    assert out["status"] == "censored" and out["duration"] == 80 and out["onset"] is None


def _walk_stop_tracks(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        v = np.r_[np.full(200, rng.uniform(0.8, 1.5)), np.zeros(150), np.full(150, 1.1)]
        ang = rng.uniform(0, 2 * np.pi)
        step = v[:, None] * 0.1 * np.array([np.cos(ang), np.sin(ang)])
        pos = np.cumsum(step, 0) + 0.01 * rng.standard_normal((len(v), 2))
        out[i] = Trajectory(id=i, positions=pos, dt=0.1)
    return TrajectorySet(out)


def test_climatology_frames_agree():
    data = _walk_stop_tracks()
    ref = fit_climatology(data, leads=(5,), seed=0)
    sc = ref.stop_cov[5]
    assert np.isclose(sc[0, 1], 0.0) and np.isclose(sc[0, 0], sc[1, 1])  # isotropic stop
    traj = data[0]
    target, y, w, mus, covs = ref.mixture_frames(traj, 5)
    t2, y2, mu_g, S_g = ref.gaussian_frames(traj, 5)
    assert np.array_equal(target, t2)
    assert np.allclose(w.sum(1), 1.0)
    mu_m, S_m = ri.mixture_moments(w, mus, covs)
    assert np.allclose(mu_m, mu_g, atol=1e-9) and np.allclose(S_m, S_g, atol=1e-6)
    speed, heading, anchor = ref.frame_geometry(traj, 5, target)
    assert np.allclose(anchor, traj.positions[target - 5])
