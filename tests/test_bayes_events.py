"""BayesianChangepoints.tuned_sg, the paper's default event source (section 3.4)."""

import numpy as np

from reactmetric.core.trajectory import Trajectory
from reactmetric.events import BayesianChangepoints

DT = 0.1


def _track(speeds, heading=0.0, noise=0.01, seed=0):
    """Positions of a walker with per-frame speed `speeds` (m/s) plus white measurement noise."""
    rng = np.random.default_rng(seed)
    step = np.asarray(speeds, float)[:, None] * DT * np.array([np.cos(heading), np.sin(heading)])
    pos = np.cumsum(step, axis=0) + noise * rng.standard_normal((len(speeds), 2))
    return Trajectory(id="t", positions=pos, dt=DT)


def test_detects_a_stop_near_the_change():
    traj = _track(np.r_[np.full(300, 1.2), np.zeros(300)])
    events = BayesianChangepoints.tuned_sg().detect(traj)
    assert len(events) == 1
    assert events[0].event_type == "stop"
    assert abs(events[0].t0 - 300) <= 10


def test_no_events_on_steady_walking():
    traj = _track(np.full(600, 1.2), heading=0.7)
    assert BayesianChangepoints.tuned_sg().detect(traj) == []


def test_tuned_settings_are_the_papers():
    det = BayesianChangepoints.tuned_sg()
    assert det.hazard == 9e-4 and det.config.min_size == 30
    assert det.smooth_window == 5 and det.velocity_channels and det.relabel_heading
