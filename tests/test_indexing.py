"""Issue frames follow the field's convention (F4): the issue frame is the last observed frame,
and the lead-h forecast targets issue + h."""

import numpy as np
import pytest

from reactmetric import baselines
from reactmetric.core.predictive import MixturePredictive, Predictive
from reactmetric.core.trajectory import Trajectory


def test_at_lead_targets_issue_plus_lead():
    p = Predictive(np.array([10, 11]), np.zeros((2, 3, 2)), np.tile(np.eye(2), (2, 3, 1, 1)))
    assert list(p.at_lead(1)[0]) == [11, 12] and list(p.at_lead(3)[0]) == [13, 14]
    m = MixturePredictive(p.issue_frames, np.ones((2, 1)), p.means[:, :, None], p.covs[:, :, None])
    assert list(m.mixture_at_lead(2)[0]) == [12, 13]


def test_nll_matrix_scores_issue_plus_lead():
    obs = np.zeros((20, 2))
    obs[13] = [3.0, 0.0]
    p = Predictive(np.array([10]), np.zeros((1, 3, 2)), np.tile(np.eye(2), (1, 3, 1, 1)))
    nll = p.nll_matrix(obs)
    assert nll[0, 2] > nll[0, 1]  # lead 3 targets frame 13, the outlier


def _walk(seed=0, n=120):
    rng = np.random.default_rng(seed)
    v = 1.2 * np.array([np.cos(0.3), np.sin(0.3)])
    return np.cumsum(np.tile(v * 0.1, (n, 1)), 0) + 0.01 * rng.standard_normal((n, 2))


@pytest.mark.parametrize("name", ["ConstantVelocity", "CTRV", "IMM", "GRUPosition"])
def test_baselines_are_causal_at_the_issue_frame(name):
    """The forecast issued at frame t depends on observation t, not on t + 1."""
    if name.startswith("GRU"):
        pytest.importorskip("torch")
    model = baselines.build(name)
    pos = _walk()
    base = model.predict(Trajectory(id=0, positions=pos, dt=0.1))
    i = len(base) // 2
    t = int(base.issue_frames[i])
    later = pos.copy()
    later[t + 1:] += 0.5
    now = pos.copy()
    now[t:] += 0.5
    p_later = model.predict(Trajectory(id=0, positions=later, dt=0.1))
    p_now = model.predict(Trajectory(id=0, positions=now, dt=0.1))
    k = int(np.searchsorted(p_later.issue_frames, t))
    assert np.allclose(p_later.means[k], base.means[i])
    assert not np.allclose(p_now.means[k], base.means[i])
