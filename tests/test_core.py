import numpy as np

from reactmetric.core import Predictive, Trajectory, TrajectorySet


def test_trajectory_shapes():
    t = Trajectory(np.zeros((10, 2)), dt=0.1, id=3)
    assert t.T == 10 and t.D == 2 and t.dt == 0.1
    t1 = Trajectory(np.arange(5.0))  # 1D promoted to (5,1)
    assert t1.D == 1 and t1.T == 5


def test_trajectory_velocity_constant():
    pos = np.cumsum(np.ones((20, 2)) * np.array([0.2, -0.1]), axis=0)
    v = Trajectory(pos, dt=0.1).velocity(window=5)
    expected = np.broadcast_to(np.array([2.0, -1.0]), v[5:].shape)
    np.testing.assert_allclose(v[5:], expected, rtol=1e-6)


def test_trajectoryset_split_disjoint():
    ts = TrajectorySet.from_arrays({i: np.zeros((5, 2)) for i in range(10)})
    a, b = ts.split(0.3, seed=0)
    assert len(a) == 3 and len(b) == 7
    assert set(a.ids).isdisjoint(b.ids)


def test_predictive_from_samples_moment():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(4, 3, 5000, 2))
    pred = Predictive.from_samples(np.arange(4), samples, method="moment")
    assert pred.means.shape == (4, 3, 2)
    assert pred.covs.shape == (4, 3, 2, 2)
    np.testing.assert_allclose(pred.means, 0.0, atol=0.1)
    np.testing.assert_allclose(pred.covs[0, 0], np.eye(2), atol=0.15)


def test_predictive_isotropic_and_nll_matrix():
    issue = np.array([0, 1])
    means = np.zeros((2, 3, 2))
    pred = Predictive.isotropic(issue, means, sigma=1.0)
    obs = np.zeros((6, 2))
    nll = pred.nll_matrix(obs)
    assert nll.shape == (2, 3)
    # NLL of the mean under unit isotropic 2D Gaussian = log(2*pi)
    np.testing.assert_allclose(nll[~np.isnan(nll)], np.log(2 * np.pi), rtol=1e-6)

