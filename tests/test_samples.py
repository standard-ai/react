"""Sample-based models: kernel densities with a bandwidth fitted on calibration tracks."""

import numpy as np

import reactmetric as rm
from reactmetric.core.trajectory import Trajectory, TrajectorySet


def _tracks(n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        step = 0.1 * rng.uniform(0.8, 1.4) * np.array([1.0, 0.2])
        pos = np.cumsum(np.tile(step, (300, 1)), 0) + 0.03 * rng.standard_normal((300, 2))
        out[f"t{i}"] = Trajectory(id=f"t{i}", positions=pos, dt=0.1)
    return TrajectorySet(out)


def sampler(traj, K=16, H=5, spread=0.05):
    cv = rm.baselines.ConstantVelocity()
    p = cv.predict(traj)
    if p is None:
        return None
    rng = np.random.default_rng(0)
    m = p.means[:, :H]
    return p.issue_frames, m[:, :, None] + spread * rng.standard_normal(m.shape[:2] + (K, 2))


def test_from_samples_is_a_kernel_density():
    rng = np.random.default_rng(1)
    s = rng.standard_normal((3, 2, 8, 2))
    mp = rm.MixturePredictive.from_samples(np.arange(3), s, bandwidth=0.3)
    assert mp.component_covs.shape == (3, 2, 8, 2, 2)
    assert np.allclose(mp.component_covs[..., 0, 0], 0.09)
    assert np.allclose(mp.means, s.mean(2))  # moment-matched mean of the kernels


def test_fitted_bandwidth_beats_scott():
    calib = _tracks(8, seed=2)
    model = rm.SampleForecaster(sampler, name="noisy").fit(calib, events=False)
    assert model.bandwidth_.shape == (5,) and np.all(model.bandwidth_ > 0)
    i = np.searchsorted(model.grid_, model.bandwidth_[2])
    assert model.fit_curve_[2].max() == model.fit_curve_[2, i]
    test = _tracks(3, seed=3)
    for tid in test.ids:
        fitted = model.predict(test[tid]).nll_matrix(test[tid].positions)
        scott = rm.SampleForecaster(sampler, bandwidth=None).predict(test[tid]).nll_matrix(
            test[tid].positions)
        assert np.nanmean(fitted) <= np.nanmean(scott) + 1e-9
