"""MixturePredictive: native Gaussian-mixture forecasts (paper section 3.2)."""

import numpy as np
import pytest

from reactmetric.core.predictive import MixturePredictive, Predictive


def _mix(n=6, H=4, J=3, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(J), n)
    mu = rng.normal(size=(n, H, J, 2))
    A = rng.normal(size=(n, H, J, 2, 2)) * 0.3
    cov = A @ np.swapaxes(A, -1, -2) + 0.05 * np.eye(2)
    return MixturePredictive(np.arange(n) + 10, w, mu, cov)


def test_moment_matched_gaussian_is_exact():
    m = _mix()
    w = m.weights
    mean = np.einsum("nk,nhkd->nhd", w, m.component_means)
    d = m.component_means - mean[:, :, None]
    cov = np.einsum("nk,nhkij->nhij", w, m.component_covs)
    cov = cov + np.einsum("nk,nhki,nhkj->nhij", w, d, d)
    assert np.allclose(m.means, mean) and np.allclose(m.covs, cov)


def test_nll_is_the_mixture_density():
    m = _mix()
    obs = np.random.default_rng(1).normal(size=(30, 2))
    nll = m.nll_matrix(obs)
    i, h = 2, 1
    y = obs[m.issue_frames[i] + h]
    dens = 0.0
    for j in range(m.weights.shape[1]):
        S = m.component_covs[i, h, j]
        r = y - m.component_means[i, h, j]
        dens += m.weights[i, j] * np.exp(-0.5 * r @ np.linalg.solve(S, r)) / (
            2 * np.pi * np.sqrt(np.linalg.det(S)))
    assert np.isclose(nll[i, h], -np.log(dens))
    assert np.isnan(nll[-1, -1]) or m.issue_frames[-1] + 3 < len(obs)


def test_one_component_equals_gaussian():
    g = Predictive(np.arange(5), np.zeros((5, 3, 2)), np.tile(np.eye(2), (5, 3, 1, 1)))
    m = MixturePredictive(g.issue_frames, np.ones((5, 1)), g.means[:, :, None], g.covs[:, :, None])
    obs = np.random.default_rng(2).normal(size=(10, 2))
    assert np.allclose(m.nll_matrix(obs), g.nll_matrix(obs), equal_nan=True)
    t, w, mu, S = m.mixture_at_lead(2)
    assert np.array_equal(t, g.at_lead(2)[0]) and w.shape == (5, 1) and mu.shape == (5, 1, 2)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        MixturePredictive(
            np.arange(3), np.ones((3, 2)), np.zeros((3, 4, 2)), np.zeros((3, 4, 2, 2))
        )
