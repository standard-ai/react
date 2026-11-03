import numpy as np
import pytest
from scipy.stats import multivariate_normal

from reactmetric.scoring.nll import (
    _eps_star_from_cov,
    _gaussian_nll,
    _GaussianMixtureND,
    _supported_distortion_from_cov,
)


def test_gaussian_nll_matches_scipy_2d():
    rng = np.random.default_rng(0)
    cov = np.array([[0.4, 0.1], [0.1, 0.3]])
    resid = rng.normal(size=(50, 2))
    got = _gaussian_nll(resid, cov)
    want = -multivariate_normal(mean=np.zeros(2), cov=cov).logpdf(resid)
    np.testing.assert_allclose(got, want, rtol=1e-4)  # ~1e-6 cov regularization


def test_gaussian_nll_fastpath_matches_general():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 2))
    cov = a @ a.T + np.eye(2)
    resid = rng.normal(size=(30, 2))
    fast = _gaussian_nll(resid, cov)
    # force the general path by padding to D=3 with an independent unit dim
    cov3 = np.eye(3)
    cov3[:2, :2] = cov
    resid3 = np.column_stack([resid, np.zeros(len(resid))])
    general = _gaussian_nll(resid3, cov3) - _gaussian_nll(np.zeros((1, 1)), np.eye(1))[0]
    np.testing.assert_allclose(fast, general, rtol=1e-6)


def test_gaussian_nll_general_3d_matches_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + np.eye(3)
    resid = rng.normal(size=(40, 3))
    got = _gaussian_nll(resid, cov)
    want = -multivariate_normal(mean=np.zeros(3), cov=cov).logpdf(resid)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_mixture_logpdf_matches_scipy():
    weights = np.array([0.3, 0.7])
    means = np.array([[0.0, 0.0], [1.0, -1.0]])
    covs = np.array([[[0.5, 0.0], [0.0, 0.5]], [[0.3, 0.1], [0.1, 0.4]]])
    gm = _GaussianMixtureND(weights, means, covs)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 2))
    want = np.log(
        weights[0] * multivariate_normal(means[0], covs[0]).pdf(x)
        + weights[1] * multivariate_normal(means[1], covs[1]).pdf(x)
    )
    np.testing.assert_allclose(gm.log_pdf(x), want, rtol=1e-5)


def test_eps_star_zero_at_supported_distortion():
    cov = np.array([[0.5, 0.0], [0.0, 0.2]])
    d = _supported_distortion_from_cov(cov)
    assert _eps_star_from_cov(cov, d) == pytest.approx(0.0, abs=1e-9)
    # tighter task needs positive information
    assert _eps_star_from_cov(cov, d / 2) > 0.0
