import numpy as np
from scipy.stats import multivariate_normal

from reactmetric.scoring.nll import _gaussian_nll


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
