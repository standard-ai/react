"""Internal log-score (negative log-likelihood) helpers, for any state dimension D.

`_gaussian_nll` is the Gaussian NLL evaluator behind ``Predictive.nll_matrix``, the kinematic
reference and the IMM, with a closed-form fast path for D == 2.
"""

from __future__ import annotations

import numpy as np

from ..constants import COV_REGULARIZATION

_LOG_2PI = float(np.log(2.0 * np.pi))


def _solve_quadratic_2d(resid: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Mahalanobis distance and log-det for batched 2x2 covs."""
    a = cov[:, 0, 0] + COV_REGULARIZATION
    b = cov[:, 0, 1]
    c = cov[:, 1, 0]
    d = cov[:, 1, 1] + COV_REGULARIZATION
    det = a * d - b * c
    dx, dy = resid[:, 0], resid[:, 1]
    mahal = (d * dx * dx - (b + c) * dx * dy + a * dy * dy) / det
    return mahal, np.log(det)


def _gaussian_nll(resid: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """NLL of zero-mean residuals under a (shared or per-row) covariance.

    Args:
        resid: (N, D) residuals (realized minus predicted).
        cov: (D, D) shared covariance, or (N, D, D) per-row covariances.

    Returns:
        (N,) negative log-likelihoods in nats.
    """
    resid = np.atleast_2d(np.asarray(resid, dtype=float))
    cov = np.asarray(cov, dtype=float)
    n, dim = resid.shape
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, (n, dim, dim))

    if dim == 2:
        mahal, logdet = _solve_quadratic_2d(resid, cov)
        return 0.5 * mahal + 0.5 * logdet + dim * 0.5 * _LOG_2PI

    # General D: regularize, batched solve + slogdet.
    reg = cov + np.eye(dim) * COV_REGULARIZATION
    sign, logdet = np.linalg.slogdet(reg)
    sol = np.linalg.solve(reg, resid[:, :, None])[:, :, 0]   # (N, D)
    mahal = np.einsum("nd,nd->n", resid, sol)
    bad = sign <= 0
    out = 0.5 * mahal + 0.5 * logdet + dim * 0.5 * _LOG_2PI
    out[bad] = np.nan
    return out
