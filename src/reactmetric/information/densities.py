"""Gaussian and Gaussian-mixture densities, moments and entropies, frame by frame."""

from __future__ import annotations

import numpy as np

_EYE2 = np.eye(2)


def gaussian_logpdf(y, mu, S):
    """log N(y; mu, S), rowwise."""
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (y - mu)[..., None])[..., 0]
    return (
        -0.5 * (z * z).sum(-1)
        - np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
        - 0.5 * y.shape[-1] * np.log(2.0 * np.pi)
    )


def gh_nodes_2d(n=5):
    """Probabilists' Gauss-Hermite nodes (n*n, 2) and weights (n*n,) for E over N(0, I_2)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    X, Y = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w)
    return np.column_stack([X.ravel(), Y.ravel()]), W.ravel()


def mixture_prep(weights, means, covs):
    """Precompute per-frame component inverses and log normalisers. weights (n, K)."""
    covs = covs + 1e-10 * _EYE2
    inv = np.linalg.inv(covs)
    logdet = np.linalg.slogdet(covs)[1]
    lognorm = np.log(np.clip(weights, 1e-300, None)) - 0.5 * logdet - np.log(2.0 * np.pi)
    return means, inv, lognorm


def mixture_logpdf(Y, prep):
    """log p_mix at points Y (n, ..., 2), mixture parameters per frame n."""
    means, inv, lognorm = prep
    n, K = lognorm.shape
    extra = Y.ndim - 2
    out = []
    for k in range(K):
        mu = means[:, k].reshape((n,) + (1,) * extra + (2,))
        P = inv[:, k].reshape((n,) + (1,) * extra + (2, 2))
        d = Y - mu
        q = np.einsum("...i,...ij,...j->...", d, P, d)
        out.append(lognorm[:, k].reshape((n,) + (1,) * extra) - 0.5 * q)
    out = np.stack(out, 0)
    peak = out.max(0)
    return peak + np.log(np.exp(out - peak).sum(0))


def mixture_moments(weights, means, covs):
    mu = np.einsum("nk,nki->ni", weights, means)
    d = means - mu[:, None]
    S = np.einsum("nk,nkij->nij", weights, covs + np.einsum("nki,nkj->nkij", d, d))
    return mu, S


def mixture_entropy(weights, means, covs, nodes, gw):
    """Differential entropy of each frame's mixture, by Gauss-Hermite within each component."""
    prep = mixture_prep(weights, means, covs)
    L = np.linalg.cholesky(covs + 1e-10 * _EYE2)  # (n, K, 2, 2)
    Y = means[:, :, None, :] + np.einsum("nkij,gj->nkgi", L, nodes)  # (n, K, G, 2)
    lp = mixture_logpdf(Y, prep)  # (n, K, G)
    return -np.einsum("nk,nkg,g->n", weights, lp, gw)


def gaussian_entropy(S):
    return 1.0 + np.log(2.0 * np.pi) + 0.5 * np.linalg.slogdet(S)[1]
