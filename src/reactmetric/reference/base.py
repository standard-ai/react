"""Shared fitting helpers for the climatology."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..constants import COV_REGULARIZATION

CV_VELOCITY_WINDOW = 5
MAX_FIT_SAMPLES = 200_000


def fit_cv_residual_cov(
    arrays: Dict[object, np.ndarray],
    track_ids: Sequence[object],
    lead: int,
    dt: float,
    window: int = CV_VELOCITY_WINDOW,
    max_samples: int = MAX_FIT_SAMPLES,
) -> np.ndarray:
    """Empirical covariance of constant-velocity extrapolation residuals."""
    w = window
    chunks = []
    dim = next(iter(arrays.values())).shape[1]
    for tid in track_ids:
        obs = arrays[tid]
        t = len(obs)
        if t <= lead + w:
            continue
        anchors = obs[w : t - lead]
        vel = (obs[w : t - lead] - obs[: t - lead - w]) / (w * dt)
        resid = obs[lead + w :] - (anchors + vel * (lead * dt))
        finite = np.isfinite(resid).all(axis=1)
        if finite.any():
            chunks.append(resid[finite])
    if not chunks:
        return np.eye(dim) * COV_REGULARIZATION
    pooled = np.concatenate(chunks, axis=0)
    if len(pooled) > max_samples:
        pooled = pooled[:max_samples]
    cov = np.cov(pooled.T)
    return np.atleast_2d(cov)


def regularize_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float).copy()
    dim = cov.shape[0]
    idx = np.arange(dim)
    cov[idx, idx] += COV_REGULARIZATION
    if not np.all(np.isfinite(cov)) or float(np.linalg.det(cov)) <= 0.0:
        return np.eye(dim) * max(COV_REGULARIZATION, 1e-3)
    return cov
