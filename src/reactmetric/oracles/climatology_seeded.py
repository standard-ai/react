"""Oracle B (climatology-seeded) and Oracle C (B + detection delay).

Oracle B is the best causal forecast given ONLY post-discontinuity information.
Told WHEN the maneuver happened, it starts from the population displacement prior
(the climatology) -- the most anything can know with zero post-change kinematic
evidence, so its information score is exactly 0 at t0. Each post-t0 sample
contributes a constant-velocity estimate of the new displacement, fused into the
prior by a Gaussian-sum Bayesian update; with < 2 post-change samples the CV term
has infinite variance and the predictive collapses to the prior (self-regularizing
cold start). Its REACT is the irreducible re-estimation lag -- the recovery FLOOR
any causal model is benchmarked against.

Oracle C adds the quickest-change-detection delay (the event's
``detection_floor_frames``): REACT_C = REACT_B + detection_floor.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..reference.base import CV_VELOCITY_WINDOW
from ..scoring.nll import fuse_with_gaussian


def _ols_velocity(points: np.ndarray) -> Optional[np.ndarray]:
    k = len(points)
    if k < 2:
        return None
    i = np.arange(k, dtype=float)
    ic = i - i.mean()
    denom = float(np.sum(ic**2))
    if denom <= 0:
        return None
    return (ic[:, None] * (points - points.mean(axis=0))).sum(axis=0) / denom


def _cv_velocity_extrap_var(k: int, lead: int, sigma: float) -> float:
    if k < 2:
        return np.inf
    i = np.arange(k, dtype=float)
    ic = i - i.mean()
    denom = float(np.sum(ic**2))
    if denom <= 0:
        return np.inf
    return (lead**2) * (sigma**2) / denom


class ClimatologySeeded:
    """Oracle B: event-conditioned recovery-floor forecaster."""

    is_event_conditioned = True

    def __init__(
        self,
        horizon: int = 15,
        history: int = 10,
        sigma_r: float = 0.05,
        pre_window: int = 30,
        post_window: int = 130,
        name: str = "oracle_B",
    ):
        self.horizon = int(horizon)
        self.history = int(history)
        self.sigma_r = float(sigma_r)
        self.pre_window = int(pre_window)
        self.post_window = int(post_window)
        self.name = name

    def info_trace(self, obs: np.ndarray, ref, lead: int, t0: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=float)
        T = len(obs)
        if lead is None or T <= lead or obs.shape[1] != 2:
            return np.array([], dtype=int), np.array([])
        cv_cov = ref.cv_residual_cov(lead)
        w = int(getattr(ref, "velocity_window", CV_VELOCITY_WINDOW))
        t0 = int(t0)
        lo = max(int(lead), t0 - self.pre_window)
        hi = min(T - 1, t0 + self.post_window)
        if hi < lo:
            return np.array([], dtype=int), np.array([])

        out_frames, out_s = [], []
        for f in range(lo, hi + 1):
            anchor = f - lead
            d = obs[f] - obs[anchor]
            if not np.all(np.isfinite(d)):
                continue
            if anchor - w >= 0:
                vel = (obs[anchor] - obs[anchor - w]) / (w * ref.dt)
                speed = float(np.hypot(vel[0], vel[1]))
                heading = float(np.arctan2(vel[1], vel[0]))
            else:
                speed, heading = 0.0, 0.0
            if not np.isfinite(speed):
                continue
            prior = ref.displacement_prior(lead, speed, heading)
            nll_clim = float(prior.nll(d[None])[0])
            k = anchor - t0 + 1
            if k < 2:
                s = 0.0
            else:
                b = _ols_velocity(obs[t0 : anchor + 1])
                if b is None or not np.all(np.isfinite(b)):
                    s = 0.0
                else:
                    est_var = _cv_velocity_extrap_var(k, lead, self.sigma_r)
                    cov = cv_cov + np.eye(2) * est_var
                    post = fuse_with_gaussian(prior, b * lead, cov)
                    s = nll_clim - float(post.nll(d[None])[0])
            out_frames.append(int(f - lead + 1))
            out_s.append(float(s))
        return np.array(out_frames, dtype=int), np.array(out_s)


def oracle_c_react(react_b_frames: float, detection_floor_frames: float) -> float:
    """REACT floor of the best causal model: Oracle B recovery + detection delay."""
    if not np.isfinite(detection_floor_frames):
        return react_b_frames
    return react_b_frames + detection_floor_frames


def oracle_c_result(oracle_b_result, events_map: Optional[Dict] = None):
    """Build an Oracle C ReactResult by adding the detection floor to Oracle B.

    Uses each measurement's stored ``detection_floor_frames``.
    """
    import copy

    res = copy.deepcopy(oracle_b_result)
    res.model_name = "oracle_C"
    for m in res.measurements:
        if not m.censored:
            m.react_frames = oracle_c_react(m.react_frames, m.detection_floor_frames)
    return res


def linear_extrapolation_variance(k: int, h: int, sigma: float) -> float:
    """Forecast variance of extrapolating an OLS fit of k noisy points h ahead."""
    if k < 2:
        return np.inf
    i = np.arange(k)
    centered = i - (k - 1) / 2.0
    denom = float(np.sum(centered**2))
    lever = (h + (k - 1) / 2.0) ** 2 / denom
    return sigma**2 * (1.0 + 1.0 / k + lever)


def analytic_reset_recovery_frames(
    sigma: float, horizon: int, nll_band: float, max_k: int = 200
) -> int:
    """Smallest k where a well-specified linear extrapolation's mean NLL re-enters the band."""
    for k in range(2, max_k):
        nlls = [
            np.log(2.0 * np.pi * linear_extrapolation_variance(k, h, sigma)) + 1.0
            for h in range(1, horizon + 1)
        ]
        if float(np.mean(nlls)) < nll_band:
            return k
    return max_k
