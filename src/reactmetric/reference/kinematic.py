"""Kinematic-conditional displacement climatology (2D).

p(displacement | speed, heading): the empirical conditional climatology -- among
observed people moving this fast in this direction, what did their lead-step
futures look like. This is NOT a constant-velocity model; it is the body-frame
conditional slice (rotation invariance: heading only sets the frame; translation
invariance: pooled over positions), with speed a continuous covariate. Each
per-(speed, lead) future is a single anisotropic Gaussian (maximum entropy given
its conditional mean and covariance).

Because speed/heading come from a causal lagging window, the reference is still
surprised by genuine maneuvers; only trivially-predictable persistence is
conditioned out. A separate isotropic stopped regime handles speed < stop_speed.

Heading is a 2D notion, so this mode requires D == 2; the facade falls back to the
global climatology for other dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from ..constants import DEFAULT_DT
from ..scoring.nll import (
    LOG_2PI,
    GaussianMixtureND,
    eps_star_from_cov,
    gaussian_nll,
    supported_distortion_from_cov,
)
from .base import (
    CV_VELOCITY_WINDOW,
    MAX_FIT_SAMPLES,
    cv_nll_trace_from_cov,
    fit_cv_residual_cov,
    regularize_cov,
)


def _weighted_moments(samples: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = weights / max(float(weights.sum()), 1e-12)
    mu = (w[:, None] * samples).sum(axis=0)
    d = samples - mu
    cov = np.einsum("n,ni,nj->ij", w, d, d)
    return mu, cov


@dataclass
class KinematicClimatology:
    """Conditional displacement reference p(displacement | speed, heading), 2D."""

    leads: Tuple[int, ...]
    dim: int
    dt: float
    velocity_window: int
    stop_speed: float
    speed_kernel_bw: float
    transition_band: float
    speed_grid: Dict[int, np.ndarray]
    mu_grid: Dict[int, np.ndarray]
    cov_grid: Dict[int, np.ndarray]
    stop_cov: Dict[int, np.ndarray]
    marginal_cov: Dict[int, np.ndarray]
    cv_covs: Dict[int, np.ndarray]
    n_samples: Dict[int, int] = field(default_factory=dict)
    n_calib_tracks: int = 0

    def _predict_moving(self, lead: int, speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grid = self.speed_grid[lead]
        mu_grid = self.mu_grid[lead]
        cov_grid = self.cov_grid[lead]
        mu = np.empty((len(speed), 2))
        mu[:, 0] = np.interp(speed, grid, mu_grid[:, 0])
        mu[:, 1] = np.interp(speed, grid, mu_grid[:, 1])
        c00 = np.interp(speed, grid, cov_grid[:, 0, 0])
        c01 = np.interp(speed, grid, cov_grid[:, 0, 1])
        c11 = np.interp(speed, grid, cov_grid[:, 1, 1])
        cov = np.empty((len(speed), 2, 2))
        cov[:, 0, 0] = c00
        cov[:, 0, 1] = c01
        cov[:, 1, 0] = c01
        cov[:, 1, 1] = c11
        return mu, cov

    def nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(traj.positions if hasattr(traj, "positions") else traj, dtype=float)
        t = len(obs)
        w = self.velocity_window
        if lead not in self.speed_grid or t <= lead + w:
            return np.array([], dtype=int), np.array([])
        frames = np.arange(lead + w, t)
        anchors = frames - lead
        vel = (obs[anchors] - obs[anchors - w]) / (w * self.dt)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        disp = obs[frames] - obs[anchors]
        cth, sth = np.cos(heading), np.sin(heading)
        disp_body = np.column_stack(
            [cth * disp[:, 0] + sth * disp[:, 1], -sth * disp[:, 0] + cth * disp[:, 1]]
        )
        mu, cov = self._predict_moving(lead, speed)
        nll_move = gaussian_nll(disp_body - mu, cov)
        nll_stop = gaussian_nll(disp, self.stop_cov[lead])
        lam = np.clip((speed - self.stop_speed) / max(self.transition_band, 1e-9), 0.0, 1.0)
        eps = 1e-12
        logp_move = np.log(lam + eps) - nll_move
        logp_stop = np.log(1.0 - lam + eps) - nll_stop
        peak = np.maximum(logp_move, logp_stop)
        logp = peak + np.log(np.exp(logp_move - peak) + np.exp(logp_stop - peak))
        nll = -logp
        ok = np.isfinite(nll)
        return frames[ok].astype(int), nll[ok]

    def cv_nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(traj.positions if hasattr(traj, "positions") else traj, dtype=float)
        return cv_nll_trace_from_cov(obs, lead, self.dt, self.cv_covs[lead])

    def score_trace(self, traj, lead: int, scoring: str = "nll"):
        if scoring == "nll":
            return self.nll_trace(traj, lead)
        from ..scoring.crps import gaussian_score

        obs = np.asarray(traj.positions if hasattr(traj, "positions") else traj, dtype=float)
        t = len(obs)
        w = self.velocity_window
        if lead not in self.speed_grid or t <= lead + w:
            return np.array([], dtype=int), np.array([])
        frames = np.arange(lead + w, t)
        anchors = frames - lead
        vel = (obs[anchors] - obs[anchors - w]) / (w * self.dt)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        mu_b, cov_b = self._predict_moving(lead, speed)
        c, s = np.cos(heading), np.sin(heading)
        rot = np.zeros((len(frames), 2, 2))
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c
        mu_world = np.einsum("nij,nj->ni", rot, mu_b)
        cov_world = np.einsum("nij,njk,nlk->nil", rot, cov_b, rot)
        mu_abs = obs[anchors] + mu_world
        resid = obs[frames] - mu_abs
        score = gaussian_score(resid, cov_world, scoring)
        ok = np.isfinite(score)
        return frames[ok].astype(int), score[ok]

    def conditional_gaussian(
        self, lead: int, speed: float, heading: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu_b, cov_b = self._predict_moving(lead, np.array([float(speed)]))
        mu_b, cov_b = mu_b[0], cov_b[0]
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        return rot @ mu_b, rot @ cov_b @ rot.T

    def displacement_prior(self, lead: int, speed: float, heading: float) -> GaussianMixtureND:
        mu_w, cov_w = self.conditional_gaussian(lead, float(speed), float(heading))
        lam = float(np.clip(
            (speed - self.stop_speed) / max(self.transition_band, 1e-9), 0.0, 1.0
        ))
        eps = 1e-12
        weights = np.array([lam + eps, 1.0 - lam + eps])
        means = np.vstack([mu_w, np.zeros(2)])
        covs = np.stack([cov_w, self.stop_cov[lead]])
        return GaussianMixtureND(weights=weights, means=means, covs=covs)

    def cv_residual_cov(self, lead: int) -> np.ndarray:
        return self.cv_covs.get(lead, np.zeros((2, 2)))

    def eps_star(self, lead: int, d_meters: float) -> float:
        return eps_star_from_cov(self.marginal_cov[lead], d_meters)

    def supported_distortion(self, lead: int) -> float:
        return supported_distortion_from_cov(self.marginal_cov[lead])

    def metadata(self) -> Dict:
        out = {
            "mode": "kinematic",
            "dim": self.dim,
            "velocity_window": self.velocity_window,
            "stop_speed": self.stop_speed,
            "speed_kernel_bw": self.speed_kernel_bw,
            "transition_band": self.transition_band,
            "n_calib_tracks": self.n_calib_tracks,
            "leads": list(self.leads),
            "per_lead": {},
        }
        for lead in self.leads:
            cov = self.marginal_cov[lead]
            out["per_lead"][str(lead)] = {
                "n_samples": int(self.n_samples.get(lead, 0)),
                "mixture_covariance": cov.tolist(),
                "differential_entropy_gaussian_nats": float(
                    1.0 + LOG_2PI + 0.5 * np.log(np.linalg.det(cov))
                ),
                "cv_residual_covariance": self.cv_covs[lead].tolist(),
            }
        return out


def fit_kinematic_climatology(
    arrays: Dict[object, np.ndarray],
    calib_ids: Sequence[object],
    leads: Sequence[int],
    dt: float = DEFAULT_DT,
    velocity_window: int = CV_VELOCITY_WINDOW,
    stop_speed: float = 0.2,
    speed_kernel_bw: float = 0.2,
    n_speed_knots: int = 16,
    transition_band: float = 0.15,
    seed: int = 0,
) -> KinematicClimatology:
    leads_t = tuple(sorted({int(l) for l in leads}))
    w = int(velocity_window)
    rng = np.random.default_rng(seed)
    speed_grid: Dict[int, np.ndarray] = {}
    mu_grid: Dict[int, np.ndarray] = {}
    cov_grid: Dict[int, np.ndarray] = {}
    stop_cov: Dict[int, np.ndarray] = {}
    marginal_cov: Dict[int, np.ndarray] = {}
    cv_covs: Dict[int, np.ndarray] = {}
    n_samples: Dict[int, int] = {}

    for lead in leads_t:
        sp_chunks, body_chunks, world_chunks = [], [], []
        for tid in calib_ids:
            obs = arrays[tid]
            t = len(obs)
            if t <= lead + w:
                continue
            idx = np.arange(w, t - lead)
            vel = (obs[idx] - obs[idx - w]) / (w * dt)
            disp = obs[idx + lead] - obs[idx]
            finite = np.isfinite(vel).all(axis=1) & np.isfinite(disp).all(axis=1)
            if not finite.any():
                continue
            vel, disp = vel[finite], disp[finite]
            sp = np.hypot(vel[:, 0], vel[:, 1])
            hd = np.arctan2(vel[:, 1], vel[:, 0])
            cth, sth = np.cos(hd), np.sin(hd)
            body = np.column_stack(
                [cth * disp[:, 0] + sth * disp[:, 1], -sth * disp[:, 0] + cth * disp[:, 1]]
            )
            sp_chunks.append(sp)
            body_chunks.append(body)
            world_chunks.append(disp)

        if not sp_chunks:
            raise ValueError(f"No kinematic samples at lead {lead}")
        speeds = np.concatenate(sp_chunks)
        body = np.concatenate(body_chunks, axis=0)
        world = np.concatenate(world_chunks, axis=0)
        if len(speeds) > MAX_FIT_SAMPLES:
            sel = rng.choice(len(speeds), MAX_FIT_SAMPLES, replace=False)
            speeds, body, world = speeds[sel], body[sel], world[sel]
        if len(speeds) < 100:
            raise ValueError(f"Only {len(speeds)} kinematic samples at lead {lead}")

        moving = speeds >= stop_speed
        stopped_world = world[~moving]
        if len(stopped_world) >= 20:
            scov = (stopped_world.T @ stopped_world) / len(stopped_world)
        else:
            scov = (world.T @ world) / len(world)
        stop_cov[lead] = regularize_cov(scov)

        sm_sp, sm_body = speeds[moving], body[moving]
        if len(sm_sp) < 50:
            sm_sp, sm_body = speeds, body
        lo = float(max(stop_speed, np.quantile(sm_sp, 0.01)))
        hi = float(np.quantile(sm_sp, 0.99))
        if hi <= lo:
            hi = lo + 1e-3
        grid = np.linspace(lo, hi, n_speed_knots)
        mug = np.empty((n_speed_knots, 2))
        covg = np.empty((n_speed_knots, 2, 2))
        for gi, sg in enumerate(grid):
            wts = np.exp(-0.5 * ((sm_sp - sg) / max(speed_kernel_bw, 1e-6)) ** 2)
            if wts.sum() < 1e-6:
                wts = np.ones_like(sm_sp)
            mu, cov = _weighted_moments(sm_body, wts)
            mug[gi] = mu
            covg[gi] = regularize_cov(cov)

        speed_grid[lead] = grid
        mu_grid[lead] = mug
        cov_grid[lead] = covg
        marginal_cov[lead] = regularize_cov(np.cov(body.T))
        cv_covs[lead] = fit_cv_residual_cov(arrays, calib_ids, lead, dt)
        n_samples[lead] = int(len(speeds))

    return KinematicClimatology(
        leads=leads_t, dim=2, dt=dt, velocity_window=w, stop_speed=float(stop_speed),
        speed_kernel_bw=float(speed_kernel_bw), transition_band=float(transition_band),
        speed_grid=speed_grid, mu_grid=mu_grid, cov_grid=cov_grid, stop_cov=stop_cov,
        marginal_cov=marginal_cov, cv_covs=cv_covs, n_samples=n_samples,
        n_calib_tracks=len(list(calib_ids)),
    )
