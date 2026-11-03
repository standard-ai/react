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
conditioned out. A separate isotropic stopped regime handles people standing still.

How the two regimes are weighted is set by ``stop_model``:

* ``"ramp"`` (default): a hand-set linear ramp from ``stop_speed`` to
  ``stop_speed + transition_band``, with each regime fitted on its side of the
  threshold.
* ``"fitted"``: the probability of the moving regime is a smooth function of speed
  estimated from data, jointly with both regimes, by EM (a mixture of experts with
  kernel-smoothed M-steps over the speed grid). A person who looks stopped but
  starts moving within the lead is then accounted for at the rate it happens in the
  calibration data, instead of by where a threshold happens to sit.

Heading is a 2D notion, so this mode requires D == 2; the facade falls back to the
global climatology for other dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from ..constants import DEFAULT_DT
from ..scoring.nll import _LOG_2PI, _gaussian_nll
from .base import (
    CV_VELOCITY_WINDOW,
    MAX_FIT_SAMPLES,
    fit_cv_residual_cov,
    regularize_cov,
)

# shortest track stretch (in target frames) the per-frame views accept
MIN_FRAMES = 15


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
    stop_model: str = "ramp"
    move_weight_grid: Dict[int, np.ndarray] = field(default_factory=dict)
    em_loglik: Dict[int, list] = field(default_factory=dict)

    def move_weight(self, lead: int, speed) -> np.ndarray:
        """Probability of the moving regime at this causal speed."""
        speed = np.asarray(speed, dtype=float)
        if self.stop_model == "fitted":
            return np.interp(speed, self.speed_grid[lead], self.move_weight_grid[lead])
        return np.clip((speed - self.stop_speed) / max(self.transition_band, 1e-9), 0.0, 1.0)

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
        nll_move = _gaussian_nll(disp_body - mu, cov)
        nll_stop = _gaussian_nll(disp, self.stop_cov[lead])
        lam = self.move_weight(lead, speed)
        eps = 1e-12
        logp_move = np.log(lam + eps) - nll_move
        logp_stop = np.log(1.0 - lam + eps) - nll_stop
        peak = np.maximum(logp_move, logp_stop)
        logp = peak + np.log(np.exp(logp_move - peak) + np.exp(logp_stop - peak))
        nll = -logp
        ok = np.isfinite(nll)
        return frames[ok].astype(int), nll[ok]

    def conditional_gaussian(
        self, lead: int, speed: float, heading: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu_b, cov_b = self._predict_moving(lead, np.array([float(speed)]))
        mu_b, cov_b = mu_b[0], cov_b[0]
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        return rot @ mu_b, rot @ cov_b @ rot.T

    # -- per-frame views used by the REACT tests (reactmetric.information) -----------------

    def _anchors(self, obs: np.ndarray, lead: int, target: np.ndarray):
        """Causal velocity (from the last velocity_window frames before issue) at each target."""
        vw = int(self.velocity_window)
        anchors = target - lead
        vel = (obs[anchors] - obs[anchors - vw]) / (vw * self.dt)
        return anchors, vel

    def gaussian_frames(self, traj, lead: int, min_frames: int = MIN_FRAMES):
        """Target frames with the climatology's moment-matched Gaussian (moving + stopped).

        Returns (target, y, mu, S) over every frame with a full velocity window before its issue
        frame, or None when the track has fewer than min_frames such frames."""
        obs = np.asarray(traj.positions, float)
        T = len(obs)
        vw = int(self.velocity_window)
        target = np.arange(lead + vw, T)
        if len(target) < min_frames:
            return None
        anchors, vel = self._anchors(obs, lead, target)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        lam = self.move_weight(lead, speed)
        stop = np.asarray(self.stop_cov[lead], float)
        mu = np.empty((len(target), 2))
        S = np.empty((len(target), 2, 2))
        for i in range(len(target)):
            mw, cw = self.conditional_gaussian(lead, float(speed[i]), float(heading[i]))
            md = lam[i] * mw
            S[i] = lam[i] * (cw + np.outer(mw, mw)) + (1.0 - lam[i]) * stop - np.outer(md, md)
            mu[i] = obs[anchors[i]] + md
        return target, obs[target], mu, S + 1e-8 * np.eye(2)

    def mixture_frames(self, traj, lead: int, min_frames: int = MIN_FRAMES):
        """Target frames with the climatology as its real two-regime mixture (moving, stopped).

        Returns (target, y, weights (n, 2), means (n, 2, 2) absolute, covs (n, 2, 2, 2)), the same
        density as nll_trace (gaussian_frames is its moment-matched Gaussian), or None."""
        obs = np.asarray(traj.positions, float)
        T = len(obs)
        vw = int(self.velocity_window)
        target = np.arange(lead + vw, T)
        if len(target) < min_frames:
            return None
        anchors, vel = self._anchors(obs, lead, target)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        mu_b, cov_b = self._predict_moving(lead, speed)
        cs, sn = np.cos(heading), np.sin(heading)
        R = np.stack([np.stack([cs, -sn], -1), np.stack([sn, cs], -1)], -2)
        mu_w = np.einsum("nij,nj->ni", R, mu_b)
        cov_w = np.einsum("nij,njk,nlk->nil", R, cov_b, R)
        lam = np.asarray(self.move_weight(lead, speed), float)
        weights = np.column_stack([lam + 1e-12, 1.0 - lam + 1e-12])
        weights /= weights.sum(1, keepdims=True)
        means = np.stack([obs[anchors] + mu_w, obs[anchors]], 1)
        stop = np.broadcast_to(np.asarray(self.stop_cov[lead], float), cov_w.shape)
        covs = np.stack([cov_w, stop], 1) + 1e-8 * np.eye(2)
        return target, obs[target], weights, means, covs

    def frame_geometry(self, traj, lead: int, target: np.ndarray):
        """Causal speed, heading and anchor (issue-frame) position at each target frame."""
        obs = np.asarray(traj.positions, float)
        anchors, vel = self._anchors(obs, lead, np.asarray(target))
        return np.hypot(vel[:, 0], vel[:, 1]), np.arctan2(vel[:, 1], vel[:, 0]), obs[anchors]

    def body_mixture(self, lead: int):
        """The climatology in the mover's body frame as a function of speed, for OverlapTable.

        Returns body(v) -> weights (V, 2), means (V, 2, 2), covs (V, 2, 2, 2). The stopped
        regime's spread is fixed in world axes; the table uses its isotropic average."""
        stop = np.asarray(self.stop_cov[lead], float)
        stop_iso = 0.5 * np.trace(stop) * np.eye(2)

        def body(v):
            mu_b, cov_b = self._predict_moving(lead, v)
            lam = np.asarray(self.move_weight(lead, v), float)
            w = np.column_stack([lam + 1e-12, 1.0 - lam + 1e-12])
            w /= w.sum(1, keepdims=True)
            means = np.stack([mu_b, np.zeros_like(mu_b)], 1)
            covs = np.stack([cov_b, np.broadcast_to(stop_iso, cov_b.shape)], 1) + 1e-8 * np.eye(2)
            return w, means, covs

        return body

    def table_axes(self, lead: int, core: float = 8.0, step: float = 1.0 / 6.0,
                   growth: float = 1.06, reach: float = 10.0):
        """Speed grid and body-frame axis of an OverlapTable for one lead.

        Speeds: the climatology's knots and their midpoints. Positions: uniform within +-core
        stopped-regime sds at `step` sds, then growing geometrically by `growth` per node out to
        `reach` moving-regime sds past the moving mean. Returns (v, core, h, growth, extent)."""
        knots = np.asarray(self.speed_grid[lead], float)
        v = np.sort(np.concatenate([knots, 0.5 * (knots[1:] + knots[:-1])]))
        sd_stop = float(np.sqrt(0.5 * np.trace(self.stop_cov[lead])))
        mu_b, cov_b = self._predict_moving(lead, v)
        far = np.abs(mu_b).max(1) + reach * np.sqrt(np.linalg.eigvalsh(cov_b)[:, 1])
        c = core * sd_stop
        return v, c, step * sd_stop, growth, max(float(far.max()), 2.0 * c)

    def metadata(self) -> Dict:
        out = {
            "mode": "kinematic",
            "dim": self.dim,
            "velocity_window": self.velocity_window,
            "stop_speed": self.stop_speed,
            "speed_kernel_bw": self.speed_kernel_bw,
            "transition_band": self.transition_band,
            "stop_model": self.stop_model,
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
                    1.0 + _LOG_2PI + 0.5 * np.log(np.linalg.det(cov))
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
    stop_model: str = "ramp",
    em_iters: int = 60,
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
    move_weight_grid: Dict[int, np.ndarray] = {}
    em_loglik: Dict[int, list] = {}
    if stop_model not in ("ramp", "fitted"):
        raise ValueError(f"stop_model must be 'ramp' or 'fitted'; got {stop_model!r}")

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

        if stop_model == "fitted":
            grid, mug, covg, scov, wgrid, trace = _fit_mixture_em(
                speeds, body, world, stop_speed, transition_band, speed_kernel_bw,
                n_speed_knots, em_iters,
            )
            speed_grid[lead], mu_grid[lead], cov_grid[lead] = grid, mug, covg
            stop_cov[lead], move_weight_grid[lead], em_loglik[lead] = scov, wgrid, trace
            marginal_cov[lead] = regularize_cov(np.cov(body.T))
            cv_covs[lead] = fit_cv_residual_cov(arrays, calib_ids, lead, dt)
            n_samples[lead] = int(len(speeds))
            continue

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
        stop_model=stop_model, move_weight_grid=move_weight_grid, em_loglik=em_loglik,
        n_calib_tracks=len(list(calib_ids)),
    )


def _fit_mixture_em(speeds, body, world, stop_speed, transition_band, bw, n_knots, iters):
    """Best of several EM starts (mixtures have local optima; keep the highest likelihood)."""
    starts = [(stop_speed, transition_band), (0.1, 0.1), (0.4, 0.3), (0.05, 0.6)]
    fits = [
        _fit_mixture_em_once(speeds, body, world, s0, b0, bw, n_knots, iters) for s0, b0 in starts
    ]
    return max(fits, key=lambda f: f[-1][-1])


def _fit_mixture_em_once(speeds, body, world, stop_speed, transition_band, bw, n_knots, iters):
    """Stopped/moving mixture of experts over speed, by EM.

    p(d | s) = (1 - pi(s)) N(d_world; 0, S_stop) + pi(s) N(d_body; mu(s), S(s)).
    E-step: responsibilities. M-step: S_stop from stopped responsibilities; mu(s),
    S(s) and pi(s) as kernel-smoothed (bandwidth ``bw``) weighted moments on a speed
    grid spanning all speeds, interpolated between knots. Initialised from the ramp.
    """
    n_knots = max(int(n_knots), 24)
    grid = np.linspace(0.0, float(np.quantile(speeds, 0.995)), n_knots)
    K = np.exp(-0.5 * ((speeds[None, :] - grid[:, None]) / max(bw, 1e-6)) ** 2)  # (G, N)
    K_sum = np.maximum(K.sum(axis=1), 1e-12)
    ramp = np.clip((speeds - stop_speed) / max(transition_band, 1e-9), 0.0, 1.0)
    r_move = np.clip(ramp, 0.02, 0.98)
    mug = np.zeros((n_knots, 2))
    covg = np.zeros((n_knots, 2, 2))
    trace = []

    def interp(values):
        return np.stack([np.interp(speeds, grid, values[:, j]) for j in range(values.shape[1])], 1)

    for _ in range(int(iters)):
        r_stop = 1.0 - r_move
        scov = regularize_cov((r_stop[:, None] * world).T @ world / max(r_stop.sum(), 1e-12))
        wgrid = np.clip((K * r_move[None, :]).sum(axis=1) / K_sum, 1e-4, 1 - 1e-4)
        for g in range(n_knots):
            wts = K[g] * r_move
            if wts.sum() < 1e-8:
                wts = K[g]
            mu, cov = _weighted_moments(body, wts)
            mug[g] = mu
            covg[g] = regularize_cov(cov)
        pi = np.interp(speeds, grid, wgrid)
        mu_i = interp(mug)
        cov_i = interp(covg.reshape(n_knots, 4)).reshape(-1, 2, 2)
        lp_move = np.log(pi) - _gaussian_nll(body - mu_i, cov_i)
        lp_stop = np.log(1.0 - pi) - _gaussian_nll(world, scov)
        peak = np.maximum(lp_move, lp_stop)
        ll = peak + np.log(np.exp(lp_move - peak) + np.exp(lp_stop - peak))
        trace.append(float(ll.mean()))
        r_move = np.exp(lp_move - ll)
        if len(trace) > 5 and abs(trace[-1] - trace[-2]) < 1e-6:
            break
    return grid, mug.copy(), covg.copy(), scov, wgrid, trace
