"""Interacting Multiple Model (IMM) baseline: dwell + CV + left/right turns.

The standard adaptive baseline from the target-tracking literature, ported from
the internal pipeline. Each mode is a Kalman filter with its own dynamics; mode
probabilities evolve via a Markov transition matrix and the mixture is
moment-matched to a single Gaussian per forecast step. 2D only.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..constants import COV_REGULARIZATION
from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from ..forecasters.base import BaseForecaster
from ..scoring.nll import LOG_2PI
from ._kinematics import CalibratedNoise


def _cv_F(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _ct_F(omega: float, dt: float) -> np.ndarray:
    if abs(omega) < 1e-9:
        return _cv_F(dt)
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    return np.array([
        [1, 0, s / omega, -(1 - c) / omega],
        [0, 1, (1 - c) / omega, s / omega],
        [0, 0, c, -s],
        [0, 0, s, c],
    ])


def _white_accel_Q(sigma_a: float, dt: float) -> np.ndarray:
    q_pp = dt**4 / 4.0
    q_pv = dt**3 / 2.0
    q_vv = dt**2
    Q1 = np.array([[q_pp, q_pv], [q_pv, q_vv]]) * sigma_a**2
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return Q


def _stop_F(velocity_decay: float) -> np.ndarray:
    F = np.eye(4)
    F[2, 2] = float(velocity_decay)
    F[3, 3] = float(velocity_decay)
    return F


class IMM(BaseForecaster):
    def __init__(
        self,
        calibration: Optional[CalibratedNoise] = None,
        measurement_noise: Optional[float] = None,
        history: int = 10,
        horizon: int = 15,
        modes: Tuple[str, ...] = ("stop", "cv", "turn_left", "turn_right"),
        turn_rate: float = 0.6,
        sigma_a_cv: float = 0.4,
        sigma_a_ct: float = 0.7,
        sigma_a_stop: float = 0.15,
        stop_velocity_decay: float = 0.15,
        p_stay: float = 0.97,
        name: str = "IMM",
    ):
        if measurement_noise is None:
            measurement_noise = calibration.measurement_noise if calibration else 0.05
        self.measurement_noise = float(measurement_noise)
        self.history = int(history)
        self.horizon = int(horizon)
        self.turn_rate = float(turn_rate)
        self.sigma_a_cv = float(sigma_a_cv)
        self.sigma_a_ct = float(sigma_a_ct)
        self.sigma_a_stop = float(sigma_a_stop)
        self.stop_velocity_decay = float(stop_velocity_decay)
        self.p_stay = float(p_stay)
        self.name = name

        spec = []
        for m in modes:
            if m == "stop":
                spec.append(("stop", 0.0))
            elif m == "cv":
                spec.append(("cv", 0.0))
            elif m == "turn_left":
                spec.append(("turn_left", self.turn_rate))
            elif m == "turn_right":
                spec.append(("turn_right", -self.turn_rate))
            else:
                raise ValueError(f"unknown IMM mode '{m}'")
        self._mode_specs = spec
        n = len(spec)
        off = (1.0 - self.p_stay) / (n - 1)
        self._M = np.full((n, n), off)
        np.fill_diagonal(self._M, self.p_stay)

    def _mode_matrices(self, dt: float):
        Fs, Qs = [], []
        for kind, omega in self._mode_specs:
            if kind == "stop":
                Fs.append(_stop_F(self.stop_velocity_decay))
                Qs.append(_white_accel_Q(self.sigma_a_stop, dt))
            else:
                Fs.append(_ct_F(omega, dt))
                Qs.append(_white_accel_Q(self.sigma_a_cv if kind == "cv" else self.sigma_a_ct, dt))
        return Fs, Qs

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        if traj.D != 2:
            return None
        obs = traj.positions
        dt = traj.dt
        T, L, H = traj.T, self.history, self.horizon
        if T < L + H:
            return None

        sigma_r = self.measurement_noise
        R = np.eye(2) * sigma_r**2
        R_obs = R + np.eye(2) * COV_REGULARIZATION
        Hmat = np.zeros((2, 4))
        Hmat[0, 0] = Hmat[1, 1] = 1.0
        Fs, Qs = self._mode_matrices(dt)
        n_modes = len(self._mode_specs)
        M = self._M

        v0 = (obs[3] - obs[0]) / (3 * dt)
        x0 = np.array([obs[0, 0], obs[0, 1], v0[0], v0[1]])
        P0 = np.diag([sigma_r**2, sigma_r**2, 0.5**2, 0.5**2])
        xs = [x0.copy() for _ in range(n_modes)]
        Ps = [P0.copy() for _ in range(n_modes)]
        mu = np.full(n_modes, 1.0 / n_modes)

        issue_frames, means_out, covs_out = [], [], []
        for t in range(1, T):
            cbar = np.maximum(M.T @ mu, 1e-12)
            mix_w = (M * mu[:, None]) / cbar[None, :]
            xs_mixed, Ps_mixed = [], []
            for j in range(n_modes):
                xm = sum(mix_w[i, j] * xs[i] for i in range(n_modes))
                Pm = sum(
                    mix_w[i, j] * (Ps[i] + np.outer(xs[i] - xm, xs[i] - xm))
                    for i in range(n_modes)
                )
                xs_mixed.append(xm)
                Ps_mixed.append(Pm)
            for j in range(n_modes):
                xs[j] = Fs[j] @ xs_mixed[j]
                Ps[j] = Fs[j] @ Ps_mixed[j] @ Fs[j].T + Qs[j]

            if L <= t <= T - H:
                f_means = np.empty((H, 2))
                f_covs = np.empty((H, 2, 2))
                xs_f = [x.copy() for x in xs]
                Ps_f = [P.copy() for P in Ps]
                mu_f = M.T @ mu
                for h in range(H):
                    xm = sum(mu_f[j] * xs_f[j] for j in range(n_modes))
                    Pm = sum(
                        mu_f[j] * (Ps_f[j] + np.outer(xs_f[j] - xm, xs_f[j] - xm))
                        for j in range(n_modes)
                    )
                    f_means[h] = xm[:2]
                    f_covs[h] = Pm[:2, :2] + R
                    if h < H - 1:
                        for j in range(n_modes):
                            xs_f[j] = Fs[j] @ xs_f[j]
                            Ps_f[j] = Fs[j] @ Ps_f[j] @ Fs[j].T + Qs[j]
                        mu_f = M.T @ mu_f
                issue_frames.append(t)
                means_out.append(f_means)
                covs_out.append(f_covs)

            z = obs[t]
            likelihoods = np.empty(n_modes)
            for j in range(n_modes):
                S = Hmat @ Ps[j] @ Hmat.T + R_obs
                S_inv = np.linalg.inv(S)
                resid = z - Hmat @ xs[j]
                K = Ps[j] @ Hmat.T @ S_inv
                xs[j] = xs[j] + K @ resid
                Ps[j] = (np.eye(4) - K @ Hmat) @ Ps[j]
                _, log_det = np.linalg.slogdet(S)
                likelihoods[j] = -0.5 * float(resid @ S_inv @ resid) - 0.5 * log_det - LOG_2PI

            log_mu = np.log(np.maximum(M.T @ mu, 1e-300)) + likelihoods
            log_mu -= log_mu.max()
            mu = np.exp(log_mu)
            mu /= mu.sum()

        if not issue_frames:
            return None
        return Predictive(
            issue_frames=np.array(issue_frames, dtype=int),
            means=np.array(means_out),
            covs=np.array(covs_out),
            horizon=H,
        )
