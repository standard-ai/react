"""Constant-velocity (persistence) baseline forecaster.

The simplest honest adaptive baseline: estimate velocity by OLS over a short
history window and extrapolate. Its predictive covariance grows with lead from
two sources -- velocity-estimate uncertainty (propagated through the
extrapolation) and anchor observation noise -- so it is reasonably calibrated out
of the box. Works in any dimension.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from ..forecasters.base import BaseForecaster
from ._kinematics import ols_velocity, slope_variance_factor


class ConstantVelocity(BaseForecaster):
    def __init__(
        self,
        history: int = 10,
        horizon: int = 15,
        measurement_noise: float = 0.05,
        name: str = "ConstantVelocity",
    ):
        self.history = int(history)
        self.horizon = int(horizon)
        self.measurement_noise = float(measurement_noise)
        self.name = name

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        obs = traj.positions
        dt = traj.dt
        t, dim = obs.shape
        L, H = self.history, self.horizon
        if t < L + H:
            return None
        issue = np.arange(L, t - H + 1)
        n = len(issue)
        means = np.empty((n, H, dim))
        covs = np.zeros((n, H, dim, dim))
        sigma_r2 = self.measurement_noise**2
        slope_factor = slope_variance_factor(L)  # per dt^2
        idx = np.arange(dim)
        for i, ti in enumerate(issue):
            hist = obs[ti - L : ti]
            vel = ols_velocity(hist, dt)
            anchor = obs[ti - 1]
            for h in range(H):
                lead_t = (h + 1) * dt
                means[i, h] = anchor + vel * lead_t
                # Var(velocity per axis) = sigma_r^2 / (dt^2 * Sxx); extrapolated.
                vel_var = sigma_r2 * slope_factor / (dt**2)
                var_h = sigma_r2 + (lead_t**2) * vel_var
                covs[i, h, idx, idx] = var_h
        return Predictive(issue_frames=issue, means=means, covs=covs, horizon=H)
