"""Oracle A (hindsight): an acausal artifact detector.

Predictive mean = the smoothed REALIZED trajectory; covariance = measurement
noise. Its NLL is flat wherever the track is physically coherent, so an event at
which even the hindsight oracle is surprised reflects incoherent observations
(tracking glitches, identity switches) rather than a behavior change. Such events
should be excluded before scoring forecasters.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from ..forecasters.base import BaseForecaster


class Hindsight(BaseForecaster):
    def __init__(
        self,
        measurement_noise: float = 0.05,
        history: int = 10,
        horizon: int = 15,
        smooth_window: int = 9,
        smooth_polyorder: int = 2,
        name: str = "oracle_hindsight",
    ):
        self.measurement_noise = float(measurement_noise)
        self.history = int(history)
        self.horizon = int(horizon)
        self.smooth_window = int(smooth_window)
        self.smooth_polyorder = int(smooth_polyorder)
        self.name = name

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        from scipy.signal import savgol_filter

        obs = traj.positions
        T, L, H, D = traj.T, self.history, self.horizon, traj.D
        if T < L + H:
            return None
        window = min(self.smooth_window, T if T % 2 == 1 else T - 1)
        if window <= self.smooth_polyorder:
            return None
        smoothed = np.column_stack(
            [savgol_filter(obs[:, d], window, self.smooth_polyorder, mode="interp")
             for d in range(D)]
        )
        sigma_r = self.measurement_noise
        R = np.eye(D) * sigma_r**2
        issue = np.arange(L, T - H + 1)
        n = len(issue)
        means = np.empty((n, H, D))
        covs = np.tile(R, (n, H, 1, 1))
        for i, t in enumerate(issue):
            means[i] = smoothed[t : t + H]
        return Predictive(issue_frames=issue.astype(int), means=means, covs=covs, horizon=H)


def filter_artifacts(
    events,
    data,
    ref,
    oracle: Optional[Hindsight] = None,
    lead: Optional[int] = None,
    config=None,
) -> Tuple[Dict[object, List], Dict[object, List]]:
    """Split events into (genuine, artifacts) using the hindsight oracle.

    An event is an artifact if the acausal hindsight oracle is itself elevated
    (sustained onset) after t0 -- i.e. even a smoother that sees the future
    cannot beat the climatology, so the raw observations are incoherent.
    """
    from ..config import ReactConfig
    from ..metric.evaluate import evaluate, normalize_events

    config = config or ReactConfig()
    lead = int(lead if lead is not None else (config.lead or config.forecast_horizon))
    oracle = oracle or Hindsight(horizon=config.forecast_horizon)
    events_map = normalize_events(events, data)

    res = evaluate(oracle, data, ref, events_map, lead=lead, config=config)
    artifact_keys = {(m.id, int(m.t0)) for m in res.measurements if not m.anticipated}

    genuine: Dict[object, List] = {}
    artifacts: Dict[object, List] = {}
    for tid, evs in events_map.items():
        for ev in evs:
            bucket = artifacts if (tid, int(ev.t0)) in artifact_keys else genuine
            bucket.setdefault(tid, []).append(ev)
    return genuine, artifacts
