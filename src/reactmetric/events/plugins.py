"""Adapter for the `ruptures` changepoint-detection library (extras=[cpd]).

Lets users plug an external, well-tested CPD stack into REACT instead of (or
alongside) the built-in kinematic detector. The detected breakpoints are wrapped
as generic Events; classification/effect-sizes are left at defaults since the
external cost models are domain-agnostic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.trajectory import Trajectory, TrajectorySet
from .base import Event


class RupturesDetector:
    """Wrap ruptures (Pelt/Binseg/...) on kinematic features as an EventSource."""

    def __init__(
        self,
        model: str = "rbf",
        penalty: float = 10.0,
        algo: str = "pelt",
        velocity_window: int = 5,
        min_size: int = 8,
    ):
        try:
            import ruptures  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "RupturesDetector requires the 'ruptures' package: pip install 'reactmetric[cpd]'"
            ) from e
        self.model = model
        self.penalty = float(penalty)
        self.algo = algo
        self.velocity_window = int(velocity_window)
        self.min_size = int(min_size)

    def _features(self, traj: Trajectory) -> Optional[np.ndarray]:
        obs = traj.positions
        w = self.velocity_window
        if traj.T <= w + 1:
            return None
        vel = (obs[w:] - obs[:-w]) / (w * traj.dt)
        speed = np.hypot(vel[:, 0], vel[:, 1]) if traj.D >= 2 else np.abs(vel[:, 0])
        if traj.D >= 2:
            heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
            turn = np.gradient(heading) / traj.dt
        else:
            turn = np.zeros_like(speed)
        feats = np.column_stack([speed, turn])
        std = feats.std(axis=0)
        std[std < 1e-6] = 1.0
        return feats / std

    def detect(self, traj: Trajectory) -> List[Event]:
        import ruptures as rpt

        feats = self._features(traj)
        if feats is None:
            return []
        algo_cls = {"pelt": rpt.Pelt, "binseg": rpt.Binseg, "window": rpt.Window}.get(
            self.algo, rpt.Pelt
        )
        algo = algo_cls(model=self.model, min_size=self.min_size).fit(feats)
        bkps = algo.predict(pen=self.penalty)
        cps = [b for b in bkps if 0 < b < len(feats)]
        return [
            Event(id=traj.id, t0=int(cp + self.velocity_window), event_type="changepoint")
            for cp in cps
        ]

    def detect_set(self, ts: TrajectorySet) -> Dict[object, List[Event]]:
        return {tid: self.detect(traj) for tid, traj in ts.items()}
