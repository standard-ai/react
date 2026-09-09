"""Synthetic regime-change trajectories with ground-truth events.

A toy generator for CI, tutorials, and parity tests. Each track is a sequence of
piecewise-constant-velocity segments separated by regime changes (turns, speed
changes, stops/starts), with Gaussian observation noise. The changepoints are
returned as ground-truth events so REACT can be exercised end-to-end without any
proprietary data.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..constants import DEFAULT_DT
from ..core.trajectory import Trajectory, TrajectorySet
from ..events.base import Event


def _one_track(
    rng: np.random.Generator,
    track_id: int,
    dt: float,
    min_segment: int,
    max_segment: int,
    noise: float,
    speed_range: Tuple[float, float],
) -> Tuple[np.ndarray, List[int]]:
    n_segments = rng.integers(3, 6)
    pos = rng.normal(0, 2.0, size=2)
    heading = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(*speed_range)
    clean: List[np.ndarray] = [pos.copy()]
    changepoints: List[int] = []
    for seg in range(n_segments):
        length = int(rng.integers(min_segment, max_segment))
        vel = speed * np.array([np.cos(heading), np.sin(heading)])
        for _ in range(length):
            pos = pos + vel * dt
            clean.append(pos.copy())
        if seg < n_segments - 1:
            changepoints.append(len(clean) - 1)
            kind = rng.integers(0, 3)
            if kind == 0:  # turn
                heading += rng.uniform(0.6, 1.8) * rng.choice([-1, 1])
            elif kind == 1:  # speed change
                speed = rng.uniform(*speed_range)
            else:  # stop then restart
                speed = 0.0 if speed > 0.1 else rng.uniform(*speed_range)
    clean_arr = np.array(clean)
    noisy = clean_arr + rng.normal(0, noise, size=clean_arr.shape)
    return noisy, changepoints


def synthetic_regime_changes(
    n: int = 200,
    seed: int = 0,
    dt: float = DEFAULT_DT,
    min_segment: int = 25,
    max_segment: int = 60,
    noise: float = 0.05,
    speed_range: Tuple[float, float] = (0.8, 1.6),
) -> Tuple[TrajectorySet, Dict[int, List[Event]]]:
    """Generate ``n`` synthetic tracks and their ground-truth regime-change events.

    Returns:
        (TrajectorySet, events_map) where events_map[id] is the list of true
        Events (with t0 set; effect sizes left at defaults).
    """
    rng = np.random.default_rng(seed)
    tracks: Dict[object, Trajectory] = {}
    events: Dict[int, List[Event]] = {}
    for tid in range(n):
        noisy, cps = _one_track(rng, tid, dt, min_segment, max_segment, noise, speed_range)
        tracks[tid] = Trajectory(noisy, dt=dt, id=tid)
        events[tid] = [Event(id=tid, t0=int(cp), event_type="synthetic") for cp in cps]
    return TrajectorySet(tracks), events
