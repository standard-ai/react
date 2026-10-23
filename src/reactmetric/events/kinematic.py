"""Kinematic features and the event taxonomy shared by REACT's changepoint detectors.

Speed and turn rate from a velocity window, and the classification of a changepoint into the
taxonomy (stop/start, speed_change, turn_onset/exit/change, sharp_turn, both) from pre/post
effect sizes. The segmentation itself is BayesianChangepoints (events/bayes.py), the paper's
detector; RupturesDetector and from_times are alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..core.trajectory import Trajectory, TrajectorySet
from .base import Event


@dataclass
class SegmentationConfig:
    """Feature window, minimum segment length and classification thresholds."""

    velocity_window: int = 10
    min_size: int = 50              # minimum frames between changepoints
    stop_speed: float = 0.2         # m/s; below this is "stopped"
    speed_change_z: float = 2.5     # |dspeed| / sigma to call a speed change
    turn_rate_z: float = 2.0        # |dturn| / sigma to call a turn
    sharp_turn_rate: float = 1.25   # rad/s; abs turn rate for 'sharp_turn'


class KinematicFeatures:
    """Base for kinematic detectors: features from positions, taxonomy for changepoints.

    Subclasses implement detect(traj) -> List[Event]."""

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()

    def _features(self, obs: np.ndarray, dt: float):
        w = self.config.velocity_window
        t = len(obs)
        if t <= w + 1:
            return None
        vel = (obs[w:] - obs[:-w]) / (w * dt)
        speed = np.hypot(vel[:, 0], vel[:, 1]) if obs.shape[1] >= 2 else np.abs(vel[:, 0])
        if obs.shape[1] >= 2:
            heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
            turn_rate = np.gradient(heading) / dt
        else:
            turn_rate = np.zeros_like(speed)
        return speed, turn_rate, w

    def _classify(self, speed, turn_rate, cp, lo, hi, sp_sigma, tr_sigma, dt, tid) -> Event:
        cfg = self.config
        pre_sp, post_sp = speed[lo:cp], speed[cp:hi]
        pre_tr, post_tr = turn_rate[lo:cp], turn_rate[cp:hi]
        d_speed = float(np.mean(post_sp) - np.mean(pre_sp))
        d_turn = float(np.mean(post_tr) - np.mean(pre_tr))
        speed_z = abs(d_speed) / sp_sigma
        turn_z = abs(d_turn) / tr_sigma

        is_speed = speed_z >= cfg.speed_change_z
        is_turn = turn_z >= cfg.turn_rate_z

        pre_mean_sp, post_mean_sp = float(np.mean(pre_sp)), float(np.mean(post_sp))
        if is_speed and post_mean_sp < cfg.stop_speed <= pre_mean_sp:
            etype = "stop"
        elif is_speed and pre_mean_sp < cfg.stop_speed <= post_mean_sp:
            etype = "start"
        elif is_turn and is_speed:
            etype = "both"
        elif is_turn:
            if abs(float(np.mean(post_tr))) >= cfg.sharp_turn_rate:
                etype = "sharp_turn"
            elif abs(float(np.mean(pre_tr))) < cfg.turn_rate_z * tr_sigma:
                etype = "turn_onset"
            elif abs(float(np.mean(post_tr))) < cfg.turn_rate_z * tr_sigma:
                etype = "turn_exit"
            else:
                etype = "turn_change"
        elif is_speed:
            etype = "speed_change"
        else:
            etype = "speed_change" if speed_z >= turn_z else "turn_change"

        if is_speed and is_turn:
            series = "both"
        elif is_turn:
            series = "heading"
        else:
            series = "speed"

        return Event(
            id=tid, t0=cp, event_type=etype, series=series,
            speed_level_z=speed_z, heading_slope_z=turn_z,
            d_speed_level=d_speed, d_turn_rate=d_turn,
        )

    def detect(self, traj: Trajectory) -> List[Event]:
        raise NotImplementedError

    def detect_set(self, ts: TrajectorySet) -> Dict[object, List[Event]]:
        return {tid: self.detect(traj) for tid, traj in ts.items()}
