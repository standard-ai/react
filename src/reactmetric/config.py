"""ReactConfig: knobs for the REACT measurement pipeline.

A domain-agnostic port of the internal ReactV2Config. The self-referenced
(per-model NLL band) legacy metric is intentionally absent: the public library is
information-referenced only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ReactConfig:
    """Configuration for the information-referenced REACT pipeline."""

    forecast_horizon: int = 15

    # REACT@h: NLL traces use the h-step-ahead forecast of each target frame
    # (None = average over all leads 1..H). Headline lead + sweep grid.
    lead: Optional[int] = 10
    lead_sweep: Tuple[int, ...] = (1, 5, 10, 15)

    # Sustained-run requirements on the target-frame trace.
    onset_consecutive: int = 3
    recovery_consecutive: int = 3

    # Measurement windows (frames).
    max_react_frames: int = 150         # censoring horizon after t0
    min_post_coverage: int = 15         # trace coverage required after t0
    pre_onset_check_frames: int = 15    # pre-t0 window for the pre-elevation flag
    first_post_event_issue_offset: int = 1

    # Steady-state masking around events (for steady-state informativeness).
    event_mask_pre_frames: int = 20
    event_mask_post_frames: int = 80

    # Calibration-split selection (for steady-state stats and reference fitting).
    calib_fraction: float = 0.15
    calib_seed: int = 0
    calib_max_tracks: int = 500
    min_steady_state_samples: int = 400

    # Reference / information score.
    epsilon: float = 0.0                # recovery margin (nats)
    reference_mode: str = "kinematic"   # 'kinematic' | 'global' | 'position'
    clim_k: int = 4
    epsilon_sweep: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    distortion_targets: Tuple[float, ...] = (0.9, 0.3)

    # Forecast-accuracy summary (ADE/FDE/NLL), independent of REACT.
    performance_max_tracks: int = 500
    performance_seed: int = 1

    # Bootstrap.
    n_boot: int = 1000
