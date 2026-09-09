"""Per-event REACT measurement on a target-frame information trace.

The information score s(f) = nll_ref(f) - nll_model(f) is fed to the detector as
trace = -s with tau_onset = 0 and tau_recovery = -epsilon, so:

- onset: s < 0 sustained (the forecast fell below the reference)
- recovery: s >= epsilon sustained (it beat the reference by the task margin)

The window after t0 is censored at ``max_react_frames`` and truncated at the next
changepoint on the same track (a model still elevated when the regime changes
again never recovered from THIS event -> flagged ``superseded``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import ReactConfig
from ..events.base import Event
from .result import EventMeasurement


def first_sustained_run(mask: np.ndarray, k: int, start: int, stop: int) -> Optional[int]:
    """First index i in [start, stop) with mask[i:i+k] all True; None otherwise."""
    n = len(mask)
    stop = min(stop, n - k + 1)
    for i in range(start, max(stop, start)):
        if mask[i : i + k].all():
            return i
    return None


def measure_event_on_trace(
    frames: np.ndarray,
    trace: np.ndarray,
    event: Event,
    model_name: str,
    tau_onset: float,
    tau_recovery: float,
    config: ReactConfig,
    next_t0: Optional[int] = None,
    force_onset_at_start: bool = False,
) -> Optional[EventMeasurement]:
    """Measure one forecaster's REACT on one event.

    ``trace`` is indexed by ``frames`` (the issue or target frame, depending on
    the caller); for info-REACT it is -s. Returns None when coverage is
    inadequate.
    """
    if len(frames) == 0:
        return None
    t0 = int(event.t0)
    eval_start_frame = t0 + int(config.first_post_event_issue_offset)
    first, last = int(frames[0]), int(frames[-1])
    if eval_start_frame < first or last - eval_start_frame < config.min_post_coverage:
        return None

    i0 = int(np.searchsorted(frames, eval_start_frame))
    if i0 >= len(frames) or frames[i0] != eval_start_frame:
        return None

    window_end_frame = min(t0 + config.max_react_frames, last)
    truncated_by_next = next_t0 is not None and next_t0 - 1 < window_end_frame
    if truncated_by_next:
        window_end_frame = next_t0 - 1
        if window_end_frame - t0 < config.min_post_coverage:
            return None
    i_end = int(np.searchsorted(frames, window_end_frame, side="right"))

    above = trace >= tau_onset
    below = trace < tau_recovery
    k_on = config.onset_consecutive
    k_off = config.recovery_consecutive

    pre_start_frame = max(t0 - config.pre_onset_check_frames, first)
    i_pre = int(np.searchsorted(frames, pre_start_frame))
    pre_elevated = first_sustained_run(above, k_on, i_pre, i0) is not None

    if force_onset_at_start:
        onset_idx: Optional[int] = i0
    else:
        onset_idx = first_sustained_run(above, k_on, i0, i_end)
    coverage = int(frames[min(i_end, len(frames)) - 1] - t0)

    common = dict(
        model_name=model_name,
        id=event.id,
        t0=t0,
        event_type=event.event_type,
        detection_floor_frames=float(event.detection_floor_frames),
        pre_elevated=pre_elevated,
        coverage_frames=coverage,
    )

    if onset_idx is None:
        return EventMeasurement(anticipated=True, react_frames=0.0, censored=False, **common)

    recovery_idx = first_sustained_run(below, k_off, onset_idx, i_end)
    if recovery_idx is None:
        return EventMeasurement(
            anticipated=False,
            react_frames=float(coverage),
            censored=True,
            superseded=truncated_by_next,
            **common,
        )

    react = float(frames[recovery_idx] - t0)
    return EventMeasurement(anticipated=False, react_frames=react, censored=False, **common)
