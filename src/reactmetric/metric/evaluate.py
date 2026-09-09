"""The REACT evaluation driver: forecaster + reference + events -> ReactResult.

The information score at issue frame t is

    s(t) = nll_ref(target) - nll_model(t -> target),   target = t + lead - 1

so REACT's x-axis is "frames since the maneuver was observed" while the reference
is still scored at the requested horizon. Onset is s < 0 sustained; recovery is
s >= epsilon sustained. The forecast is scored AS DELIVERED -- no recalibration.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..config import ReactConfig
from ..core.trajectory import Trajectory, TrajectorySet
from ..events.base import Event, EventSource, events_by_track
from ..forecasters.adapters import as_forecaster
from .measure import measure_event_on_trace
from .result import ReactResult

EventsLike = Union[Dict[object, List[Event]], List[Event], EventSource]


def normalize_events(events: EventsLike, data: TrajectorySet) -> Dict[object, List[Event]]:
    """Coerce events input to a mapping id -> list[Event]."""
    if isinstance(events, dict):
        return events
    if isinstance(events, EventSource):
        return events.detect_set(data)
    if isinstance(events, (list, tuple)):
        return events_by_track(list(events))
    raise TypeError(f"cannot interpret {type(events)!r} as events")


def _model_issue_trace(
    pred, obs: np.ndarray, lead: int, scoring: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Issue-frame model NLL for the lead-step target (column lead-1)."""
    if scoring == "nll":
        nll = pred.nll_matrix(obs)
    else:
        from ..scoring.crps import predictive_score_matrix

        nll = predictive_score_matrix(pred, obs, scoring=scoring)
    h = int(lead)
    if not 1 <= h <= nll.shape[1]:
        raise ValueError(f"lead {h} outside horizon 1..{nll.shape[1]}")
    vals = nll[:, h - 1]
    ok = ~np.isnan(vals)
    return pred.issue_frames[ok], vals[ok]


def _reference_trace(ref, traj: Trajectory, lead: int, reference_kind: str, scoring: str):
    if reference_kind == "cv":
        # Persistence reference is always scored by its log-likelihood.
        return ref.cv_nll_trace(traj, lead)
    if reference_kind != "climatology":
        raise ValueError(f"unknown reference_kind '{reference_kind}'")
    if scoring == "nll" or not hasattr(ref, "score_trace"):
        return ref.nll_trace(traj, lead)
    return ref.score_trace(traj, lead, scoring)


def info_trace(
    forecaster,
    traj: Trajectory,
    ref,
    lead: int,
    reference_kind: str = "climatology",
    scoring: str = "nll",
) -> Tuple[np.ndarray, np.ndarray]:
    """Issue-frame information score s = nll_ref - nll_model at the lead."""
    pred = forecaster.predict(traj)
    if pred is None or len(pred.issue_frames) == 0:
        return np.array([], dtype=int), np.array([])
    m_frames, m_trace = _model_issue_trace(pred, traj.positions, lead, scoring)
    if len(m_frames) == 0:
        return np.array([], dtype=int), np.array([])
    r_frames, r_trace = _reference_trace(ref, traj, lead, reference_kind, scoring)
    if len(r_frames) == 0:
        return np.array([], dtype=int), np.array([])
    target_frames = m_frames + int(lead) - 1
    common, mi, ri = np.intersect1d(target_frames, r_frames, return_indices=True)
    return m_frames[mi].astype(int), r_trace[ri] - m_trace[mi]


def steady_state_informativeness(
    forecaster,
    data: TrajectorySet,
    events_map: Dict[object, List[Event]],
    config: ReactConfig,
    ref,
    lead: int,
    reference_kind: str = "climatology",
    scoring: str = "nll",
) -> Tuple[float, float, int]:
    """Mean s and P(s > 0) on event-masked calibration-split frames."""
    calib, _ = data.split(fraction=config.calib_fraction, seed=config.calib_seed)
    ids = calib.ids[: config.calib_max_tracks]
    values = []
    for tid in ids:
        traj = data[tid]
        frames, s = info_trace(forecaster, traj, ref, lead, reference_kind, scoring)
        if len(s) == 0:
            continue
        mask = np.ones(len(frames), dtype=bool)
        for ev in events_map.get(tid, []):
            lo = ev.t0 - config.event_mask_pre_frames
            hi = ev.t0 + config.event_mask_post_frames
            mask &= ~((frames >= lo) & (frames <= hi))
        values.append(s[mask])
    if not values:
        return float("nan"), float("nan"), 0
    all_vals = np.concatenate(values)
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(all_vals)), float(np.mean(all_vals > 0)), int(len(all_vals))


def evaluate(
    forecaster,
    data: TrajectorySet,
    reference,
    events: EventsLike,
    lead: Optional[int] = None,
    epsilon: float = 0.0,
    reference_kind: str = "climatology",
    scoring: str = "nll",
    config: Optional[ReactConfig] = None,
    name: Optional[str] = None,
    verbose: bool = False,
) -> ReactResult:
    """Measure information-referenced REACT for one forecaster over all events.

    ``forecaster`` may be a Forecaster, a dict[id -> Predictive], a callable, or
    an event-conditioned oracle (exposing ``info_trace(obs, ref, lead, t0)`` and
    ``is_event_conditioned = True``).
    """
    config = config or ReactConfig()
    lead = int(lead if lead is not None else (config.lead or config.forecast_horizon))
    events_map = normalize_events(events, data)

    event_conditioned = getattr(forecaster, "is_event_conditioned", False)
    model_name = name or getattr(forecaster, "name", "model")

    if not event_conditioned:
        forecaster = as_forecaster(forecaster, name=model_name)
        model_name = forecaster.name
        mean_s, frac_pos, n_ss = steady_state_informativeness(
            forecaster, data, events_map, config, reference, lead, reference_kind, scoring
        )
    else:
        mean_s, frac_pos, n_ss = float("nan"), float("nan"), 0

    result = ReactResult(
        model_name=model_name,
        reference=reference_kind,
        epsilon=float(epsilon),
        lead=int(lead),
        steady_state_mean_info=mean_s,
        steady_state_frac_informative=frac_pos,
        n_steady_state_samples=n_ss,
    )

    for tid in data.ids:
        track_events = events_map.get(tid, [])
        if not track_events:
            continue
        traj = data[tid]
        track_events = sorted(track_events, key=lambda e: e.t0)
        if not event_conditioned:
            frames, s = info_trace(forecaster, traj, reference, lead, reference_kind, scoring)
            for k, ev in enumerate(track_events):
                next_t0 = track_events[k + 1].t0 if k + 1 < len(track_events) else None
                meas = measure_event_on_trace(
                    frames, -s, ev, model_name, 0.0, -float(epsilon), config, next_t0=next_t0
                )
                if meas is not None:
                    result.measurements.append(meas)
        else:
            for k, ev in enumerate(track_events):
                next_t0 = track_events[k + 1].t0 if k + 1 < len(track_events) else None
                frames, s = forecaster.info_trace(traj.positions, reference, lead, ev.t0)
                if len(frames) == 0:
                    continue
                meas = measure_event_on_trace(
                    frames, -s, ev, model_name, 0.0, -float(epsilon), config,
                    next_t0=next_t0, force_onset_at_start=True,
                )
                if meas is not None:
                    result.measurements.append(meas)

    if verbose:
        from ..constants import react_at_lead

        print(
            f"{model_name}: {react_at_lead(lead)} measured {result.n_measured} events | "
            f"anticipation {100 * result.anticipation_rate:.1f}% | "
            f"failure {100 * result.failure_rate:.1f}%"
        )
    return result
