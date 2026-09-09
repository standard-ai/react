"""Sweeps, rate-distortion task anchors, and an accuracy summary.

These are the "vary one knob" analyses that turn a single REACT number into a
picture: REACT vs forecast lead, REACT vs recovery margin epsilon, the
information thresholds for concrete localization tasks, and the classic ADE/FDE
accuracy summary (so REACT can be shown alongside the metric it complements).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import ReactConfig
from .core.trajectory import TrajectorySet
from .forecasters.adapters import as_forecaster
from .metric.evaluate import evaluate, normalize_events

ForecastersLike = Union[Dict[str, object], List, object]


def _named_forecasters(forecasters: ForecastersLike) -> List[Tuple[str, object]]:
    if isinstance(forecasters, dict):
        return [(name, as_forecaster(obj, name=name)) for name, obj in forecasters.items()]
    if isinstance(forecasters, (list, tuple)):
        out = []
        for obj in forecasters:
            f = as_forecaster(obj, name=getattr(obj, "name", "model"))
            out.append((f.name, f))
        return out
    f = as_forecaster(forecasters, name=getattr(forecasters, "name", "model"))
    return [(f.name, f)]


def lead_sweep(
    forecasters: ForecastersLike,
    data: TrajectorySet,
    ref,
    events,
    leads: Sequence[int] = (1, 5, 10, 15),
    epsilon: float = 0.0,
    config: Optional[ReactConfig] = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """REACT@h grid: anticipation/failure/KM-median per model per lead."""
    config = config or ReactConfig()
    events_map = normalize_events(events, data)
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name, f in _named_forecasters(forecasters):
        out[name] = {}
        for lead in leads:
            res = evaluate(f, data, ref, events_map, lead=lead, epsilon=epsilon, config=config)
            out[name][int(lead)] = {
                "lead": int(lead),
                "react_metric": f"REACT@{int(lead)}",
                "anticipation_rate": res.anticipation_rate,
                "failure_rate": res.failure_rate,
                "km_median_react_frames": res.km_median(),
                "mean_react_uncensored": res.mean_react_uncensored,
            }
    return out


def epsilon_sweep(
    forecasters: ForecastersLike,
    data: TrajectorySet,
    ref,
    events,
    epsilons: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    lead: int = 10,
    config: Optional[ReactConfig] = None,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """REACT vs recovery margin: how much harder is it to beat the reference by epsilon nats."""
    config = config or ReactConfig()
    events_map = normalize_events(events, data)
    out: Dict[str, Dict[float, Dict[str, float]]] = {}
    for name, f in _named_forecasters(forecasters):
        out[name] = {}
        for eps in epsilons:
            res = evaluate(f, data, ref, events_map, lead=lead, epsilon=eps, config=config)
            out[name][float(eps)] = {
                "lead": int(lead),
                "react_metric": f"REACT@{int(lead)}",
                "anticipation_rate": res.anticipation_rate,
                "failure_rate": res.failure_rate,
                "km_median_react_frames": res.km_median(),
            }
    return out


def rate_distortion_anchors(
    ref, lead: int = 10, distortion_targets_m: Sequence[float] = (0.9, 0.3)
) -> Dict[float, float]:
    """Information thresholds eps_star (nats) to localize at each RMS distortion."""
    return {float(d): float(ref.eps_star(lead, d)) for d in distortion_targets_m}


def task_react(
    forecasters: ForecastersLike,
    data: TrajectorySet,
    ref,
    events,
    lead: int = 10,
    distortion_m: float = 0.5,
    config: Optional[ReactConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """REACT to recover enough information to hit a concrete localization task.

    Sets the recovery margin epsilon = eps_star(lead, distortion_m), i.e. the
    nats needed to localize the agent at RMS error ``distortion_m``.
    """
    config = config or ReactConfig()
    eps_star = float(ref.eps_star(lead, distortion_m))
    events_map = normalize_events(events, data)
    out: Dict[str, Dict[str, float]] = {}
    for name, f in _named_forecasters(forecasters):
        res = evaluate(f, data, ref, events_map, lead=lead, epsilon=eps_star, config=config)
        out[name] = {
            "lead": int(lead),
            "react_metric": f"REACT@{int(lead)}",
            "eps_star_nats": eps_star,
            "distortion_m": distortion_m,
            "anticipation_rate": res.anticipation_rate,
            "failure_rate": res.failure_rate,
            "km_median_react_frames": res.km_median(),
        }
    return out


def forecast_accuracy(
    forecasters: ForecastersLike,
    data: TrajectorySet,
    leads: Sequence[int] = (1, 5, 10, 15),
    max_tracks: Optional[int] = None,
    scoring: str = "nll",
) -> Dict[str, Dict[str, object]]:
    """Classic ADE/FDE/NLL-by-lead accuracy summary (NOT REACT).

    Reported so REACT can be shown next to the accuracy metric it complements --
    the central motivating story (high accuracy can hide slow adaptation).
    """
    leads = tuple(int(l) for l in leads)
    ids = data.ids if max_tracks is None else data.ids[:max_tracks]
    out: Dict[str, Dict[str, object]] = {}
    for name, f in _named_forecasters(forecasters):
        per_lead_err: Dict[int, List[float]] = {l: [] for l in leads}
        per_lead_nll: Dict[int, List[float]] = {l: [] for l in leads}
        for tid in ids:
            traj = data[tid]
            pred = f.predict(traj)
            if pred is None or len(pred) == 0:
                continue
            nll_mat = pred.nll(traj, scoring=scoring)
            for lead in leads:
                if lead > pred.horizon:
                    continue
                target, mean, _ = pred.at_lead(lead)
                valid = target < traj.T
                if not valid.any():
                    continue
                err = np.linalg.norm(traj.positions[target[valid]] - mean[valid], axis=1)
                per_lead_err[lead].extend(err.tolist())
                col = nll_mat[:, lead - 1][valid]
                per_lead_nll[lead].extend(col[np.isfinite(col)].tolist())
        disp = {l: float(np.mean(v)) if v else float("nan") for l, v in per_lead_err.items()}
        nll = {l: float(np.mean(v)) if v else float("nan") for l, v in per_lead_nll.items()}
        valid_disp = [disp[l] for l in leads if np.isfinite(disp[l])]
        out[name] = {
            "displacement_error_by_lead_m": disp,
            "nll_by_lead": nll,
            "ade_m": float(np.mean(valid_disp)) if valid_disp else float("nan"),
            "fde_m": disp[leads[-1]],
        }
    return out
