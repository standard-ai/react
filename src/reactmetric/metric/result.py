"""Result containers: EventMeasurement and ReactResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .survival import cluster_bootstrap_ci, kaplan_meier, km_median


@dataclass
class EventMeasurement:
    """REACT measurement of one forecaster on one event."""

    model_name: str
    id: object
    t0: int
    event_type: str
    detection_floor_frames: float
    anticipated: bool
    react_frames: float          # 0 if anticipated
    censored: bool               # window ended before recovery
    pre_elevated: bool           # model already elevated before t0
    coverage_frames: int
    superseded: bool = False     # window truncated by the NEXT event's onset
    excess_react: float = float("nan")


@dataclass
class ReactResult:
    """All measurements for one forecaster plus its reference settings."""

    model_name: str
    reference: str = "climatology"
    epsilon: float = 0.0
    lead: int | None = None  # forecast lead h for REACT@h (required for interpretation)
    measurements: List[EventMeasurement] = field(default_factory=list)
    steady_state_mean_info: float = float("nan")     # mean s (nats)
    steady_state_frac_informative: float = float("nan")  # P(s > 0)
    n_steady_state_samples: int = 0

    @property
    def react_label(self) -> str:
        from ..constants import react_at_lead

        return react_at_lead(self.lead)

    # ---- summary statistics ----

    @property
    def n_measured(self) -> int:
        return len(self.measurements)

    @property
    def anticipation_rate(self) -> float:
        if not self.measurements:
            return float("nan")
        return float(np.mean([m.anticipated for m in self.measurements]))

    @property
    def failure_rate(self) -> float:
        """Fraction never recovered within their window (includes superseded)."""
        if not self.measurements:
            return float("nan")
        return float(np.mean([m.censored for m in self.measurements]))

    @property
    def superseded_rate(self) -> float:
        if not self.measurements:
            return float("nan")
        return float(np.mean([m.superseded for m in self.measurements]))

    @property
    def pre_elevated_rate(self) -> float:
        if not self.measurements:
            return float("nan")
        return float(np.mean([m.pre_elevated for m in self.measurements]))

    @property
    def mean_react_uncensored(self) -> float:
        vals = [m.react_frames for m in self.measurements if not m.censored]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_excess_react(self) -> float:
        vals = [m.excess_react for m in self.measurements if np.isfinite(m.excess_react)]
        return float(np.mean(vals)) if vals else float("nan")

    def km_curve(self, conditional: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """KM survival of REACT; conditional=True drops anticipated events."""
        ms = [m for m in self.measurements if not (conditional and m.anticipated)]
        durations = np.array([m.react_frames for m in ms])
        observed = np.array([not m.censored for m in ms])
        return kaplan_meier(durations, observed)

    def km_median(self, conditional: bool = True) -> float:
        times, surv = self.km_curve(conditional=conditional)
        return km_median(times, surv)

    def by_event_type(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        types = sorted({m.event_type for m in self.measurements})
        for etype in types:
            vals = [
                m.react_frames
                for m in self.measurements
                if m.event_type == etype and not m.censored
            ]
            if vals:
                out[etype] = {
                    "n": len(vals),
                    "mean_react_frames": float(np.mean(vals)),
                    "median_react_frames": float(np.median(vals)),
                }
        return out

    def bootstrap_ci(
        self, stat: str = "anticipation_rate", n_boot: int = 1000, seed: int = 0
    ) -> Tuple[float, float]:
        stat_fns = {
            "anticipation_rate": lambda ms: float(np.mean([m.anticipated for m in ms])),
            "failure_rate": lambda ms: float(np.mean([m.censored for m in ms])),
            "mean_react_uncensored": lambda ms: float(
                np.mean([m.react_frames for m in ms if not m.censored])
            ),
            "mean_excess_react": lambda ms: float(
                np.mean([m.excess_react for m in ms if np.isfinite(m.excess_react)])
            ),
        }
        if stat not in stat_fns:
            raise ValueError(f"unknown stat '{stat}'; choose from {list(stat_fns)}")
        return cluster_bootstrap_ci(self.measurements, stat_fns[stat], n_boot=n_boot, seed=seed)

    def attach_excess(self, oracle_result: ReactResult) -> ReactResult:
        """Set each measurement's excess REACT relative to the oracle floor.

        Excess REACT(e) = REACT_model(e) - REACT_oracle(e), matched per event
        (track id, t0). Uncensored events only.
        """
        floor: Dict[Tuple[object, int], float] = {}
        for m in oracle_result.measurements:
            if not m.censored:
                floor[(m.id, int(m.t0))] = m.react_frames
        for m in self.measurements:
            key = (m.id, int(m.t0))
            if not m.censored and key in floor:
                m.excess_react = m.react_frames - floor[key]
        return self

    def to_dict(self, n_boot: int = 1000) -> Dict:
        label = self.react_label
        return {
            "reference": self.reference,
            "epsilon_nats": self.epsilon,
            "lead": self.lead,
            "react_metric": label,
            "n_measured": self.n_measured,
            "anticipation_rate": self.anticipation_rate,
            "anticipation_rate_ci95": list(self.bootstrap_ci("anticipation_rate", n_boot)),
            "failure_rate": self.failure_rate,
            "superseded_rate": self.superseded_rate,
            "pre_elevated_rate": self.pre_elevated_rate,
            # Values are REACT@lead; bare key names kept for schema stability.
            "km_median_react_frames": self.km_median(),
            "mean_react_uncensored_frames": self.mean_react_uncensored,
            "mean_excess_react_frames": self.mean_excess_react,
            "steady_state_mean_info_nats": self.steady_state_mean_info,
            "steady_state_frac_informative": self.steady_state_frac_informative,
            "n_steady_state_samples": self.n_steady_state_samples,
            "react_by_event_type": self.by_event_type(),
        }
