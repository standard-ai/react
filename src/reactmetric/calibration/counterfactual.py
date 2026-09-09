"""Opt-in counterfactual: 'what would REACT be if covariances were rescaled?'

This is NOT the headline number and is never applied automatically. It answers a
clearly-labeled hypothetical: if you applied a single scalar temperature to every
predictive covariance (the standard cheap recalibration), how would REACT change?

Two caveats motivate keeping it opt-in (see the plan's calibration policy):
  (a) it scores a DIFFERENT, hypothetical forecast, not the one you deliver; and
  (b) a steady-state scalar fix does not even guarantee the model clears the
      event-region recovery threshold, so a "calibrated" counterfactual can still
      fail REACT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import ReactConfig
from ..core.predictive import Predictive
from ..forecasters.adapters import as_forecaster
from ..forecasters.base import BaseForecaster
from ..metric.result import ReactResult
from .report import report as calibration_report


class _TemperatureScaled(BaseForecaster):
    """Wrap a forecaster, multiplying every predictive covariance by ``temperature``."""

    def __init__(self, inner, temperature: float):
        self._inner = inner
        self._t = float(temperature)
        self.name = f"{inner.name}@T={temperature:.2f}"

    def predict(self, traj) -> Optional[Predictive]:
        pred = self._inner.predict(traj)
        if pred is None:
            return None
        return Predictive(pred.issue_frames, pred.means, pred.covs * self._t, pred.horizon)


@dataclass
class CounterfactualResult:
    temperature: float
    delivered: ReactResult
    recalibrated: ReactResult

    def summary(self) -> str:
        from ..constants import react_at_lead

        lead = getattr(self.delivered, "lead", None)
        label = react_at_lead(lead)
        return (
            f"Counterfactual scalar recalibration (T={self.temperature:.3f}) -- HYPOTHETICAL:\n"
            f"  anticipation  delivered {self.delivered.anticipation_rate:.3f} "
            f"-> recalibrated {self.recalibrated.anticipation_rate:.3f}\n"
            f"  failure       delivered {self.delivered.failure_rate:.3f} "
            f"-> recalibrated {self.recalibrated.failure_rate:.3f}\n"
            f"  KM median {label} delivered {self.delivered.km_median():.1f} "
            f"-> recalibrated {self.recalibrated.km_median():.1f} frames\n"
            f"  NOTE: this rescales covariances and scores a different forecast than "
            f"the one you delivered; not the headline {label}."
        )


def counterfactual_react(
    forecaster,
    data,
    ref,
    events,
    lead: Optional[int] = None,
    temperature: Optional[float] = None,
    epsilon: float = 0.0,
    reference_kind: str = "climatology",
    config: Optional[ReactConfig] = None,
) -> CounterfactualResult:
    """REACT delivered vs. with a scalar covariance temperature (opt-in only).

    ``temperature`` defaults to the steady-state estimate ``mean_mahalanobis / D``
    (the value that would make the coverage chi2 nominal).
    """
    from ..metric.evaluate import evaluate

    config = config or ReactConfig()
    lead = int(lead if lead is not None else (config.lead or config.forecast_horizon))
    inner = as_forecaster(forecaster, name=getattr(forecaster, "name", "model"))

    if temperature is None:
        rep = calibration_report(inner, data, ref, events, lead=lead, config=config,
                                 emit_warning=False)
        temperature = rep.temperature if np.isfinite(rep.temperature) else 1.0

    delivered = evaluate(inner, data, ref, events, lead=lead, epsilon=epsilon,
                         reference_kind=reference_kind, config=config)
    scaled = _TemperatureScaled(inner, temperature)
    recal = evaluate(scaled, data, ref, events, lead=lead, epsilon=epsilon,
                    reference_kind=reference_kind, config=config)
    return CounterfactualResult(temperature=float(temperature), delivered=delivered,
                                recalibrated=recal)
