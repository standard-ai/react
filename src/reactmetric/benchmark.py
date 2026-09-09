"""Benchmark: compare several forecasters (and baselines) on REACT at once.

`Benchmark` is the composable object; `benchmark(...)` is the one-call entry point
that fits the reference, detects events, runs calibration diagnostics, and
evaluates every model + baselines + an oracle floor. Results come back as a
`BenchmarkReport` with a pandas table, pairwise Wilcoxon comparisons, JSON
serialization, and plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ReactConfig
from .core.trajectory import TrajectorySet
from .forecasters.adapters import as_forecaster
from .metric.evaluate import evaluate, normalize_events
from .metric.result import ReactResult
from .results import build_results_json, save_json


@dataclass
class BenchmarkReport:
    results: Dict[str, ReactResult]
    config: ReactConfig
    reference_metadata: Dict = field(default_factory=dict)
    oracle_results: Dict[str, ReactResult] = field(default_factory=dict)
    lead: int = 10
    n_boot: int = 1000

    # ---- tabular summary ----

    def rows(self) -> List[Dict]:
        from .constants import react_at_lead

        label = react_at_lead(self.lead)
        rows = []
        for name, res in self.results.items():
            rows.append({
                "model": name,
                "lead": self.lead,
                "react_metric": label,
                "n_events": res.n_measured,
                "anticipation_rate": res.anticipation_rate,
                "failure_rate": res.failure_rate,
                "superseded_rate": res.superseded_rate,
                # Stable keys + lead-tagged aliases (values are always REACT@lead).
                "km_median_react": res.km_median(),
                f"km_median_{label}": res.km_median(),
                "mean_react_uncensored": res.mean_react_uncensored,
                f"mean_{label}_uncensored": res.mean_react_uncensored,
                "mean_excess_react": res.mean_excess_react,
                f"mean_excess_{label}": res.mean_excess_react,
                "steady_state_mean_info": res.steady_state_mean_info,
                "steady_state_frac_informative": res.steady_state_frac_informative,
            })
        return rows

    def table(self):
        rows = self.rows()
        try:
            import pandas as pd

            return pd.DataFrame(rows).set_index("model")
        except ImportError:
            return rows

    def wilcoxon(self, baseline: Optional[str] = None):
        """Pairwise Wilcoxon signed-rank tests on matched per-event REACT.

        Compares each model against ``baseline`` (default: the first added model)
        over events both measured (uncensored). Lower REACT@lead is better.
        """
        from scipy.stats import wilcoxon as _wilcoxon

        from .constants import react_at_lead

        names = list(self.results.keys())
        if not names:
            return {}
        base = baseline or names[0]
        label = react_at_lead(self.lead)
        base_map = {
            (m.id, int(m.t0)): m.react_frames
            for m in self.results[base].measurements
            if not m.censored
        }
        out = {}
        for name in names:
            if name == base:
                continue
            other = {
                (m.id, int(m.t0)): m.react_frames
                for m in self.results[name].measurements
                if not m.censored
            }
            keys = sorted(set(base_map) & set(other), key=lambda k: (str(k[0]), k[1]))
            a = np.array([base_map[k] for k in keys])
            b = np.array([other[k] for k in keys])
            if len(keys) < 5 or np.all(a == b):
                out[name] = {"n_paired": len(keys), "statistic": float("nan"),
                             "p_value": float("nan")}
                continue
            stat, p = _wilcoxon(a, b)
            out[name] = {
                "n_paired": len(keys),
                "statistic": float(stat),
                "p_value": float(p),
                f"median_{label}_diff_vs_{base}": float(np.median(b - a)),
            }
        try:
            import pandas as pd

            return pd.DataFrame(out).T
        except ImportError:
            return out

    # ---- serialization / plots ----

    def to_dict(self, include_measurements: bool = False) -> Dict:
        all_results = dict(self.results)
        all_results.update(self.oracle_results)
        return build_results_json(
            all_results, config=self.config, reference_metadata=self.reference_metadata,
            n_boot=self.n_boot, include_measurements=include_measurements,
        )

    def to_json(self, path: str, include_measurements: bool = False) -> None:
        save_json(self.to_dict(include_measurements=include_measurements), path)

    def plot_km(self, save: Optional[str] = None, conditional: bool = True):
        from .viz.plots import km_recovery_curves

        return km_recovery_curves(
            self.results, save=save, conditional=conditional, lead=self.lead
        )


class Benchmark:
    def __init__(
        self, reference, events, config: Optional[ReactConfig] = None, lead: Optional[int] = None
    ):
        self.reference = reference
        self._events = events
        self.config = config or ReactConfig()
        self.lead = int(
            lead if lead is not None else (self.config.lead or self.config.forecast_horizon)
        )
        self._models: List[Tuple[str, object]] = []
        self._oracles: List[str] = []
        self._measurement_noise: Optional[float] = None

    def set_measurement_noise(self, sigma: float) -> Benchmark:
        self._measurement_noise = float(sigma)
        return self

    def add(self, name: str, forecaster) -> Benchmark:
        self._models.append((name, as_forecaster(forecaster, name=name)))
        return self

    def add_baseline(self, name: str, **kwargs) -> Benchmark:
        from . import baselines

        if self._measurement_noise is not None and name in ("CTRV", "IMM"):
            kwargs.setdefault("measurement_noise", self._measurement_noise)
        self._models.append((name, baselines.build(name, **kwargs)))
        return self

    def add_oracle(self, which: str) -> Benchmark:
        self._oracles.append(which.upper())
        return self

    def run(
        self,
        data: TrajectorySet,
        epsilon: Optional[float] = None,
        reference_kind: str = "climatology",
        scoring: str = "nll",
        run_calibration: bool = True,
    ) -> BenchmarkReport:
        epsilon = float(self.config.epsilon if epsilon is None else epsilon)
        events_map = normalize_events(self._events, data)

        results: Dict[str, ReactResult] = {}
        for name, f in self._models:
            if run_calibration and reference_kind == "climatology" and scoring == "nll":
                from .calibration.report import report as calib_report

                try:
                    calib_report(f, data, self.reference, events_map, lead=self.lead,
                                 config=self.config)
                except Exception:
                    pass
            results[name] = evaluate(
                f, data, self.reference, events_map, lead=self.lead, epsilon=epsilon,
                reference_kind=reference_kind, scoring=scoring, config=self.config, name=name,
            )

        oracle_results = self._run_oracles(data, events_map, epsilon)
        excess_floor = oracle_results.get("oracle_C") or oracle_results.get("oracle_B")
        if excess_floor is not None:
            for res in results.values():
                res.attach_excess(excess_floor)

        meta = self.reference.metadata() if hasattr(self.reference, "metadata") else {}
        return BenchmarkReport(
            results=results, config=self.config, reference_metadata=meta,
            oracle_results=oracle_results, lead=self.lead, n_boot=self.config.n_boot,
        )

    def _run_oracles(self, data, events_map, epsilon) -> Dict[str, ReactResult]:
        from . import oracles

        out: Dict[str, ReactResult] = {}
        need_b = "B" in self._oracles or "C" in self._oracles
        if need_b:
            sigma = self._measurement_noise if self._measurement_noise is not None else 0.05
            ob = oracles.ClimatologySeeded(
                horizon=self.config.forecast_horizon, history=10, sigma_r=sigma
            )
            res_b = evaluate(ob, data, self.reference, events_map, lead=self.lead,
                             epsilon=epsilon, config=self.config)
            out["oracle_B"] = res_b
            if "C" in self._oracles:
                out["oracle_C"] = oracles.oracle_c_result(res_b, events_map)
        return out


def benchmark(
    forecasts,
    data: TrajectorySet,
    calibration: TrajectorySet,
    baselines: Sequence[str] = (),
    oracle: Optional[str] = None,
    events=None,
    reference_mode: str = "kinematic",
    lead: int = 10,
    epsilon: float = 0.0,
    scoring: str = "nll",
    config: Optional[ReactConfig] = None,
) -> BenchmarkReport:
    """One-call REACT benchmark: fit reference, detect events, evaluate everything.

    Args:
        forecasts: dict[name -> (dict[id->Predictive] | Forecaster)] or a single
            Forecaster / dict[id->Predictive].
        data: evaluation TrajectorySet.
        calibration: calibration TrajectorySet (reference fitting + noise estimate).
        baselines: baseline names to auto-build and include.
        oracle: "B" or "C" to include a recovery-floor oracle (and excess-REACT).
        events: optional EventSource / events_map; defaults to the kinematic detector.
        reference_mode: 'kinematic' | 'global' | 'position'.
    """
    from .baselines import CalibratedNoise
    from .events import KinematicChangepoints
    from .reference import Climatology

    config = config or ReactConfig(lead=lead, epsilon=epsilon, forecast_horizon=max(lead, 15))
    # Keep config.lead in sync with the headline lead used for evaluation.
    config.lead = int(lead)
    leads = tuple(sorted(set(config.lead_sweep) | {lead}))
    ref = Climatology.fit(calibration, leads=leads, mode=reference_mode, k=config.clim_k)
    noise = CalibratedNoise.estimate(calibration)

    if events is None:
        events = KinematicChangepoints()

    bench = Benchmark(reference=ref, events=events, config=config, lead=lead)
    bench.set_measurement_noise(noise.measurement_noise)

    # Models. A dict whose values are all Predictive is a single model's
    # id -> Predictive mapping; otherwise it's name -> (Forecaster | mapping).
    from .core.predictive import Predictive

    if isinstance(forecasts, dict) and forecasts and all(
        isinstance(v, Predictive) for v in forecasts.values()
    ):
        bench.add("model", forecasts)
    elif isinstance(forecasts, dict):
        for name, obj in forecasts.items():
            bench.add(name, obj)
    else:
        bench.add(getattr(forecasts, "name", "model"), forecasts)

    for b in baselines:
        bench.add_baseline(b)
    if oracle:
        bench.add_oracle(oracle)

    return bench.run(data, epsilon=epsilon, scoring=scoring)
