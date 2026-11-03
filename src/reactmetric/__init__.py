"""reactmetric: REACT (Regime Event Adaptive Correction Time).

Event-aligned evaluation of probabilistic trajectory forecasts around changes in motion
(Breseman and Woollard, "REACT: Measuring Regime Adaptation in Trajectories", TGF 2026).
After a detected regime change, REACT reports whether a forecast anticipated it, how long it
took to become confidently more useful than a climatological reference again, whether it
recovered within the window, and whether the recovery held:

    import reactmetric as rm

    calib, test = data.split(fraction=0.3, seed=0)
    scorer = rm.ReactScorer.fit(calib, info=(3.0, 4.2, 6.0))
    result = scorer.evaluate({"MyModel": my_forecaster}, test)
    result.table(lead=10, distortion=0.1)          # REACT(10, 0.1 m; headline level)
    result.compare("MyModel", "ConstantVelocity", lead=10)

``rm.benchmark`` does the same with kinematic baselines added.
"""

from __future__ import annotations

from . import baselines, constants, datasets, events, forecasters, information, reference, scoring
from .benchmark import benchmark
from .core import MixturePredictive, Predictive, Trajectory, TrajectorySet
from .io import Forecasts
from .metric import kaplan_meier, km_median
from .results import ReactResult, select_info_level
from .scorer import ReactScorer

__version__ = "0.2.0.dev0"

__all__ = [
    "Trajectory",
    "TrajectorySet",
    "Predictive",
    "MixturePredictive",
    "Forecasts",
    "ReactScorer",
    "ReactResult",
    "benchmark",
    "select_info_level",
    "kaplan_meier",
    "km_median",
    "baselines",
    "constants",
    "datasets",
    "events",
    "forecasters",
    "information",
    "reference",
    "scoring",
]


def __getattr__(name):  # viz needs matplotlib; import it on first use
    if name == "viz":
        import importlib

        return importlib.import_module(f"{__name__}.viz")
    raise AttributeError(f"module 'reactmetric' has no attribute {name!r}")
