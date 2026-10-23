"""reactmetric: REACT (Regime Event Adaptive Correction Time).

A metric for how quickly trajectory forecasts become genuinely useful again
(vs. a climatological baseline, in information terms) after a disruptive regime
change. Complements ADE/FDE with an adaptation-speed axis.

Typical use::

    import reactmetric as rm

    data = rm.TrajectorySet.from_dataframe(df, id="track_id", coords=["x", "y"], dt=0.1)
    calib, test = data.split(fraction=0.2, seed=0)
    report = rm.benchmark(
        forecasts={"MyModel": my_predictives},
        data=test, calibration=calib,
        baselines=["CTRV", "ConstantVelocity"], oracle="B",
    )
    print(report.table())
"""

from __future__ import annotations

from . import baselines, constants, datasets, events, forecasters, reference, scoring
from .config import ReactConfig
from .constants import react_at_lead
from .core import Predictive, Trajectory, TrajectorySet, collapse_nll_by_target
from .io import Forecasts
from .metric import (
    EventMeasurement,
    ReactResult,
    cluster_bootstrap_ci,
    evaluate,
    kaplan_meier,
)

__version__ = "0.1.0"

__all__ = [
    "Trajectory",
    "TrajectorySet",
    "Predictive",
    "Forecasts",
    "collapse_nll_by_target",
    "ReactConfig",
    "evaluate",
    "ReactResult",
    "EventMeasurement",
    "kaplan_meier",
    "cluster_bootstrap_ci",
    "react_at_lead",
    "constants",
    "reference",
    "events",
    "forecasters",
    "baselines",
    "scoring",
    "datasets",
]

# Optional / later-layer surfaces; imported defensively so a partial install or
# missing optional dependency does not break `import reactmetric`.
try:
    from . import calibration  # noqa: F401

    __all__.append("calibration")
except Exception:  # pragma: no cover
    pass

try:
    from . import oracles  # noqa: F401
    from .oracles import filter_artifacts  # noqa: F401

    __all__ += ["oracles", "filter_artifacts"]
except Exception:  # pragma: no cover
    pass

try:
    from .sweeps import (  # noqa: F401
        epsilon_sweep,
        forecast_accuracy,
        lead_sweep,
        rate_distortion_anchors,
        task_react,
    )

    __all__ += [
        "lead_sweep",
        "epsilon_sweep",
        "rate_distortion_anchors",
        "task_react",
        "forecast_accuracy",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from .benchmark import Benchmark, benchmark  # noqa: F401

    __all__ += ["Benchmark", "benchmark"]
except Exception:  # pragma: no cover
    pass

try:  # REACT as published (TGF 2026); supersedes evaluate/benchmark below
    from . import information, scorer  # noqa: F401
    from .scorer import ReactScorer  # noqa: F401

    __all__ += ["ReactScorer", "information", "scorer"]
except Exception:  # pragma: no cover
    pass

try:
    from . import viz  # noqa: F401

    __all__.append("viz")
except Exception:  # pragma: no cover
    pass
