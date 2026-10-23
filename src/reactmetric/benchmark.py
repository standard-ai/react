"""One call: your forecasts against kinematic baselines and Oracle B under REACT.

    result = rm.benchmark({"MyModel": my_predictives}, test, calibration=calib)
    result.table(lead=10, info=4.2, distortion=0.1)

A convenience over ReactScorer: it fits the scorer on the calibration tracks, adds the requested
baselines (constant velocity is fitted by maximum likelihood on the calibration tracks, as in
the paper; CTRV and IMM keep their default noise), and evaluates everything on the test tracks.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from . import baselines as _baselines
from .scorer import LEADS, ReactResult, ReactScorer

DEFAULT_BASELINES = ("ConstantVelocity", "CTRV", "IMM")


def build_baselines(names: Sequence[str], calibration=None) -> dict:
    """Baseline forecasters by name; ConstantVelocity is fitted on `calibration` when given."""
    out = {}
    for name in names:
        if name == "ConstantVelocity" and calibration is not None:
            out[name] = _baselines.ConstantVelocity.fit(calibration)
        else:
            out[name] = _baselines.build(name)
    return out


def benchmark(
    forecasts: Mapping[str, object],
    data,
    calibration,
    baselines: Sequence[str] = DEFAULT_BASELINES,
    info: Sequence[float] = (3.0, 4.2, 6.0),
    leads: Sequence[int] = LEADS,
    distortions: Sequence[float] = (0.1, 0.3),
    n_jobs: int = 1,
    scorer: Optional[ReactScorer] = None,
    **fit_kwargs,
) -> ReactResult:
    """Score `forecasts` (name -> Forecaster or {track id: Predictive}) and the named baselines
    on the test tracks `data`, with REACT fitted on the `calibration` tracks.

    info: assumed information levels (nats); every reported number carries its level.
    distortions: target distances d (m) besides d = infinity. fit_kwargs go to
    ReactScorer.fit (e.g. sigma, events, table_dir). Pass a fitted `scorer` to reuse one."""
    models = dict(forecasts)
    clash = set(models) & set(baselines)
    if clash:
        raise ValueError(f"model names clash with baselines: {sorted(clash)}")
    models.update(build_baselines(baselines, calibration))
    if scorer is None:
        scorer = ReactScorer.fit(calibration, info=info, leads=leads, distortions=distortions,
                                 n_jobs=n_jobs, **fit_kwargs)
    return scorer.evaluate(models, data, n_jobs=n_jobs)
