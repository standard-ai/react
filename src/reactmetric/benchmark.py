"""One call: your forecasts against kinematic baselines and Oracle B under REACT.

    result = rm.benchmark({"MyModel": my_predictives}, test, calibration=calib)
    result.table(lead=10, distortion=0.1)

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
    info: Optional[Sequence[float]] = None,
    leads: Optional[Sequence[int]] = None,
    distortions: Optional[Sequence[float]] = None,
    n_jobs: int = 1,
    scorer: Optional[ReactScorer] = None,
    event_set: str = "shared",
    **fit_kwargs,
) -> ReactResult:
    """Score `forecasts` (name -> Forecaster or {track id: Predictive}) and the named baselines
    on the test tracks `data`, with REACT fitted on the `calibration` tracks.

    info: assumed information levels (nats; default 3, 4.2, 6). leads: default 1, 5, 10, 15.
    distortions: target distances d (m) besides d = infinity (default 0.1, 0.3). fit_kwargs go
    to ReactScorer.fit (e.g. sigma, events, window, table_dir). Pass a fitted `scorer` to reuse
    one; info, leads and distortions must then be left out or match it. With n_jobs > 1, call
    this under `if __name__ == "__main__":`."""
    models = dict(forecasts)
    clash = set(models) & set(baselines)
    if clash:
        raise ValueError(f"model names clash with baselines: {sorted(clash)}")
    models.update(build_baselines(baselines, calibration))
    if scorer is None:
        scorer = ReactScorer.fit(
            calibration, info=(3.0, 4.2, 6.0) if info is None else info,
            leads=LEADS if leads is None else leads,
            distortions=(0.1, 0.3) if distortions is None else distortions,
            n_jobs=n_jobs, **fit_kwargs,
        )
    else:
        have = {"info": sorted({lv.info for v in scorer.levels.values() for lv in v}),
                "leads": list(scorer.leads), "distortions": list(scorer.distortions)}
        asked = {"info": info, "leads": leads, "distortions": distortions}
        for k, v in asked.items():
            if v is not None and sorted(float(x) for x in v) != sorted(float(x) for x in have[k]):
                raise ValueError(f"{k}={list(v)} conflicts with the scorer passed ({have[k]}); "
                                 "leave it out or refit")
        if fit_kwargs:
            raise ValueError(f"fit arguments {sorted(fit_kwargs)} have no effect with a fitted "
                             "scorer")
    return scorer.evaluate(models, data, n_jobs=n_jobs, event_set=event_set)
