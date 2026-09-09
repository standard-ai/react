"""Adapters that turn user inputs into a Forecaster.

`from_callable` wraps a function ``trajectory -> Predictive``. `from_precomputed`
wraps an already-computed mapping ``id -> Predictive`` (the common case: a user
ran their model offline and just wants REACT on the outputs).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from .base import BaseForecaster


class CallableForecaster(BaseForecaster):
    def __init__(self, fn: Callable[[Trajectory], Optional[Predictive]], name: str = "model"):
        self._fn = fn
        self.name = name

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        return self._fn(traj)


class PrecomputedForecaster(BaseForecaster):
    """Serve predictives from a dict keyed by trajectory id."""

    def __init__(self, predictives: Dict[object, Predictive], name: str = "model"):
        self._by_id = dict(predictives)
        self.name = name

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        return self._by_id.get(traj.id)


def from_callable(
    fn: Callable[[Trajectory], Optional[Predictive]], name: str = "model"
) -> CallableForecaster:
    return CallableForecaster(fn, name=name)


def from_precomputed(
    predictives: Dict[object, Predictive], name: str = "model"
) -> PrecomputedForecaster:
    return PrecomputedForecaster(predictives, name=name)


def as_forecaster(obj, name: str = "model"):
    """Coerce a Forecaster, a dict[id -> Predictive], or a callable into a Forecaster."""
    if isinstance(obj, BaseForecaster):
        return obj
    if hasattr(obj, "predict") and hasattr(obj, "name"):
        return obj
    if isinstance(obj, dict):
        return from_precomputed(obj, name=name)
    if callable(obj):
        return from_callable(obj, name=name)
    raise TypeError(f"cannot interpret {type(obj)!r} as a Forecaster")
