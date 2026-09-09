"""Forecaster interface and adapters."""

from .adapters import (
    CallableForecaster,
    PrecomputedForecaster,
    as_forecaster,
    from_callable,
    from_precomputed,
)
from .base import BaseForecaster, Forecaster

__all__ = [
    "Forecaster",
    "BaseForecaster",
    "from_callable",
    "from_precomputed",
    "as_forecaster",
    "CallableForecaster",
    "PrecomputedForecaster",
]
