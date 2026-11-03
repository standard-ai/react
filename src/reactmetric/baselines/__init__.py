"""Baseline forecasters and a registry for benchmark.add_baseline."""

from .constant_velocity import ConstantVelocity

__all__ = ["ConstantVelocity", "available", "build"]

# Lazily import heavier baselines so the core install stays light.
_OPTIONAL = {}
try:
    from .ctrv import CTRV  # noqa: F401

    _OPTIONAL["CTRV"] = CTRV
    __all__.append("CTRV")
except Exception:  # pragma: no cover
    pass
try:
    from .imm import IMM  # noqa: F401

    _OPTIONAL["IMM"] = IMM
    __all__.append("IMM")
except Exception:  # pragma: no cover
    pass


_REGISTRY = {"ConstantVelocity": ConstantVelocity, **_OPTIONAL}

_GRU_NAMES = ("GRUPosition", "GRUOrientation")


def available() -> list:
    """Names usable with benchmark.add_baseline / build()."""
    names = list(_REGISTRY.keys())
    try:
        import torch  # noqa: F401

        from .gru import GRUOrientation, GRUPosition  # noqa: F401

        names.extend(_GRU_NAMES)
    except Exception:  # pragma: no cover
        pass
    return names


def build(name: str, **kwargs):
    """Construct a baseline forecaster by name."""
    if name in _GRU_NAMES:
        from .gru import GRUOrientation, GRUPosition

        return (GRUOrientation if name == "GRUOrientation" else GRUPosition)(**kwargs)
    if name not in _REGISTRY:
        raise KeyError(f"unknown baseline '{name}'; available: {available()}")
    return _REGISTRY[name](**kwargs)


# Re-export GRU classes when torch is present so `rm.baselines.GRUPosition` works.
try:
    from .gru import GRUOrientation, GRUPosition

    __all__ += ["GRUPosition", "GRUOrientation"]
except Exception:  # pragma: no cover
    pass
