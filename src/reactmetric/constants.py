"""Shared constants for reactmetric.

The library is domain-agnostic, but spatial (2D/3D) trajectory forecasting is
the first-class use case, so the defaults below match the 10 Hz pedestrian
setting REACT was developed on. Override per call where your domain differs.
"""

from __future__ import annotations

__all__ = ["DEFAULT_DT"]

# Default frame interval (seconds). REACT's frame-count settings (onset window, 150-frame
# window cap) are those of the paper, at 10 Hz. Trajectories carry their own dt.
DEFAULT_DT: float = 0.1

# Internal: numerical floor added to covariance diagonals before inversion.
COV_REGULARIZATION: float = 1e-6
