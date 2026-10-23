"""The climatological reference REACT scores against (paper section 3.5).

``fit_climatology`` fits the paper's reference: the distribution of a person's position h frames
ahead given only their most recent displacement, a speed-conditioned mixture of a moving and a
stopped regime. ``Climatology.fit`` exposes the underlying fitter's options.
"""

from __future__ import annotations

import pickle
from typing import Optional, Sequence

import numpy as np

from ..constants import DEFAULT_DT
from .kinematic import KinematicClimatology, fit_kinematic_climatology


class Climatology:
    """Factory + (de)serialization for the REACT reference."""

    @staticmethod
    def fit(
        data,
        leads: Sequence[int],
        mode: str = "kinematic",
        dt: Optional[float] = None,
        seed: int = 0,
        calib_ids: Optional[Sequence[object]] = None,
        **mode_kwargs,
    ) -> KinematicClimatology:
        """Fit the kinematic climatology on a calibration TrajectorySet (or id -> array).

        Args:
            data: TrajectorySet or dict[id -> (T, 2) array].
            leads: forecast leads to fit references for.
            mode: 'kinematic' (the only mode; kept for compatibility).
            dt: frame interval; inferred from a TrajectorySet if omitted.
            mode_kwargs: forwarded to fit_kinematic_climatology.
        """
        if mode != "kinematic":
            raise ValueError(f"unknown reference mode {mode!r}; REACT uses 'kinematic'")
        if hasattr(data, "as_arrays"):
            arrays = data.as_arrays()
            dt = dt if dt is not None else data.dt
        else:
            arrays = dict(data)
            dt = dt if dt is not None else DEFAULT_DT
        ids = list(calib_ids) if calib_ids is not None else list(arrays.keys())
        dim = next(iter(arrays.values())).shape[1]
        if dim != 2:
            raise ValueError(f"the kinematic climatology needs 2D positions (got D={dim})")
        return fit_kinematic_climatology(arrays, ids, leads, dt=dt, seed=seed, **mode_kwargs)


def fit_climatology(
    data,
    leads: Sequence[int] = (1, 5, 10, 15),
    dt: Optional[float] = None,
    seed: int = 0,
    velocity_window: int = 1,
    isotropic_stop: bool = True,
) -> KinematicClimatology:
    """REACT's climatology (paper section 3.5): the distribution of a person's position
    `lead` frames ahead given only their most recent `velocity_window`-frame displacement.

    A speed-conditioned mixture of a moving regime (in the body frame of the current speed
    and heading) and a stopped regime around the current position, weights and spreads fitted
    jointly by EM on the calibration tracks (stop_model="fitted"). isotropic_stop replaces the
    stopped regime's fitted spread, fixed in world axes and a few percent wider along one of
    them, by its isotropic average; the overlap tables of the exact tests rely on it, since the
    climatology then depends on the speed alone in the mover's body frame."""
    ref = Climatology.fit(
        data,
        leads=leads,
        mode="kinematic",
        velocity_window=velocity_window,
        stop_model="fitted",
        dt=dt,
        seed=seed,
    )
    if isotropic_stop:
        for lead in leads:
            ref.stop_cov[lead] = 0.5 * np.trace(np.asarray(ref.stop_cov[lead], float)) * np.eye(2)
    return ref


def save(reference: KinematicClimatology, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(reference, f)


def load(path: str) -> KinematicClimatology:
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = [
    "Climatology",
    "KinematicClimatology",
    "fit_climatology",
    "fit_kinematic_climatology",
    "save",
    "load",
]
