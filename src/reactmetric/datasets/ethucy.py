"""ETH/UCY adapter: the ungated public default for tutorials and CI smoke runs.

ETH/UCY ships either as preprocessed whitespace-delimited rows
``frame  ped_id  x  y`` or, for raw ETH/EWAP ``obsmat.txt`` files, as
``frame  ped_id  pos_x  pos_z  pos_y  v_x  v_z  v_y``. This loader maps either
format to a TrajectorySet. We ship the ADAPTER only -- point it at a local copy
of the (freely available) data.

Column/dt defaults live in ``reactmetric.datasets.mappings.ethucy``.
"""

from __future__ import annotations

from typing import Optional

from ..core.trajectory import TrajectorySet
from .mapping import mappings

_SCENES = ("eth", "hotel", "univ", "zara1", "zara2")


def load_eth_ucy(
    scene_or_path: str = "eth",
    root: Optional[str] = None,
    dt: Optional[float] = None,
    min_length: Optional[int] = None,
) -> TrajectorySet:
    """Load an ETH/UCY scene into a TrajectorySet.

    Mapping defaults come from ``mappings.ethucy`` (dt, min_length). Raw
    obsmat ``x z y`` layout is still auto-detected from column count.
    """
    import os

    from .load import load as load_mapped

    mapping = mappings.ethucy
    min_length = mapping.min_length if min_length is None else min_length
    dt = mapping.dt if dt is None else dt

    if os.path.exists(scene_or_path):
        path = scene_or_path
    else:
        scene = scene_or_path.lower()
        if scene not in _SCENES:
            raise ValueError(
                f"unknown ETH/UCY scene '{scene}'; choose from {_SCENES} or pass a path"
            )
        if root is None:
            raise FileNotFoundError(
                f"provide root=<dir with {scene}.txt> or a direct file path. "
                f"ETH/UCY data is freely available; we ship only the adapter."
            )
        path = os.path.join(root, f"{scene}.txt")

    return load_mapped(
        path,
        mapping.with_updates(dt=dt, min_length=min_length),
    )
