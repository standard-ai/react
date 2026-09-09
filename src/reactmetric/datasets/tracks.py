"""Standard AI / Standard Cognition Tracks adapter.

Tracks is a 10 Hz retail pedestrian trajectory corpus. The public packaging is
not assumed to be pre-partitioned into train/eval files: load a single local
CSV/parquet (or a directory containing one), then use ``TrajectorySet.split``
for calib/eval inside REACT / quickstart.

Column and unit assumptions live in ``reactmetric.datasets.mappings.tracks``.
An optional ``dataset_split`` only selects a named file under a directory when
the user already has partitioned files on disk.
"""

from __future__ import annotations

import os
from typing import Optional

from ..core.trajectory import TrajectorySet
from .load import load
from .mapping import mappings


def _resolve_tracks_path(
    path: Optional[str] = None,
    root: Optional[str] = None,
    dataset_split: Optional[str] = None,
) -> str:
    if path is not None:
        if os.path.isdir(path):
            root = path
            path = None
        else:
            return path

    if root is None:
        raise FileNotFoundError(
            "Tracks loading expects a local file or directory. Pass path=/file.csv|.parquet "
            "or root=/dir containing a tracks table. Calib/eval partitioning is done after "
            "load with TrajectorySet.split — not via a required external train/eval file."
        )

    if dataset_split is not None:
        for ext in (".parquet", ".csv"):
            candidate = os.path.join(root, f"{dataset_split}{ext}")
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"no {dataset_split}.parquet/.csv under {root}. "
            "If you have a single table, pass path= to that file and omit dataset_split."
        )

    # Prefer a single obvious table in the directory.
    preferred = (
        "tracks.parquet",
        "tracks.csv",
        "data.parquet",
        "data.csv",
        "eval.parquet",
        "eval.csv",
        "train.parquet",
        "train.csv",
    )
    for name in preferred:
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate

    found = sorted(
        f
        for f in os.listdir(root)
        if f.lower().endswith((".parquet", ".csv"))
    )
    if len(found) == 1:
        return os.path.join(root, found[0])
    if not found:
        raise FileNotFoundError(f"no CSV/parquet tracks table found under {root}")
    raise FileNotFoundError(
        f"multiple tables under {root}: {found}. Pass path= to the file to use, "
        "or dataset_split= to select a named partition file."
    )


def load_tracks(
    path: Optional[str] = None,
    root: Optional[str] = None,
    dataset_split: Optional[str] = None,
    use_3d: bool = False,
    max_tracks: Optional[int] = None,
    dt: Optional[float] = None,
    min_length: Optional[int] = None,
) -> TrajectorySet:
    """Load Standard AI Tracks from a local CSV/parquet table.

    Args:
        path: Direct path to a tracks table, or a directory of tables.
        root: Directory containing a tracks table when ``path`` is not given.
        dataset_split: Optional name of a partitioned file under ``root`` /
            directory ``path`` (e.g. ``\"eval\"`` -> ``eval.parquet``). Not
            required; a single unpartitioned file is the default path.
        use_3d: Keep z via the mapping's ``coords_3d`` extras.
        max_tracks: Optional track cap.
        dt / min_length: Optional overrides of the registered tracks mapping.
    """
    resolved = _resolve_tracks_path(path=path, root=root, dataset_split=dataset_split)
    mapping = mappings.tracks
    return load(
        resolved,
        mapping,
        max_tracks=max_tracks,
        dt=dt,
        min_length=min_length if min_length is not None else mapping.min_length,
        use_3d=use_3d,
    )


def prepare_tracks(
    source: str,
    output_dir: str,
    *,
    dataset_split=0.25,
    max_tracks: Optional[int] = None,
    seed: int = 0,
    fmt: str = "parquet",
):
    """Package a single Tracks table into on-disk dataset_split files.

    After this, load with ``load_tracks(root=output_dir, dataset_split=\"eval\")``.
    For in-memory REACT calib/eval without packaging, skip this and call
    ``TrajectorySet.split`` after ``load_tracks(path=...)``.
    """
    from .prepare import prepare

    return prepare(
        source,
        mappings.tracks,
        output_dir=output_dir,
        dataset_split=dataset_split,
        max_tracks=max_tracks,
        seed=seed,
        fmt=fmt,
    )
