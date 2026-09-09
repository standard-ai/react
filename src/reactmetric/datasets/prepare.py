"""Optional on-disk dataset_split packaging for large or downloaded corpora.

This is separate from ``TrajectorySet.split`` (in-memory REACT calib/eval).
Use ``prepare`` / ``create_dataset_split`` when you want named partition files
on disk (e.g. ``train.parquet`` / ``eval.parquet``) after download or before
repeated quickstart runs. Loading a single unpartitioned table remains fully
supported and is the default path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .load import _read_table
from .mapping import MappingLike, mappings

FractionSpec = Union[float, Mapping[str, float]]


@dataclass(frozen=True)
class PrepareResult:
    """Paths written by :func:`prepare` / :func:`create_dataset_split`."""

    output_dir: str
    paths: Dict[str, str]
    n_tracks: Dict[str, int]
    source: str
    mapping: str


def _normalize_fractions(dataset_split: FractionSpec) -> Dict[str, float]:
    if isinstance(dataset_split, (int, float)):
        # Single float => eval fraction; remainder is train.
        eval_frac = float(dataset_split)
        if not 0.0 < eval_frac < 1.0:
            raise ValueError("dataset_split float must be in (0, 1)")
        return {"train": 1.0 - eval_frac, "eval": eval_frac}
    if not isinstance(dataset_split, Mapping) or not dataset_split:
        raise ValueError("dataset_split must be a float or a non-empty name->fraction map")
    fracs = {str(k): float(v) for k, v in dataset_split.items()}
    total = sum(fracs.values())
    if total <= 0:
        raise ValueError("dataset_split fractions must sum to a positive value")
    if abs(total - 1.0) > 1e-6:
        fracs = {k: v / total for k, v in fracs.items()}
    return fracs


def _assign_partitions(
    track_ids: Sequence[object],
    fractions: Dict[str, float],
    seed: int,
) -> Dict[str, list]:
    ids = list(track_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    shuffled = [ids[i] for i in order]
    n = len(shuffled)
    names = list(fractions.keys())
    # Allocate by cumulative rounded counts; last partition gets the remainder.
    counts = []
    assigned = 0
    for i, name in enumerate(names):
        if i == len(names) - 1:
            counts.append(n - assigned)
        else:
            c = int(round(fractions[name] * n))
            counts.append(c)
            assigned += c
    # Fix possible rounding drift that overshoots.
    while sum(counts) > n:
        for i in range(len(counts)):
            if counts[i] > 0 and sum(counts) > n:
                counts[i] -= 1
    while sum(counts) < n:
        counts[-1] += 1

    out: Dict[str, list] = {}
    cursor = 0
    for name, count in zip(names, counts):
        out[name] = shuffled[cursor : cursor + count]
        cursor += count
    return out


def _mapping_columns(mapping) -> list:
    cols = [mapping.id, *mapping.coords]
    if mapping.time is not None:
        cols = [mapping.id, mapping.time, *mapping.coords]
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _write_table(df, path: str) -> None:
    lower = path.lower()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if lower.endswith(".csv"):
        df.to_csv(path, index=False)
        return
    if lower.endswith((".parquet", ".pq")):
        try:
            df.to_parquet(path, index=False)
        except ImportError as e:
            raise ImportError(
                "Writing parquet partitions requires pyarrow: "
                "pip install 'reactmetric[datasets]'"
            ) from e
        return
    raise ValueError(f"unsupported output type for {path!r} (use .csv or .parquet)")


def create_dataset_split(
    source: str,
    mapping: MappingLike,
    *,
    output_dir: str,
    dataset_split: FractionSpec = 0.25,
    max_tracks: Optional[int] = None,
    seed: int = 0,
    fmt: str = "parquet",
    columns: Optional[Sequence[str]] = None,
) -> PrepareResult:
    """Create named on-disk dataset_split files from a single source table.

    Args:
        source: Local CSV/parquet path (unpartitioned corpus).
        mapping: Registry name / DatasetMapping / dict used to locate the id column.
        output_dir: Directory to write ``<name>.<fmt>`` partition files into.
        dataset_split: Either an eval fraction in (0, 1) — remainder becomes
            ``train`` — or an explicit ``{name: fraction}`` map
            (e.g. ``{\"train\": 0.7, \"eval\": 0.3}``).
        max_tracks: Optional cap on total tracks before partitioning (useful for
            extremely large downloads).
        seed: RNG seed for track assignment.
        fmt: ``\"parquet\"`` or ``\"csv\"``.
        columns: Optional column subset to retain; defaults to mapping columns.

    Returns:
        :class:`PrepareResult` with written paths and per-partition track counts.

    Notes:
        This does **not** replace ``TrajectorySet.split``. Loaders still accept
        a single unpartitioned file. Use this when you want reusable on-disk
        partitions after download or for large user dumps.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(source)
    if fmt not in {"parquet", "csv"}:
        raise ValueError("fmt must be 'parquet' or 'csv'")

    resolved = mappings.resolve(mapping)
    if resolved.format != "tabular":
        raise ValueError(
            f"create_dataset_split currently supports tabular mappings; "
            f"got format={resolved.format!r}"
        )

    fracs = _normalize_fractions(dataset_split)
    keep_cols = list(columns) if columns is not None else _mapping_columns(resolved)
    if resolved.id not in keep_cols:
        keep_cols = [resolved.id, *keep_cols]

    df = _read_table(source, columns=keep_cols)
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"source missing columns {missing} required for prepare")

    track_ids = list(dict.fromkeys(df[resolved.id].tolist()))
    if max_tracks is not None and len(track_ids) > max_tracks:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(track_ids), size=max_tracks, replace=False)
        track_ids = [track_ids[i] for i in sorted(pick.tolist())]
        df = df[df[resolved.id].isin(set(track_ids))]

    partitions = _assign_partitions(track_ids, fracs, seed=seed)
    os.makedirs(output_dir, exist_ok=True)

    paths: Dict[str, str] = {}
    n_tracks: Dict[str, int] = {}
    for name, ids in partitions.items():
        out_path = os.path.join(output_dir, f"{name}.{fmt}")
        part = df[df[resolved.id].isin(set(ids))]
        _write_table(part, out_path)
        paths[name] = out_path
        n_tracks[name] = len(ids)

    return PrepareResult(
        output_dir=output_dir,
        paths=paths,
        n_tracks=n_tracks,
        source=source,
        mapping=resolved.name,
    )


def prepare(
    source: str,
    mapping: MappingLike,
    *,
    output_dir: str,
    dataset_split: FractionSpec = 0.25,
    max_tracks: Optional[int] = None,
    seed: int = 0,
    fmt: str = "parquet",
    columns: Optional[Sequence[str]] = None,
) -> PrepareResult:
    """Alias for :func:`create_dataset_split` — package a table into partitions."""
    return create_dataset_split(
        source,
        mapping,
        output_dir=output_dir,
        dataset_split=dataset_split,
        max_tracks=max_tracks,
        seed=seed,
        fmt=fmt,
        columns=columns,
    )
