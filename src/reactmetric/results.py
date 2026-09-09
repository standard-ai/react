"""Results JSON schema and serialization for REACT runs.

A single, documented JSON layout so results are portable across runs, machines,
and the CLI. Numpy types are coerced to plain Python so the output is stable and
diffable.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np

SCHEMA_VERSION = "react-results/1.1"


def _to_native(obj):
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def measurement_to_dict(m) -> Dict:
    return {
        "id": m.id,
        "t0": int(m.t0),
        "event_type": m.event_type,
        "anticipated": bool(m.anticipated),
        "react_frames": float(m.react_frames),
        "censored": bool(m.censored),
        "superseded": bool(m.superseded),
        "pre_elevated": bool(m.pre_elevated),
        "detection_floor_frames": float(m.detection_floor_frames),
        "excess_react": float(m.excess_react),
    }


def result_to_dict(result, n_boot: int = 1000, include_measurements: bool = False) -> Dict:
    d = dict(result.to_dict(n_boot=n_boot))
    d["model_name"] = result.model_name
    if include_measurements:
        d["measurements"] = [measurement_to_dict(m) for m in result.measurements]
    return d


def build_results_json(
    results: Dict[str, object],
    config: Optional[object] = None,
    reference_metadata: Optional[Dict] = None,
    extra: Optional[Dict] = None,
    n_boot: int = 1000,
    include_measurements: bool = False,
) -> Dict:
    """Assemble the canonical results dict for a set of named ReactResults."""
    payload = {
        "schema": SCHEMA_VERSION,
        "reference": reference_metadata or {},
        "models": {
            name: result_to_dict(res, n_boot=n_boot, include_measurements=include_measurements)
            for name, res in results.items()
        },
    }
    if config is not None:
        from .constants import react_at_lead

        lead = getattr(config, "lead", None)
        payload["config"] = {
            "lead": lead,
            "react_metric": react_at_lead(lead),
            "epsilon": getattr(config, "epsilon", None),
            "forecast_horizon": getattr(config, "forecast_horizon", None),
            "onset_consecutive": getattr(config, "onset_consecutive", None),
            "recovery_consecutive": getattr(config, "recovery_consecutive", None),
            "max_react_frames": getattr(config, "max_react_frames", None),
        }
    if extra:
        payload.update(extra)
    return _to_native(payload)


def save_json(payload: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(_to_native(payload), f, indent=2)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
