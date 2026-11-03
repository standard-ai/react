"""Pandas-first / file I/O for forecasts, mirroring established conventions.

Two forecast formats are supported, matching what the field actually emits:

1. Per-step bivariate Gaussian ``(mu_x, mu_y, sigma_x, sigma_y, rho)`` -- the
   Social-LSTM / Trajectron++ / Graves parameterization. Explicit covariance
   columns (``cov_xx, cov_xy, cov_yy``) are also accepted. -> Predictive.
2. K trajectories + probabilities (Argoverse / Waymo / nuScenes submission style)
   -- arrays ``(K, H, D)`` with optional scores -> Predictive.from_samples.

A long DataFrame has one row per (track, issue, lead). ``id`` groups tracks, ``issue`` is the
issue frame (the last observed frame, the field's convention) and ``lead`` is 1..H, so a row
forecasts the position at frame issue + lead.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .core.predictive import MixturePredictive, Predictive

__all__ = ["Forecasts"]


def _covs_from_sigma_rho(sx, sy, rho) -> np.ndarray:
    sx = np.asarray(sx, float)
    sy = np.asarray(sy, float)
    rho = np.asarray(rho, float) if rho is not None else np.zeros_like(sx)
    n = len(sx)
    cov = np.empty((n, 2, 2))
    cov[:, 0, 0] = sx**2
    cov[:, 1, 1] = sy**2
    cov[:, 0, 1] = cov[:, 1, 0] = rho * sx * sy
    return cov


def _forecasts_from_dataframe(
    df,
    id: str = "track_id",
    issue: str = "issue_frame",
    lead: str = "lead",
    mean: Sequence[str] = ("mu_x", "mu_y"),
    cov: Optional[Sequence[str]] = ("sigma_x", "sigma_y", "rho"),
) -> Dict[object, Predictive]:
    """Build dict[id -> Predictive] from a long bivariate-Gaussian DataFrame.

    ``cov`` may be ``("sigma_x", "sigma_y", "rho")`` (the documented default) or
    explicit ``("cov_xx", "cov_xy", "cov_yy")``. For higher D pass the matching
    sigma list and use explicit covariance columns.

    Leads are read by value: every track gets horizon H = the file's largest lead and a row
    with lead h fills column h - 1. Leads (or issue/lead cells) absent from the file have NaN
    means and identity covariances, so ``at_lead`` of a missing lead returns all-NaN means.
    """
    out: Dict[object, Predictive] = {}
    dim = len(mean)
    if len(df) and int(df[lead].min()) < 1:
        raise ValueError(f"leads must be >= 1; got {int(df[lead].min())}")
    H = int(df[lead].max()) if len(df) else 0
    for tid, sub in df.groupby(id):
        sub = sub.sort_values([issue, lead])
        issues = np.sort(sub[issue].unique())
        issue_index = {int(t): i for i, t in enumerate(issues)}
        n = len(issues)
        means = np.full((n, H, dim), np.nan)
        covs = np.tile(np.eye(dim), (n, H, 1, 1))
        sigma_convention = cov is not None and any("sigma" in c for c in cov)
        for _, row in sub.iterrows():
            i = issue_index[int(row[issue])]
            h = int(row[lead]) - 1
            means[i, h] = [row[c] for c in mean]
            if cov is None:
                continue
            if sigma_convention:
                rho = row[cov[2]] if len(cov) > 2 else 0.0
                covs[i, h] = _covs_from_sigma_rho([row[cov[0]]], [row[cov[1]]], [rho])[0]
            else:
                c = np.zeros((dim, dim))
                c[0, 0] = row[cov[0]]
                c[0, 1] = c[1, 0] = row[cov[1]]
                c[1, 1] = row[cov[2]]
                covs[i, h] = c
        out[tid] = Predictive(issue_frames=issues.astype(int), means=means, covs=covs, horizon=H)
    return out


class Forecasts:
    """Namespace for forecast ingestion (mirrors rm.Predictive constructors)."""

    from_dataframe = staticmethod(_forecasts_from_dataframe)

    @staticmethod
    def from_npz(path: str) -> Dict[object, Predictive]:
        """Load forecasts written by ``to_npz``, keeping ids' int/str type and mixtures.

        Files from 0.2 pre-releases (arrays keyed ``<id>__issue|means|covs``) still load,
        with string ids and Gaussian forecasts.
        """
        with np.load(path, allow_pickle=False) as data:
            if "__ids__" not in data.files:
                return _from_legacy_npz(data)
            out: Dict[object, Predictive] = {}
            for k, (tid, kind) in enumerate(zip(data["__ids__"], data["__id_types__"])):
                tid = int(tid) if kind == "int" else str(tid)
                horizon = int(data[f"{k}__horizon"])
                if f"{k}__weights" in data.files:
                    out[tid] = MixturePredictive(
                        data[f"{k}__issue"], data[f"{k}__weights"], data[f"{k}__means"],
                        data[f"{k}__covs"], horizon=horizon,
                    )
                else:
                    out[tid] = Predictive(
                        data[f"{k}__issue"], data[f"{k}__means"], data[f"{k}__covs"],
                        horizon=horizon,
                    )
        return out

    @staticmethod
    def to_npz(forecasts: Dict[object, Predictive], path: str) -> None:
        """Write forecasts to one .npz; ids must be ints or strings. A MixturePredictive is
        stored as its weights and components, so it reads back as the same mixture."""
        ids, kinds, bundle = [], [], {}
        for k, (tid, pred) in enumerate(forecasts.items()):
            if isinstance(tid, (int, np.integer)) and not isinstance(tid, bool):
                kinds.append("int")
            elif isinstance(tid, str):
                kinds.append("str")
            else:
                raise TypeError(f"to_npz needs int or str track ids; got {type(tid).__name__}")
            ids.append(str(tid))
            bundle[f"{k}__issue"] = pred.issue_frames
            bundle[f"{k}__horizon"] = np.int64(pred.horizon)
            if isinstance(pred, MixturePredictive):
                bundle[f"{k}__weights"] = pred.weights
                bundle[f"{k}__means"] = pred.component_means
                bundle[f"{k}__covs"] = pred.component_covs
            else:
                bundle[f"{k}__means"] = pred.means
                bundle[f"{k}__covs"] = pred.covs
        bundle["__ids__"] = np.array(ids, dtype=str)
        bundle["__id_types__"] = np.array(kinds, dtype=str)
        np.savez(path, **bundle)


def _from_legacy_npz(data) -> Dict[object, Predictive]:
    """Pre-release layout: arrays keyed ``<id>__issue|means|covs``, ids as strings."""
    ids = sorted({k.rsplit("__", 1)[0] for k in data.files})
    return {
        tid: Predictive(data[f"{tid}__issue"], data[f"{tid}__means"], data[f"{tid}__covs"])
        for tid in ids
    }
