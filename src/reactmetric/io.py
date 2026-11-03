"""Pandas-first / file I/O for forecasts, mirroring established conventions.

Two forecast formats are supported, matching what the field actually emits:

1. Per-step bivariate Gaussian ``(mu_x, mu_y, sigma_x, sigma_y, rho)`` -- the
   Social-LSTM / Trajectron++ / Graves parameterization. Explicit covariance
   columns (``cov_xx, cov_xy, cov_yy``) are also accepted. -> Predictive.gaussian.
2. K trajectories + probabilities (Argoverse / Waymo / nuScenes submission style)
   -- arrays ``(K, H, D)`` with optional scores -> Predictive.from_samples.

A long DataFrame has one row per (track, issue, lead). ``id`` groups tracks, ``issue`` is the
issue frame (the last observed frame, the field's convention) and ``lead`` is 1..H, so a row
forecasts the position at frame issue + lead.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .core.predictive import Predictive


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


def forecasts_from_dataframe(
    df,
    id: str = "track_id",
    issue: str = "issue_frame",
    lead: str = "lead",
    mean: Sequence[str] = ("mu_x", "mu_y"),
    cov: Optional[Sequence[str]] = ("sigma_x", "sigma_y", "rho"),
) -> Dict[object, Predictive]:
    """Build dict[id -> Predictive.gaussian] from a long bivariate-Gaussian DataFrame.

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

    from_dataframe = staticmethod(forecasts_from_dataframe)

    @staticmethod
    def from_npz(path: str) -> Dict[object, Predictive]:
        """Load forecasts from an .npz bundle with arrays keyed ``<id>__issue``,
        ``<id>__means``, ``<id>__covs``."""
        data = np.load(path, allow_pickle=True)
        ids = {k.split("__")[0] for k in data.files}
        out: Dict[object, Predictive] = {}
        for tid in ids:
            out[tid] = Predictive(
                issue_frames=data[f"{tid}__issue"],
                means=data[f"{tid}__means"],
                covs=data[f"{tid}__covs"],
            )
        return out

    @staticmethod
    def to_npz(forecasts: Dict[object, Predictive], path: str) -> None:
        bundle = {}
        for tid, pred in forecasts.items():
            bundle[f"{tid}__issue"] = pred.issue_frames
            bundle[f"{tid}__means"] = pred.means
            bundle[f"{tid}__covs"] = pred.covs
        np.savez(path, **bundle)
