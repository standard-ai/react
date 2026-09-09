"""Calibration diagnostics: call out, never correct.

REACT scores the forecast AS DELIVERED. A model that recovers slowly because its
uncertainty is wrong has genuinely failed -- that is information REACT should
report, not erase. So this module DIAGNOSES calibration (and catches gross unit
bugs) and emits loud warnings, but never modifies the forecast or the headline
REACT number.

The statistics are computed on the steady-state (event-masked) frames of the
calibration split, where forecasts should be well-calibrated if they are
calibrated anywhere:

- coverage_chi2: mean squared Mahalanobis distance; equals D when calibrated,
  > D when overconfident, < D when underconfident.
- pit: per-axis probability-integral-transform uniformity (KS p-value); catches
  bias and variance miscalibration the scalar chi2 can miss.
- sharpness: mean predictive log-volume (smaller = sharper); only meaningful
  alongside calibration.
- mean_bias: per-axis mean residual (should be ~0).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..config import ReactConfig
from ..forecasters.adapters import as_forecaster


@dataclass
class CalibrationReport:
    """Diagnostics for one forecaster's predictive covariances."""

    model_name: str
    dim: int
    n_samples: int
    coverage_chi2: float                 # mean squared Mahalanobis (calibrated == D)
    pit: Tuple[float, ...]               # per-axis KS p-value vs uniform
    sharpness: float                     # mean 0.5 * logdet(cov) (nats)
    mean_bias: Tuple[float, ...]         # per-axis mean residual
    warnings: List[str] = field(default_factory=list)
    temperature: float = 1.0             # mean_mahalanobis / D (for the counterfactual)

    @property
    def is_calibrated(self) -> bool:
        ratio = self.coverage_chi2 / max(self.dim, 1)
        chi2_ok = 0.7 <= ratio <= 1.5
        pit_ok = all(p > 0.01 for p in self.pit)
        bias_ok = all(abs(b) < 0.5 for b in self.mean_bias)
        return bool(chi2_ok and pit_ok and bias_ok)

    def summary(self) -> str:
        status = "CALIBRATED" if self.is_calibrated else "MISCALIBRATED"
        lines = [
            f"Calibration report for '{self.model_name}' [{status}]",
            f"  samples: {self.n_samples}  dim: {self.dim}",
            f"  coverage chi2 (mean Mahalanobis^2): {self.coverage_chi2:.3f}  "
            f"(calibrated = {self.dim}; ratio {self.coverage_chi2 / max(self.dim,1):.2f})",
            f"  PIT KS p-values per axis: "
            f"[{', '.join(f'{p:.3f}' for p in self.pit)}]",
            f"  mean bias per axis: [{', '.join(f'{b:+.3f}' for b in self.mean_bias)}]",
            f"  sharpness (mean 0.5*logdet): {self.sharpness:.3f} nats",
            f"  scalar-temperature (counterfactual only): {self.temperature:.3f}",
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    def plot(self, save: Optional[str] = None):
        from ..viz.plots import plot_calibration

        return plot_calibration(self, save=save)


def _steady_state_residuals(
    forecaster,
    data,
    events_map: Dict[object, List],
    ref,
    config: ReactConfig,
    lead: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (residual, cov) pairs on event-masked calibration frames."""
    calib, _ = data.split(fraction=config.calib_fraction, seed=config.calib_seed)
    ids = calib.ids[: config.calib_max_tracks]
    resids, covs = [], []
    for tid in ids:
        traj = data[tid]
        pred = forecaster.predict(traj)
        if pred is None or len(pred) == 0:
            continue
        target, mean, cov = pred.at_lead(lead)
        valid = target < traj.T
        if not valid.any():
            continue
        target_v = target[valid]
        r = traj.positions[target_v] - mean[valid]
        c = cov[valid]
        mask = np.ones(len(target_v), dtype=bool)
        for ev in events_map.get(tid, []):
            lo = ev.t0 - config.event_mask_pre_frames
            hi = ev.t0 + config.event_mask_post_frames
            mask &= ~((target_v >= lo) & (target_v <= hi))
        if mask.any():
            resids.append(r[mask])
            covs.append(c[mask])
    if not resids:
        return np.empty((0, data.D)), np.empty((0, data.D, data.D))
    return np.concatenate(resids), np.concatenate(covs)


def report(
    forecaster,
    data,
    ref,
    events=None,
    lead: Optional[int] = None,
    config: Optional[ReactConfig] = None,
    emit_warning: bool = True,
) -> CalibrationReport:
    """Diagnose a forecaster's calibration on the steady-state calibration split.

    Never modifies the forecast. Emits a ``UserWarning`` when materially
    miscalibrated (so it surfaces in benchmark runs) unless ``emit_warning`` is
    False.
    """
    from ..metric.evaluate import normalize_events

    config = config or ReactConfig()
    lead = int(lead if lead is not None else (config.lead or config.forecast_horizon))
    forecaster = as_forecaster(forecaster, name=getattr(forecaster, "name", "model"))
    events_map = normalize_events(events, data) if events is not None else {}

    resid, cov = _steady_state_residuals(forecaster, data, events_map, ref, config, lead)
    dim = data.D
    if len(resid) < 10:
        rep = CalibrationReport(
            model_name=forecaster.name, dim=dim, n_samples=len(resid),
            coverage_chi2=float("nan"), pit=tuple([float("nan")] * dim),
            sharpness=float("nan"), mean_bias=tuple([float("nan")] * dim),
            warnings=["insufficient steady-state samples for a calibration report"],
        )
        return rep

    reg = cov + np.eye(dim) * 1e-9
    sol = np.linalg.solve(reg, resid[:, :, None])[:, :, 0]
    mahal = np.einsum("nd,nd->n", resid, sol)
    coverage_chi2 = float(np.mean(mahal))

    # per-axis PIT via marginal predictive std
    std = np.sqrt(np.clip(np.einsum("nii->ni", cov), 1e-12, None))
    z = resid / std
    pit_vals = stats.norm.cdf(z)
    pit_p = tuple(float(stats.kstest(pit_vals[:, d], "uniform").pvalue) for d in range(dim))

    sign, logdet = np.linalg.slogdet(reg)
    sharpness = float(np.mean(0.5 * logdet))
    mean_bias = tuple(float(b) for b in resid.mean(axis=0))
    temperature = coverage_chi2 / max(dim, 1)

    warns: List[str] = []
    ratio = temperature
    if ratio > 1.5:
        warns.append(
            f"OVERCONFIDENT: mean Mahalanobis^2 = {coverage_chi2:.2f} >> D = {dim} "
            f"(covariances ~{ratio:.1f}x too small). REACT will (correctly) penalize this; "
            f"see calibration.counterfactual_react for the scalar-calibrated counterfactual."
        )
    elif ratio < 0.7:
        warns.append(
            f"UNDERCONFIDENT: mean Mahalanobis^2 = {coverage_chi2:.2f} < D = {dim} "
            f"(covariances ~{1/max(ratio,1e-6):.1f}x too large)."
        )
    for d, p in enumerate(pit_p):
        if p < 0.01:
            warns.append(f"axis {d}: PIT is non-uniform (KS p={p:.3g}); shape miscalibration.")
    for d, b in enumerate(mean_bias):
        if abs(b) > 0.5:
            warns.append(f"axis {d}: large mean bias ({b:+.2f}); possible unit/frame bug.")

    rep = CalibrationReport(
        model_name=forecaster.name, dim=dim, n_samples=len(resid),
        coverage_chi2=coverage_chi2, pit=pit_p, sharpness=sharpness,
        mean_bias=mean_bias, warnings=warns, temperature=temperature,
    )
    if emit_warning and not rep.is_calibrated:
        warnings.warn(
            f"[reactmetric] '{forecaster.name}' appears miscalibrated; REACT scores it "
            f"as delivered (not corrected). " + " ".join(warns),
            stacklevel=2,
        )
    return rep
