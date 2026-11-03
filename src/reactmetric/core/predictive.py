"""Predictive: a forecaster's per-track predictive distributions over the future.

The field's convention (ETH/UCY, Argoverse, nuScenes, trajdata, Trajectron++): the issue frame
is the last observed frame. A forecast issued at ``issue_frames[i]`` used observations up to
and including that frame, and ``means[i, h - 1]`` / ``covs[i, h - 1]`` describe the position at
frame ``issue_frames[i] + h`` (lead h = 1..H).

`gaussian` is the direct port (you supply means + covariances). `isotropic` is a
diagonal shortcut. `MixturePredictive` holds Gaussian-mixture forecasts, which REACT
scores natively. `from_samples` moment-matches K sampled trajectories to one Gaussian;
for a multimodal forecaster that collapse is costly, so pass its mixture instead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..scoring.nll import _gaussian_nll


class Predictive:
    """Gaussian predictive distributions for one track from one forecaster."""

    __slots__ = ("issue_frames", "means", "covs", "horizon")

    def __init__(
        self,
        issue_frames: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        horizon: Optional[int] = None,
    ):
        self.issue_frames = np.asarray(issue_frames, dtype=int)
        self.means = np.asarray(means, dtype=float)
        self.covs = np.asarray(covs, dtype=float)
        if self.means.ndim != 3:
            raise ValueError(f"means must be (n, H, D); got {self.means.shape}")
        if self.covs.ndim != 4:
            raise ValueError(f"covs must be (n, H, D, D); got {self.covs.shape}")
        self.horizon = int(horizon if horizon is not None else self.means.shape[1])

    # ---- constructors -----------------------------------------------------

    @classmethod
    def isotropic(
        cls, issue_frames: np.ndarray, means: np.ndarray, sigma
    ) -> Predictive:
        """Diagonal/isotropic shortcut.

        ``sigma`` (standard deviation) may be a scalar, a per-dim vector (D,), or an array
        of shape (n, H) or (n, H, D).
        """
        means = np.asarray(means, dtype=float)
        n, h, dim = means.shape
        sigma = np.asarray(sigma, dtype=float)
        var = np.empty((n, h, dim))
        if sigma.ndim == 0:
            var[...] = sigma**2
        elif sigma.shape == (dim,):
            var[...] = (sigma**2)[None, None, :]
        elif sigma.shape == (n, h):
            var[...] = (sigma**2)[:, :, None]
        elif sigma.shape == (n, h, dim):
            var[...] = sigma**2
        else:
            raise ValueError(f"sigma shape {sigma.shape} not broadcastable to {means.shape}")
        covs = np.zeros((n, h, dim, dim))
        idx = np.arange(dim)
        covs[:, :, idx, idx] = var
        return cls(issue_frames, means, covs)

    @classmethod
    def from_samples(
        cls,
        issue_frames: np.ndarray,
        samples: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Predictive:
        """One Gaussian per (issue, lead), moment-matched to samples (n, H, S, D).

        weights (n, S) or (n, H, S) optionally weight the samples (e.g. mode probabilities).
        This discards multimodality; MixturePredictive.from_samples keeps it."""
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 4:
            raise ValueError(f"samples must be (n, H, S, D); got {samples.shape}")
        n, h, s, dim = samples.shape
        w = _sample_weights(weights, n, h, s)
        means = np.einsum("nhs,nhsd->nhd", w, samples)
        diff = samples - means[:, :, None, :]
        covs = np.einsum("nhs,nhsd,nhse->nhde", w, diff, diff)
        return cls(issue_frames, means, covs)

    # ---- properties -------------------------------------------------------

    @property
    def D(self) -> int:
        return self.means.shape[2]

    def __len__(self) -> int:
        return len(self.issue_frames)

    def at_lead(self, lead: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(target_frames, means, covs) for the lead-h forecast: targets issue + h."""
        h = int(lead)
        if not 1 <= h <= self.horizon:
            raise ValueError(f"lead {h} outside horizon 1..{self.horizon}")
        target = self.issue_frames + h
        return target, self.means[:, h - 1], self.covs[:, h - 1]

    # ---- scoring ----------------------------------------------------------

    def nll_matrix(self, observations: np.ndarray) -> np.ndarray:
        """Per-(issue, step) Gaussian NLL of the realized observations, (n, H).

        NaN where the target frame is outside the track (before frame 0 or past its end)
        or the covariance is not positive-definite.
        """
        observations = np.asarray(observations, dtype=float)
        n, hh, dim = self.means.shape
        t = len(observations)
        targets = self.issue_frames[:, None] + np.arange(1, hh + 1)[None, :]
        valid = (targets >= 0) & (targets < t)
        targets_clipped = np.clip(targets, 0, t - 1)
        diff = observations[targets_clipped] - self.means              # (n, H, D)
        flat_resid = diff.reshape(-1, dim)
        flat_cov = self.covs.reshape(-1, dim, dim)
        nll = _gaussian_nll(flat_resid, flat_cov).reshape(n, hh)
        out = np.where(valid, nll, np.nan)
        return out


class MixturePredictive(Predictive):
    """Gaussian-mixture predictive distributions for one track (e.g. Trajectron++'s 25 modes).

    weights (n, J) or (n, H, J) (normalized per issue frame; mode weights shared across the
    horizon when 2D), component_means (n, H, J, D), component_covs (n, H, J, D, D). REACT
    scores the mixture itself (paper section 3.2); collapsing it to one Gaussian costs 0.4-0.8
    nats per frame on pedestrian data. The inherited ``means`` / ``covs`` / ``at_lead`` are the
    mixture's exact moment-matched Gaussian, for consumers that need one.
    """

    __slots__ = ("weights", "component_means", "component_covs")

    def __init__(
        self,
        issue_frames: np.ndarray,
        weights: np.ndarray,
        component_means: np.ndarray,
        component_covs: np.ndarray,
        horizon: Optional[int] = None,
    ):
        w = np.asarray(weights, dtype=float)
        mu = np.asarray(component_means, dtype=float)
        cv = np.asarray(component_covs, dtype=float)
        if mu.ndim != 4 or cv.ndim != 5:
            raise ValueError(
                f"component_means must be (n, H, J, D) and component_covs (n, H, J, D, D); "
                f"got {mu.shape}, {cv.shape}"
            )
        if w.shape not in (mu.shape[:1] + mu.shape[2:3], mu.shape[:3]):
            raise ValueError(f"weights must be (n, J) or (n, H, J); got {w.shape}")
        # moment matching through log weights, as the Trajectron++ adapter of the paper run did
        lw = np.exp(np.log(np.clip(w, 1e-300, None)))
        lw = lw / lw.sum(-1, keepdims=True)
        if lw.ndim == 2:
            mean = np.einsum("bk,bhkd->bhd", lw, mu)
            diff = mu - mean[:, :, None, :]
            cov = np.einsum("bk,bhkij->bhij", lw, cv) + np.einsum(
                "bk,bhki,bhkj->bhij", lw, diff, diff
            )
        else:
            mean = np.einsum("bhk,bhkd->bhd", lw, mu)
            diff = mu - mean[:, :, None, :]
            cov = np.einsum("bhk,bhkij->bhij", lw, cv) + np.einsum(
                "bhk,bhki,bhkj->bhij", lw, diff, diff
            )
        super().__init__(issue_frames, mean, 0.5 * (cov + np.swapaxes(cov, -1, -2)), horizon)
        self.weights, self.component_means, self.component_covs = w, mu, cv

    def mixture_at_lead(self, lead: int):
        """(target_frames, weights (n, J), means (n, J, D), covs (n, J, D, D)) at one lead."""
        h = int(lead)
        if not 1 <= h <= self.horizon:
            raise ValueError(f"lead {h} outside horizon 1..{self.horizon}")
        w = self.weights if self.weights.ndim == 2 else self.weights[:, h - 1]
        return (self.issue_frames + h, w, self.component_means[:, h - 1],
                self.component_covs[:, h - 1])

    @classmethod
    def from_samples(cls, issue_frames, samples, weights=None, bandwidth=None):
        """A kernel density over sampled trajectories (n, H, S, D): one isotropic Gaussian
        kernel per sample, so the forecast keeps its modes.

        bandwidth: the kernel's standard deviation (m), a scalar or one per lead (H,); None:
        Scott's rule per (issue, lead), sd * S^(-1/(D+4)). REACT scores the density, so the
        bandwidth matters: SampleForecaster.fit chooses it by log score on calibration
        tracks. weights (n, S) or (n, H, S) weight the samples (e.g. mode probabilities)."""
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 4:
            raise ValueError(f"samples must be (n, H, S, D); got {samples.shape}")
        n, h, s, dim = samples.shape
        w = _sample_weights(weights, n, h, s)
        if bandwidth is None:
            mean = np.einsum("nhs,nhsd->nhd", w, samples)
            var = np.einsum("nhs,nhsd->nh", w, (samples - mean[:, :, None]) ** 2) / dim
            bw = np.sqrt(var) * s ** (-1.0 / (dim + 4.0))
        else:
            bw = np.broadcast_to(np.asarray(bandwidth, float).reshape(-1)[None, :]
                                 if np.ndim(bandwidth) else float(bandwidth), (n, h))
        bw = np.maximum(bw, 1e-4)
        covs = (bw**2)[:, :, None, None, None] * np.eye(dim)
        return cls(issue_frames, w, samples, np.broadcast_to(covs, (n, h, s, dim, dim)).copy())

    @property
    def _mix(self):
        """The raw arrays as a dict (issue_frames, weights, means, covs)."""
        return {"issue_frames": self.issue_frames, "weights": self.weights,
                "means": self.component_means, "covs": self.component_covs}

    def nll_matrix(self, observations: np.ndarray) -> np.ndarray:
        """Per-(issue, step) mixture NLL of the realized observations, (n, H); NaN where the
        target frame is outside the track."""
        observations = np.asarray(observations, dtype=float)
        n, hh, J, dim = self.component_means.shape
        t = len(observations)
        targets = self.issue_frames[:, None] + np.arange(1, hh + 1)[None, :]
        valid = (targets >= 0) & (targets < t)
        y = observations[np.clip(targets, 0, t - 1)]                     # (n, H, D)
        diff = (y[:, :, None, :] - self.component_means).reshape(-1, dim)
        comp = -_gaussian_nll(diff, self.component_covs.reshape(-1, dim, dim)).reshape(n, hh, J)
        w = self.weights if self.weights.ndim == 3 else self.weights[:, None, :]
        w = w / w.sum(-1, keepdims=True)
        lp = comp + np.log(np.clip(w, 1e-300, None))
        peak = lp.max(-1, keepdims=True)
        ll = (peak + np.log(np.exp(lp - peak).sum(-1, keepdims=True)))[..., 0]
        return np.where(valid, -ll, np.nan)


def _sample_weights(weights, n, h, s):
    """Sample weights as (n, H, S), normalized per (issue, lead)."""
    if weights is None:
        return np.full((n, h, s), 1.0 / s)
    w = np.asarray(weights, dtype=float)
    if w.ndim == 2:
        w = np.broadcast_to(w[:, None, :], (n, h, s))
    return w / np.clip(w.sum(axis=2, keepdims=True), 1e-12, None)
