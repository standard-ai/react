"""Models that output sampled trajectories instead of a density.

REACT scores the forecast density at the realized position, and K sampled futures have no
density until one is chosen. SampleForecaster turns them into a kernel density (one isotropic
Gaussian per sample) whose bandwidth is fitted by log score on calibration tracks, the same
tracks REACT is fitted on, so every sample-based model is treated the same way:

    model = rm.SampleForecaster(my_sampler).fit(calib)
    result = scorer.evaluate({"MyGAN": model}, test)

my_sampler(traj) returns (issue_frames (n,), samples (n, H, K, 2)) or that plus weights
(n, K) or (n, H, K), or None for a track it cannot forecast.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.special import logsumexp

from .core.predictive import MixturePredictive, _sample_weights
from .forecasters.base import BaseForecaster

__all__ = ["SampleForecaster"]

# candidate kernel standard deviations (m): 1 mm to 3 m, geometric
BANDWIDTHS = np.geomspace(1e-3, 3.0, 49)


def _kde_logpdf(y, samples, logw, bw):
    """log density at y (n, D) of the kernel densities over samples (n, K, D), per bw."""
    d2 = ((samples - y[:, None, :]) ** 2).sum(-1)  # (n, K)
    dim = y.shape[1]
    bw = np.asarray(bw, float)[:, None, None]
    return logsumexp(logw[None] - d2[None] / (2 * bw**2), axis=-1) - dim * np.log(
        np.sqrt(2 * np.pi) * bw[:, :, 0])


class SampleForecaster(BaseForecaster):
    """A sample-based model scored as a kernel density over its samples.

    sampler: traj -> (issue_frames, samples (n, H, K, D)[, weights]) or None.
    bandwidth: "fit" (call fit() first), a kernel sd in metres (scalar or one per lead), or
        None for Scott's rule.
    """

    def __init__(self, sampler: Callable, bandwidth: Union[str, float, Sequence[float], None]
                 = "fit", name: Optional[str] = None):
        self.sampler = sampler
        self.bandwidth = bandwidth
        self.name = name or getattr(sampler, "__name__", "SampleForecaster")
        self.fit_curve_ = None

    def _draw(self, traj):
        out = self.sampler(traj)
        if out is None:
            return None
        issue, samples, *rest = out
        weights = rest[0] if rest else None
        return np.asarray(issue, int), np.asarray(samples, float), weights

    def fit(self, calib, events=None, calm_exclusion=(20, 60),
            grid: Sequence[float] = BANDWIDTHS):
        """Choose one bandwidth per lead by mean log score at the realized positions of the
        calibration tracks, on calm stretches (frames away from detected events, as REACT's
        own calibration). events: an event source (default BayesianChangepoints.tuned_sg()),
        or False to use every frame. Sets bandwidth_ (H,) and fit_curve_ (H, len(grid))."""
        from .events import BayesianChangepoints

        detector = None if events is False else (
            events if events is not None else BayesianChangepoints.tuned_sg())
        grid = np.asarray(grid, float)
        tot, cnt = None, None
        for tid in calib.ids:
            traj = calib[tid]
            drawn = self._draw(traj)
            if drawn is None:
                continue
            issue, samples, weights = drawn
            n, H, K, _ = samples.shape
            if tot is None:
                tot, cnt = np.zeros((H, len(grid))), np.zeros(H)
            obs = np.asarray(traj.positions, float)
            ev = detector.detect(traj) if detector is not None else []
            w = _sample_weights(weights, n, H, K)
            for h in range(min(H, tot.shape[0])):
                target = issue + h + 1
                ok = (target >= 0) & (target < len(obs)) & np.isfinite(samples[:, h]).all(
                    (1, 2))
                for e in ev:
                    ok &= ~((target >= e.t0 - calm_exclusion[0])
                            & (target <= e.t0 + calm_exclusion[1]))
                if not ok.any():
                    continue
                lp = _kde_logpdf(obs[target[ok]], samples[ok, h],
                                 np.log(np.clip(w[ok, h], 1e-300, None)), grid)
                tot[h] += lp.sum(1)
                cnt[h] += ok.sum()
        if tot is None or not cnt.any():
            raise ValueError("the sampler forecast no calibration frames to fit a bandwidth on")
        curve = tot / np.maximum(cnt, 1)[:, None]
        best = grid[np.argmax(curve, 1)]
        # leads without frames take the nearest fitted lead's bandwidth
        have = np.flatnonzero(cnt)
        best = best[have[np.abs(have[None, :] - np.arange(len(cnt))[:, None]).argmin(1)]]
        self.bandwidth_, self.fit_curve_, self.grid_ = best, curve, grid
        return self

    def predict(self, traj) -> Optional[MixturePredictive]:
        drawn = self._draw(traj)
        if drawn is None:
            return None
        issue, samples, weights = drawn
        if isinstance(self.bandwidth, str):
            if not hasattr(self, "bandwidth_"):
                raise RuntimeError("bandwidth='fit': call fit(calib) before predicting")
            bw = self.bandwidth_[: samples.shape[1]]
        else:
            bw = self.bandwidth
        return MixturePredictive.from_samples(issue, samples, weights, bandwidth=bw)

