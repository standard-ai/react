"""A model that only outputs K sampled trajectories (no density).

REACT scores densities. This example shows the two ways to turn samples into one:

1. ``rm.Predictive.from_samples``: moment-match the K samples at each lead to one Gaussian.
   Simple, but a multimodal sample cloud becomes one broad blob.
2. A kernel density: each sample becomes a component of an ``rm.MixturePredictive`` with a
   small covariance (the bandwidth). Keeps the modes; the bandwidth is your assumption.

The stand-in sampler below is constant velocity with noise, plus a 25% chance of stopping, so
its samples are genuinely bimodal.

    python examples/integrations/samples_to_density.py            # 400 Eindhoven tracks
    python examples/integrations/samples_to_density.py --sample   # bundled sample
"""

from __future__ import annotations

import numpy as np
from _common import fit_scorer, load, parse_args, report

import reactmetric as rm


class StopOrGo:
    """Stand-in sampling model: K trajectories per issue frame, ~25% of them stopping."""

    def __init__(self, k: int = 32, seed: int = 0):
        self.k, self.rng = k, np.random.default_rng(seed)
        self.cv = rm.baselines.ConstantVelocity()

    def sample(self, traj):
        p = self.cv.predict(traj)
        if p is None:
            return None, None
        n, H, _ = p.means.shape
        L = np.linalg.cholesky(p.covs)                                   # (n, H, 2, 2)
        z = self.rng.standard_normal((n, H, self.k, 2))
        go = p.means[:, :, None] + np.einsum("nhij,nhkj->nhki", L, z)
        last = traj.positions[p.issue_frames][:, None, None, :]
        stop = last + 0.03 * self.rng.standard_normal((n, H, self.k, 2))
        stops = self.rng.random((n, 1, self.k, 1)) < 0.25
        return p.issue_frames, np.where(stops, stop, go)                # (n, H, K, 2)


def samples_to_mixture(issue_frames, samples, bandwidth_scale=1.0, floor=0.02):
    """Gaussian kernel density over K samples: one equal-weight component per sample, with
    a Scott's-rule bandwidth from the sample spread (2D: K^(-1/6)) and a floor in metres."""
    n, H, K, _ = samples.shape
    centred = samples - samples.mean(2, keepdims=True)
    cov = np.einsum("nhki,nhkj->nhij", centred, centred) / (K - 1)
    h2 = (bandwidth_scale * K ** (-1.0 / 6.0)) ** 2
    bw = h2 * cov + floor**2 * np.eye(2)
    covs = np.broadcast_to(bw[:, :, None], (n, H, K, 2, 2))
    return rm.MixturePredictive(issue_frames, np.full((n, K), 1.0 / K), samples, covs)


class AsMoments:
    def __init__(self, model):
        self.model = model

    def predict(self, traj):
        issue, s = self.model.sample(traj)
        return None if s is None else rm.Predictive.from_samples(issue, s)


class AsKDE:
    def __init__(self, model):
        self.model = model

    def predict(self, traj):
        issue, s = self.model.sample(traj)
        return None if s is None else samples_to_mixture(issue, s)


def main():
    args = parse_args(__doc__)
    calib, test = load(args)
    scorer = fit_scorer(calib, args)
    models = {"StopOrGo (moments)": AsMoments(StopOrGo()), "StopOrGo (KDE)": AsKDE(StopOrGo()),
              "ConstantVelocity": rm.baselines.ConstantVelocity.fit(calib)}
    result = scorer.evaluate(models, test, n_jobs=args.jobs)
    report(result, models)


if __name__ == "__main__":
    main()
