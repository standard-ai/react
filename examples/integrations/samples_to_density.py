"""A model that only outputs K sampled trajectories (no density).

REACT scores densities. This example compares the ways to turn samples into one:

1. ``rm.Predictive.from_samples``: moment-match the K samples at each lead to one Gaussian.
   Simple, but a multimodal sample cloud becomes one broad blob.
2. ``rm.SampleForecaster``: a kernel density, one Gaussian per sample, with the bandwidth
   chosen per lead by log score on the calibration tracks (``.fit(calib)``). Keeps the modes.
3. The same kernel density with Scott's rule of thumb for the bandwidth, for contrast.

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
            return None
        n, H, _ = p.means.shape
        L = np.linalg.cholesky(p.covs)                                   # (n, H, 2, 2)
        z = self.rng.standard_normal((n, H, self.k, 2))
        go = p.means[:, :, None] + np.einsum("nhij,nhkj->nhki", L, z)
        last = traj.positions[p.issue_frames][:, None, None, :]
        stop = last + 0.03 * self.rng.standard_normal((n, H, self.k, 2))
        stops = self.rng.random((n, 1, self.k, 1)) < 0.25
        return p.issue_frames, np.where(stops, stop, go)                # (n, H, K, 2)


class AsMoments:
    def __init__(self, model):
        self.model = model

    def predict(self, traj):
        out = self.model.sample(traj)
        return None if out is None else rm.Predictive.from_samples(*out)


def main():
    args = parse_args(__doc__)
    calib, test = load(args)
    scorer = fit_scorer(calib, args)
    kde = rm.SampleForecaster(StopOrGo().sample).fit(calib)
    print("fitted bandwidth per lead (cm):", np.round(100 * kde.bandwidth_, 1))
    models = {"StopOrGo (moments)": AsMoments(StopOrGo()),
              "StopOrGo (KDE, fitted)": kde,
              "StopOrGo (KDE, Scott)": rm.SampleForecaster(StopOrGo().sample, bandwidth=None),
              "ConstantVelocity": rm.baselines.ConstantVelocity.fit(calib)}
    result = scorer.evaluate(models, test, n_jobs=args.jobs)
    report(result, models)


if __name__ == "__main__":
    main()
