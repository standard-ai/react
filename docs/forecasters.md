# Bring your own forecaster

REACT scores a forecast's density at the position the person actually reached, at every frame of
every test track. Your model needs to provide that density in one of four forms. This page shows
each in a few lines; [I/O formats](io.md) covers files and the command line.

## The layout

For one track, a forecast is a set of **issue frames** and, for each, a distribution for each
**lead** `h = 1 … H`:

- the issue frame is the index of the last observation the forecast saw;
- lead `h` is the forecast for frame `issue + h`, so row `means[i, h - 1]` targets frame
  `issue_frames[i] + h`;
- `H` must reach the largest lead you score (the default leads are 1, 5, 10 and 15);
- positions are in the same units as the tracks (metres).

Issue a forecast at every frame you can: REACT follows usefulness frame by frame through each
event window. Frames a model does not forecast (too little history, or NaN values) are treated
as missing, and by default models are compared only on events every one of them covers.

## Gaussian forecasts

The common case. Anything with a `predict(traj)` method that returns an `rm.Predictive` (or
None for a track it cannot forecast) is a forecaster:

```python
import numpy as np
import reactmetric as rm

class Linear:
    """Constant velocity from the last two observations, with a spread growing with lead."""

    def __init__(self, horizon=15, sigma=0.05):
        self.horizon, self.sigma = horizon, sigma

    def predict(self, traj):
        x = np.asarray(traj.positions)
        issue = np.arange(1, len(x))                      # last observed frame
        v = x[issue] - x[issue - 1]                       # per-frame velocity
        h = np.arange(1, self.horizon + 1)
        means = x[issue][:, None] + v[:, None] * h[None, :, None]       # (n, H, 2)
        sd = self.sigma * np.sqrt(h)                                     # (H,)
        return rm.Predictive.isotropic(issue, means, np.broadcast_to(sd, means.shape[:2]))
```

`rm.Predictive(issue_frames, means, covs)` takes full covariances `(n, H, 2, 2)`;
`rm.Predictive.isotropic` takes standard deviations.

## Mixtures

Multimodal models (Trajectron++, mixture density networks) should pass the mixture itself;
REACT scores it as a mixture. Collapsing it to one Gaussian costs 0.4–0.8 nats per frame on
pedestrian data, which is about the size of the effects REACT measures. For your own accuracy
tables, `pred.nll_matrix(track.positions)` gives the mixture's negative log likelihood per
(issue frame, lead).

```python
rm.MixturePredictive(issue_frames, weights, component_means, component_covs)
# weights (n, J) or (n, H, J); component_means (n, H, J, 2); component_covs (n, H, J, 2, 2)
```

## Sampled trajectories

Models that output K futures with no density (GANs, diffusion models, many motion-prediction
networks) need one chosen for them. `rm.SampleForecaster` places a small Gaussian kernel on each
sample and picks the kernel width per lead by log score on your calibration tracks, so every
sample-based model is treated the same way:

```python
def sampler(traj):
    issue_frames, samples = my_model.sample(traj)        # samples (n, H, K, 2)
    return issue_frames, samples                         # optionally a third item: weights

model = rm.SampleForecaster(sampler, name="MyGAN").fit(calib)
print(model.bandwidth_)                                  # kernel sd per lead, metres
```

With only a handful of samples the density is rough whatever the bandwidth, so more samples give
a fairer score. `rm.Predictive.from_samples(issue_frames, samples)` moment-matches the samples to
one Gaussian instead; it is simpler but discards the modes.

## Precomputed forecasts

If forecasts already exist, pass `{track id: Predictive}` in place of a forecaster. The ids must
match the test tracks' ids. To read them from a table, see [I/O formats](io.md).

## Running

```python
scorer = rm.ReactScorer.fit(calib)
result = scorer.evaluate({"Mine": Linear(), "ConstantVelocity": rm.baselines.ConstantVelocity.fit(calib)},
                         test, n_jobs=8)
```

With `n_jobs > 1` the work runs in separate processes, which re-import your script on macOS and
Windows: put the calls under `if __name__ == "__main__":`. A model that is slow to build or
cannot be pickled can be passed as a zero-argument function returning the mapping; each worker
builds it once.

If a model's `predict` raises, that track is skipped for that model and you get one warning
naming the error. A model whose scores sit far below the climatology everywhere also triggers a
warning: it usually means covariances in the wrong units or forecasts off by a frame.
