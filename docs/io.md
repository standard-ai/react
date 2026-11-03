# I/O formats and interoperability

`reactmetric` mirrors established trajectory-forecasting conventions rather than inventing its own schema.

## Input trajectories — Argoverse 2 tabular schema

A tidy long DataFrame / parquet with `track_id`, a time column, and one column per coordinate:

```python
data = rm.TrajectorySet.from_dataframe(
    df, id="track_id", time="timestep",
    coords=["position_x", "position_y"],   # add "position_z" for 3D
    dt=0.1,
)
```

This matches Argoverse 2 / `trajdata`-style inputs, so existing dataset loaders point straight at us.

## Forecasts — the two formats people actually emit

### Issue frames

The issue frame is the **last observed frame**, the convention of ETH/UCY, Argoverse, nuScenes, trajdata and Trajectron++. A forecast issued at frame `t` may use observations up to and including `t`, and its lead-`h` row predicts the position at frame `t + h`. REACT checks nothing about causality for you: a forecast that peeks at frame `t + 1` will look better than it is.

REACT needs a forecast issued at (nearly) every frame of each track, not one per scenario: it follows usefulness frame by frame through each event. Tracks must be long enough to hold event windows (at least 25 observations between events by default); short benchmark snippets (8 observed + 12 predicted frames) are too short.

### 1. Per-step bivariate Gaussian `(mu_x, mu_y, sigma_x, sigma_y, rho)`

The Social-LSTM / Trajectron++ / Graves parameterization (the documented default). A long DataFrame with one row per `(track, issue, lead)`:

```python
forecasts = rm.Forecasts.from_dataframe(
    df, id="track_id", issue="issue_frame", lead="lead",
    mean=["mu_x", "mu_y"],
    cov=["sigma_x", "sigma_y", "rho"],     # or explicit ["cov_xx", "cov_xy", "cov_yy"]
)   # -> dict[id -> rm.Predictive]
```

Leads are read by value: a file with leads 1, 5, 10 and 15 gives forecasts at those leads and
none in between. Or directly from arrays:

```python
pred = rm.Predictive(issue_frames, means, covs)           # means (n,H,D), covs (n,H,D,D)
pred = rm.Predictive.isotropic(issue_frames, means, sigma) # diagonal/isotropic shortcut
```

### 2. Gaussian mixtures (Trajectron++ and other multimodal models)

```python
pred = rm.MixturePredictive(issue_frames, weights, component_means, component_covs)
# weights (n, J) or (n, H, J); component_means (n, H, J, 2); component_covs (n, H, J, 2, 2)
```

REACT scores mixtures natively. Collapsing a mixture to one Gaussian costs 0.4–0.8 nats per frame on pedestrian data, comparable to the whole signal, so pass the mixture when you have it.

### 3. K trajectories without a density

REACT is defined on densities, so samples need one chosen for them. `rm.SampleForecaster` puts
a Gaussian kernel on each sample with a bandwidth fitted on the calibration tracks (see
[Bring your own forecaster](forecasters.md#sampled-trajectories)); from arrays:

```python
pred = rm.MixturePredictive.from_samples(issue_frames, samples,    # samples (n,H,S,D)
                                         bandwidth=0.08)            # metres, or one per lead
pred = rm.Predictive.from_samples(issue_frames, samples)          # one moment-matched Gaussian
```

## Programmatic vs. CLI

- **Python**: `dict[id -> Predictive]`, as `rm.Forecasts.from_dataframe` returns.
- **CLI**: parquet (preferred), CSV, or an `.npz` bundle from `rm.Forecasts.to_npz`, which keeps
  mixtures and the type of the track ids.

```bash
reactmetric evaluate --data tracks.parquet --forecasts my_preds.parquet --name MyModel \
    --info 3 4.2 6 --leads 5 10 --distortions 0.1 \
    --out result.json --records records.csv --plots out/
```

It prints each model's REACT profile, writes the result (reload it with
`rm.ReactResult.load("result.json")`), and makes its own calibration split unless you pass
`--calibration calib.parquet`. `reactmetric evaluate --help` lists every option with its default.
```
