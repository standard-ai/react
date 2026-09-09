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

### 1. Per-step bivariate Gaussian `(mu_x, mu_y, sigma_x, sigma_y, rho)`

The Social-LSTM / Trajectron++ / Graves parameterization (the documented default). A long DataFrame with one row per `(track, issue, lead)`:

```python
forecasts = rm.Forecasts.from_dataframe(
    df, id="track_id", issue="issue_frame", lead="lead",
    mean=["mu_x", "mu_y"],
    cov=["sigma_x", "sigma_y", "rho"],     # or explicit ["cov_xx", "cov_xy", "cov_yy"]
)   # -> dict[id -> rm.Predictive]
```

Or directly from arrays:

```python
pred = rm.Predictive.gaussian(issue_frames, means, covs)  # means (n,H,D), covs (n,H,D,D)
pred = rm.Predictive.isotropic(issue_frames, means, sigma) # diagonal/isotropic shortcut
```

### 2. K trajectories + probabilities (Argoverse / Waymo / nuScenes submission style)

```python
pred = rm.Predictive.from_samples(issue_frames, samples,   # samples (n,H,S,D)
                                  weights=mode_probs,       # optional (n,H,S)
                                  method="moment")          # or "kde"
```

Samples are converted to a Gaussian predictive (moment matching, optionally KDE-smoothed) so multimodal/ensemble forecasts ride the same fast log-score path.

## Programmatic vs. CLI

- **Python**: `dict[id -> Predictive]` or a `rm.Forecasts` object.
- **CLI**: parquet (preferred, pandas-native) or an `.npz` bundle.

```bash
reactmetric evaluate --data tracks.parquet --forecasts my_preds.parquet \
    --calibration-fraction 0.2 --lead 10 --baselines CTRV IMM --oracle B \
    --reference-mode kinematic --out react_results.json --plots out/
```
