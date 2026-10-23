# API reference

`import reactmetric as rm`. Three layers: one call, the scorer, and the pieces it is built from.

## One call

- `rm.benchmark(forecasts, data, calibration, baselines=("ConstantVelocity", "CTRV", "IMM"), info=(3.0, 4.2, 6.0), leads=(1, 5, 10, 15), distortions=(0.1, 0.3), n_jobs=1, **fit_kwargs)` → `ReactResult`

## The scorer

- `rm.ReactScorer.fit(calib, info=(3.0, 4.2, 6.0), *, leads, sigma="estimate", distortions=(0.1, 0.3), events=None, p=0.95, lead_info=None, climatology=None, forecasts=None, model_leads=None, table_dir=None, n_jobs=1, seed=0)`
  - `.evaluate(forecasts, data, *, model_leads=None, n_jobs=1)` → `ReactResult`
  - `.score_track(traj, predictives, events=None)` → per-event records for one track
  - `.levels[lead]` (information levels with fitted correlations), `.sigma`, `.noise_floor`, `.calibration[lead]` (including `calm_mean_score`, the demonstrated information, when `forecasts` were given)
- `rm.ReactResult`
  - `.records`: one dict per (lead, information level, distortion, model, event): `status` (`recovered` / `censored` / `better` / `indistinguishable`), `onset`, `react`, `duration`, `relapse`, `relapse3`, event descriptors, realized scores
  - `.summary()` → `{(lead, info, distortion, model): profile}` with `anticipation`, `recovered`, `censored`, `km_react` (median given onset), `km_react_overall`, `rmst`, `rmst_cond`, `relapse`, `relapse3`, `mean_I`, …
  - `.table(lead=, info=, distortion=)` (pandas), `.to_dataframe()`, `.to_json(path)`, `.to_csv(path)`, `.settings()`, `.info_level(distortion)`
- `rm.select_info_level(summary, distortion, threshold=0.95)`: the paper's rule for the headline level

`forecasts` is a mapping name → forecaster (anything with `predict(traj) -> Predictive`) or `{track id: Predictive}`, or a zero-argument callable returning that mapping (built once per worker process).

## Data

- `rm.Trajectory(positions, dt=, id=, times=)`, `rm.TrajectorySet({id: traj})`: `.ids`, `.split(fraction, seed)`, `.from_dataframe(...)`, `.from_arrays(...)`
- `rm.Predictive(issue_frames, means, covs)`: Gaussian forecasts, means (n, H, D), covs (n, H, D, D). The issue frame is the last observed frame; lead h targets issue + h. `.gaussian(...)`, `.isotropic(...)`, `.from_samples(...)` (moment-matched), `.at_lead(h)`, `.nll_matrix(obs)`
- `rm.MixturePredictive(issue_frames, weights, means, covs)`: Gaussian-mixture forecasts, scored natively. `.mixture_at_lead(h)`; its `means`/`covs` are the moment-matched Gaussian
- `rm.Forecasts.from_dataframe(...)`, `.from_npz(...)`, `.to_npz(...)`: see [I/O formats](io.md)

## Baselines

- `rm.baselines.ConstantVelocity(...)` (`.fit(calib)`: maximum likelihood), `CTRV(...)`, `IMM(...)` (`.fit(calib, bounds=...)`)
- `rm.baselines.GRUPosition()`, `GRUOrientation()` (extras=[torch])
- `rm.baselines.build(name, **kwargs)`, `rm.baselines.available()`

## Pieces

- `rm.events.BayesianChangepoints.tuned_sg()`: the paper's detector; `rm.events.from_times({id: [t0, ...]})`; `rm.events.RupturesDetector(...)` (extras=[cpd])
- `rm.reference.fit_climatology(calib, leads)`: the two-regime climatology; `.mixture_frames(traj, lead)`, `.gaussian_frames(...)`, `.nll_trace(traj, lead)`; `rm.reference.save` / `load`
- `rm.information`: `truth_posterior`, `RegimeTests`, `RegimeTestsMixture`, `OverlapTable`, `regime_predictive` (Oracle B), `classify_lazy`, `distortion_margin`, `estimate_noise_floor`
- `rm.kaplan_meier`, `rm.km_median`, `rm.cluster_bootstrap_ci`
- `rm.viz.km_curves(result, lead, info, distortion=0.0, given_onset=True)`, `rm.viz.outcome_profile(...)`, `rm.viz.react_label(lead, d, info)` (extras=[viz])
- `rm.datasets`: see [Datasets](datasets.md)

## Command line

```bash
reactmetric evaluate --data tracks.parquet --forecasts preds.parquet \
    --info 3 4.2 6 --leads 5 10 --distortions 0.1 --out summary.json --records records.csv
```
