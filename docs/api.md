# API overview

`import reactmetric as rm`. The stable API is what this page lists; the generated
[reference](reference.md) has every signature and docstring. Names not listed here (the scorer's
internals, most of `rm.information`) may change between releases.

## Scoring

- `rm.ReactScorer.fit(calib, info=(3.0, 4.2, 6.0), *, leads=(1, 5, 10, 15), sigma="estimate", distortions=(0.1, 0.3), events=None, p=0.95, eps=0.0, climatology=None, forecasts=None, window=150, min_window=25, onset_margin=5, relapse_window=10, relapse_run=3, calm_exclusion=(20, 60), pre_window=30, table_dir=None, n_jobs=1, seed=0)`: the only constructor.
  - `.evaluate(forecasts, data, *, model_leads=None, n_jobs=1, event_set="shared", progress=None)` → `ReactResult`
  - `.save(path)`, `rm.ReactScorer.load(path)`
  - `.climatology`, `.events`, `.sigma`, `.noise_floor`, `.levels[lead]`, `.diagnostics[lead]` (including `calm_mean_score`, the demonstrated information, when `forecasts` were given to `fit`)
- `rm.benchmark(forecasts, data, calib, baselines=("ConstantVelocity", "CTRV", "IMM"), info=None, leads=None, distortions=None, n_jobs=1, scorer=None, event_set="shared", **fit_kwargs)` → `ReactResult`: fit, add baselines, evaluate.

`forecasts` maps a name to a forecaster (anything with `predict(traj)` returning a `Predictive`, `MixturePredictive` or None), to `{track id: Predictive}`, or is a zero-argument callable returning that mapping (built once per worker). See [Bring your own forecaster](forecasters.md).

## Results

- `rm.ReactResult`: `.records`, `.models`, `.leads`, `.info_levels`, `.distortions`, `.settings()`, `.coverage`
  - `.table(lead=None, info="auto", distortion=None, model=None, by=None, columns="brief", ci=False)` (pandas)
  - `.summary(by=None)`, `.subset(where=None, **fields)`, `.to_dataframe()`
  - `.ci(stat, *, model, lead, ...)`, `.compare(model_a, model_b, stat, *, lead, ...)`
  - `.headline_info()`, `.info_level(distortion="rule", threshold=0.95, model="oracle_B")`
  - `.to_json(path)`, `.to_csv(path)`, `rm.ReactResult.load(path, settings=None)`
- `rm.select_info_level(summary, distortion=None, threshold=0.95)`
- `rm.results.GLOSSARY`: what every summary column means. See [Reading results](results.md).

## Data and forecasts

- `rm.Trajectory(positions, dt=, id=, times=)`, `rm.TrajectorySet({id: traj})`: `.ids`, `.dt`, `.split(fraction=0.3, seed=0)`, `.from_dataframe(...)`
- `rm.Predictive(issue_frames, means, covs)`: Gaussian forecasts, means `(n, H, D)`, covs `(n, H, D, D)`; `.isotropic(issue_frames, means, sigma)`, `.from_samples(issue_frames, samples)` (moment-matched), `.at_lead(h)`, `.nll_matrix(obs)`
- `rm.MixturePredictive(issue_frames, weights, component_means, component_covs)`: scored as a mixture; `.from_samples(issue_frames, samples, bandwidth=None)` (a kernel density), `.mixture_at_lead(h)`
- `rm.SampleForecaster(sampler, bandwidth="fit")`: `.fit(calib)` chooses the kernel bandwidth per lead
- `rm.Forecasts.from_dataframe(...)`, `.from_npz(path)`, `.to_npz(path, forecasts)`: see [I/O formats](io.md)

## Events

- `rm.events.BayesianChangepoints()`: the paper's detector (`.tuned_sg()` is the same); `rm.events.select(source, types=None, where=None)`; `rm.events.from_times({id: [t0, ...]})`; `rm.events.RupturesDetector(...)` (extras=[cpd]); `rm.events.Event`. See [Choosing events](events.md).

## Baselines

- `rm.baselines.ConstantVelocity()` (`.fit(calib)`: maximum likelihood), `CTRV()`, `IMM()` (`.fit(calib)`)
- `rm.baselines.GRUPosition()`, `GRUOrientation()`: trained on Eindhoven (extras=[torch])
- `rm.baselines.build(name, **kwargs)`, `rm.baselines.available()`

## Experimental

Public but subject to change: `rm.reference.fit_climatology`, `save_climatology`, `load_climatology`; `rm.information.distortion_margin`, `estimate_noise_floor`, `info_to_c`; `rm.events.fit_hazard`; `rm.kaplan_meier`, `rm.km_median`; `ReactResult.timing`; `rm.viz.km_curves`, `outcome_profile`, `react_label` (extras=[viz]); `rm.datasets` (see [Datasets](datasets.md)).

## Command line

```bash
reactmetric evaluate --data tracks.parquet --forecasts preds.parquet --name MyModel \
    --info 3 4.2 6 --leads 5 10 --distortions 0.1 --out result.json --records records.csv
```
