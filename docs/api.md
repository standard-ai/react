# API reference

The API is layered: a one-call high-level path, a composable mid-level, and the low-level primitives. `import reactmetric as rm`.

## Layer 0 — one call

- `rm.benchmark(forecasts, data, calibration, baselines=..., oracle=..., lead=..., epsilon=...)` → `BenchmarkReport`

## Layer 1 — core data model

- `rm.Trajectory(positions, dt=, id=, times=)` — `.T`, `.D`, `.positions`, `.velocity(window)`, `.from_dataframe(...)`
- `rm.TrajectorySet({id: traj})` — `.ids`, `.split(fraction, seed)`, `.from_dataframe(...)`, `.from_arrays(...)`
- `rm.Predictive.gaussian(issue_frames, means, covs)` / `.isotropic(...)` / `.from_samples(...)`
  - `.at_lead(h)`, `.nll_matrix(obs)`, `.nll(traj, scoring=)`
- `rm.Forecasts.from_dataframe(...)`, `.from_npz(...)`, `.to_npz(...)`

## Layer 2 — forecasters and baselines

- `rm.forecasters.from_callable(fn, name=)`, `from_precomputed(dict, name=)`
- `rm.baselines.ConstantVelocity(...)`, `CTRV(...)`, `IMM(...)`
- `rm.baselines.GRUPosition()`, `GRUOrientation()` — pretrained on a large
  retail pedestrian corpus (extras=[torch]; checkpoints ship with the package)
- `rm.baselines.available()`, `rm.baselines.build(name, **kwargs)`

## Layer 3 — reference / climatology

- `rm.reference.Climatology.fit(calib, leads=, mode="kinematic"|"global"|"position", k=)`
- `ref.nll_trace(traj, lead)`, `cv_nll_trace(...)`, `score_trace(traj, lead, scoring)`
- `ref.eps_star(lead, d_meters)`, `supported_distortion(lead)`, `displacement_prior(lead, speed, heading)`
- `rm.reference.save(ref, path)`, `rm.reference.load(path)`

## Layer 4 — events

- `rm.events.KinematicChangepoints(config=rm.events.SegmentationConfig(...))` — `.detect(traj)`, `.detect_set(ts)`, `.calibrate_penalty(...)`
- `rm.events.KinematicChangepoints.from_preset("high"|"medium"|"low"|"very-low")`
- `det.calibrate_events_per_track(calib, target_events_per_track=...)`, `det.calibrate_events_per_minute(calib, target_events_per_minute=...)`
- `rm.events.from_times({id: [t0, ...]})`
- `rm.events.RupturesDetector(...)` (extras=[cpd])

## Layer 5 — calibration

- `rm.calibration.report(...)` → `CalibrationReport` (`.coverage_chi2`, `.pit`, `.sharpness`, `.mean_bias`, `.is_calibrated`, `.summary()`)
- `rm.calibration.counterfactual_react(...)` — opt-in only

## Layer 6 — the REACT measurement

- `rm.evaluate(forecaster, data, ref, events, lead=, epsilon=, reference_kind="climatology"|"cv", scoring="nll"|"crps"|"energy", config=)` → `ReactResult`
- `ReactResult`: `.anticipation_rate`, `.failure_rate`, `.superseded_rate`, `.mean_react_uncensored`, `.km_curve(...)`, `.km_median(...)`, `.steady_state_mean_info`, `.by_event_type()`, `.bootstrap_ci(stat, n_boot)`, `.attach_excess(oracle_result)`, `.mean_excess_react`, `.measurements`, `.to_dict()`

## Layer 7 — oracles

- `rm.filter_artifacts(events, data, ref, oracle=rm.oracles.Hindsight(...))` → `(genuine, artifacts)`
- `rm.oracles.ClimatologySeeded(...)` (Oracle B; pass to `rm.evaluate`)
- `rm.oracles.oracle_c_react(react_b, detection_floor)`, `oracle_c_result(result_b, events)`

## Layer 8 — sweeps and accuracy

- `rm.lead_sweep(...)`, `rm.epsilon_sweep(...)`
- `rm.rate_distortion_anchors(ref, lead, distortion_targets_m)`, `rm.task_react(...)`
- `rm.forecast_accuracy(forecasters, data, leads)` — ADE/FDE/NLL by lead

## Layer 9 — benchmark, viz, datasets

- `rm.Benchmark(reference, events, config)` — `.add(...)`, `.add_baseline(...)`, `.add_oracle(...)`, `.run(data)`
- `BenchmarkReport`: `.table()`, `.wilcoxon()`, `.to_json(path)`, `.plot_km(path)`
- `rm.viz.km_recovery_curves(...)`, `info_score_histograms(...)`, `react_by_event_type(...)`, `excess_react_distribution(...)`, `event_detail(...)`, `three_part_summary(...)`
- `rm.datasets.load(path, mapping=...)`, `rm.datasets.load_dataset(name, ...)`
- `rm.datasets.prepare(...)` / `create_dataset_split(...)` → on-disk partition files
- `rm.datasets.mappings.eindhoven|tracks|ethucy`, `rm.datasets.specify_mapping(...)`
- `rm.datasets.DatasetMapping.from_yaml(...)` / `.to_yaml(...)`
- `rm.datasets.load_eindhoven_sample()`, `load_eindhoven(...)`, `download_eindhoven(...)`
- `rm.datasets.synthetic_regime_changes(...)`, `load_eth_ucy(...)`, `load_tracks(path=...)`, `prepare_tracks(...)`
