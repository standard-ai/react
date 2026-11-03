# Changelog

## 0.2.0 (unreleased)

REACT as published in Breseman and Woollard, *REACT: Measuring Regime Adaptation in
Trajectories* (Collective Dynamics, TGF 2026). The library reproduces the paper's per-event
results exactly (`tests/parity/`). 0.1.0 was never released; its metric is removed.

Scoring
- `ReactScorer.fit(calib, info=...)` / `.evaluate(forecasts, test)` → `ReactResult`: onset,
  recovery, censoring and relapse per event, lead, information level and target distance, with
  Oracle B. `ReactScorer.save` / `load`.
- Windows are settings counted in observations (`window`, `min_window`, `onset_margin`,
  `relapse_window`, `relapse_run`, `calm_exclusion`, `pre_window`); the defaults are the
  paper's at 10 Hz. Data at other frame rates work; see docs/concepts.md.
- `evaluate(event_set="shared")` by default compares models on the events they all cover and
  warns when they disagree; `"all"` keeps each model's own events, as the paper did.
- Overlap tables are named by a fingerprint of what they were built from and reused when it
  matches. Non-finite forecasts count as missing frames; a model whose `predict` raises, a
  degenerate model and a missing noise floor each warn instead of failing silently.
- `reactmetric.information`: the truth posterior with a measurement-noise floor, exact regime
  tests against the two-regime climatology, overlap tables, native mixture scoring.
- `events.BayesianChangepoints()` defaults to the paper's detector (`tuned_sg()` is the same);
  `events.select` scores only some event types.
- `reference.fit_climatology`: the paper's two-regime climatology.

Results
- `ReactResult.table()`: a brief REACT profile (`anticipated`, `recovered`, `censored`,
  `median_react`, `rmst`, `relapsed`; `results.GLOSSARY` defines every column), the headline
  information level chosen by the paper's rule, `by="event_type"`, filters that raise when
  they match nothing. d = ∞ is `distortion=None` (0, inf and "clim" accepted).
- `ci()` and `compare()`: track-level bootstrap intervals and paired model differences.
- `to_json()` includes the records; `ReactResult.load()` reads JSON or CSV back. `subset()`.

Forecasts and data
- `MixturePredictive(issue_frames, weights, component_means, component_covs)`, scored natively;
  `MixturePredictive.from_samples` and `SampleForecaster` turn sampled trajectories into a
  kernel density with a fitted bandwidth.
- Issue frames follow the field's convention: the issue frame is the last observed frame, and
  lead h targets issue + h (was issue + h - 1).
- `Forecasts.from_dataframe` reads leads by value; `.npz` files keep mixtures and id types.
- The GRU baselines are retrained on Eindhoven (CC-BY-4.0) and load with
  `weights_only=True`; the bundled sample is 40 real Eindhoven tracks.
- `load_eindhoven` raises instead of falling back to the sample, shows download progress, and
  `max_tracks` filters before capping (with `seed=`). `TrajectorySet.split` defaults to 0.3.
- CTRV and IMM require a history of at least 4 observations.

Interface
- `benchmark(forecasts, data, calib, ...)` refuses settings that conflict with a passed
  scorer. The CLI gains `--name`, `--calibration`, `--window`, `--event-set`, a printed
  summary and defaults in `--help`.
- Python 3.10 or newer; licence metadata as `License-Expression: Apache-2.0` with a NOTICE.

Removed or renamed
- `ReactScorer.ref`, `.detector`, `.calibration` → `.climatology`, `.events`, `.diagnostics`.
- `Predictive.gaussian` (use `Predictive(...)`), `Predictive.nll` (use `nll_matrix`),
  `Predictive.from_samples(method="kde")`, `MixturePredictive.mix`.
- `metric.cluster_bootstrap_ci` (use `ReactResult.ci`); `reference.save` / `load` →
  `save_climatology` / `load_climatology`; the `GRU` alias (use `GRUPosition`);
  `datasets.load_dataset("eindhoven-sample")` (use `load_eindhoven_sample()`);
  `io.forecasts_from_dataframe` (use `Forecasts.from_dataframe`); `constants.EVENT_TYPES`;
  the `scoring` package's public names; `baselines.CalibratedNoise` and
  `default_checkpoint_path` are private.
- From 0.1.0: the three-consecutive-frames rule and `evaluate`, `ReactConfig`, the old oracles,
  CRPS / energy scoring, sweeps and `forecast_accuracy`, the calibration report and
  counterfactual REACT, the global and position references, the PELT detector.
  scikit-learn and huggingface-hub are no longer dependencies.

## 0.1.0 (not released)

Pre-paper version, superseded by 0.2.0.

- Core ND data model: `Trajectory`, `TrajectorySet`, `Predictive` (gaussian / isotropic / from_samples).
- REACT measurement: onset/recovery, Kaplan-Meier survival, cluster-bootstrap CIs, by-event-type, steady-state informativeness, excess-REACT.
- Reference climatology: `global`, `kinematic` (2D speed/heading-conditioned), `position` modes with the robust floor component; generalized to N dimensions.
- Calibration diagnostics (coverage χ², PIT, sharpness, mean bias) — call out, never correct — with an opt-in counterfactual recalibration.
- Pluggable event detection: built-in kinematic changepoint (PELT) detector, `ruptures` adapter, and bring-your-own timestamps.
- Calibration-robust scoring: CRPS / energy-score REACT variants.
- Oracle ladder: Hindsight (artifact filter), ClimatologySeeded (Oracle B floor), Oracle C composition.
- Baselines: ConstantVelocity, CTRV, IMM, plus pretrained `GRUPosition` /
  `GRUOrientation` (extras=[torch]; checkpoints ship in-package).
- Sweeps and anchors: lead sweep, epsilon sweep, rate-distortion anchors, task-REACT, and an ADE/FDE accuracy summary.
- `Benchmark` / `BenchmarkReport` with a pandas table, Wilcoxon comparisons, JSON serialization, and a plotting suite.
- Dataset adapters: synthetic regime-change generator, ETH/UCY, and Standard Cognition Tracks (gated).
- CLI (`reactmetric evaluate`), pandas/parquet/npz I/O mirroring Argoverse 2 and bivariate-Gaussian conventions.
