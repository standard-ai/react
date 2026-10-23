# Changelog

## 0.2.0 (unreleased)

REACT as published in Breseman and Woollard, *REACT: Measuring Regime Adaptation in
Trajectories* (TGF 2026). The library reproduces the paper's per-event results exactly
(`tests/parity/`). 0.1.0 was never released; its metric is removed.

- `ReactScorer.fit(calib, info=...)` / `.evaluate(forecasts, test)` → `ReactResult`: onset,
  recovery, censoring and relapse per event, lead, information level and target distance, with
  Oracle B; Kaplan-Meier summaries over all events and given onset, restricted means.
- `reactmetric.information`: the truth posterior with a measurement-noise floor, exact regime
  tests against the two-regime climatology, overlap tables, native mixture scoring, distortion
  margins, and the paper's information-level rule (`select_info_level`).
- `MixturePredictive`: Gaussian-mixture forecasts, scored natively.
- `events.BayesianChangepoints.tuned_sg()`: the paper's event detector.
- `reference.fit_climatology`: the paper's two-regime climatology.
- Issue frames now follow the field's convention: the issue frame is the last observed frame,
  and lead h targets issue + h (was issue + h - 1).
- `benchmark`, the CLI and `viz` rewired onto the scorer.
- Removed: the three-consecutive-frames rule and `evaluate`, `ReactConfig`, the old oracles,
  CRPS / energy scoring, sweeps and the non-standard `forecast_accuracy`, the calibration
  report and counterfactual REACT, the global and position references, the PELT detector.
  scikit-learn is no longer a dependency.

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
