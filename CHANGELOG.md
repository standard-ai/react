# Changelog

## 0.1.0 (unreleased)

Initial public release of `reactmetric`.

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
