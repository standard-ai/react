# Calibration: call out, never correct

REACT is built on the log score, a **strictly proper** scoring rule. That makes it calibration-sensitive *by design*: a forecast that fails to recover utility because its uncertainty is wrong has genuinely failed, and REACT should report that — not erase it.

So `reactmetric` **diagnoses** calibration loudly but **never modifies** your forecast.

## The report

```python
rep = rm.calibration.report(forecaster_or_predictives, data, ref, events=events)
rep.coverage_chi2     # mean squared Mahalanobis; == D when calibrated, > D overconfident
rep.pit               # per-axis PIT uniformity (KS p-values)
rep.sharpness         # mean predictive log-volume (only meaningful with calibration)
rep.mean_bias         # per-axis mean residual (catches unit/frame bugs)
rep.is_calibrated
print(rep.summary())
```

Computed on the **steady-state** (event-masked) frames of the calibration split, where forecasts should be calibrated if they are calibrated anywhere. A materially miscalibrated model raises a `UserWarning` (so it surfaces in benchmark runs).

## Why not just rescale?

Two reasons we refuse to auto-correct:

1. Rescaling covariances scores a **different, hypothetical forecast** — not the one you deliver.
2. A steady-state scalar fix does **not even guarantee** the model clears the event-region recovery threshold, so a "calibrated" model can still fail REACT.

## The opt-in counterfactual

If you *do* want to ask "what would REACT be if my covariances were scalar-calibrated?", that is an explicit, clearly-labeled analysis — never the headline number:

```python
cf = rm.calibration.counterfactual_react(predictives, data, ref, events, lead=10)
print(cf.summary())   # delivered vs. T = mean_mahalanobis/D recalibrated; marked HYPOTHETICAL
```

## Escape hatch

If you cannot supply trustworthy covariances at all, use the calibration-robust scoring variants:

```python
rm.evaluate(..., scoring="crps")     # or scoring="energy"
```
