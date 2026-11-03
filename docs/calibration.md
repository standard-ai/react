# Calibration

REACT is built on the log score, a strictly proper scoring rule, so it scores a forecast as delivered: a forecast whose uncertainty is wrong pays for it, and REACT reports that rather than correcting it. In the paper's known-truth simulation, REACT confirms recovery for an overconfident version of the oracle on only 14 of 734 events: overconfidence is penalized even when the forecast centre is right.

To separate skill from calibration, the paper rescales each forecaster's covariance by its best-fitting factor $q^*$ on the calibration tracks and repeats the evaluation across scales around it (section 4.4, Fig. 9). Rescaling trades anticipation against completeness of recovery, but does not remove the differences in recovery timing between model families. That analysis is `experiments/exp11_react_vs_calibration.py`; it scores a hypothetical forecast, so keep it apart from the headline numbers.

Practical checks before reading REACT numbers:

- **Units.** Positions in metres, covariances in square metres. Unit errors look like extreme miscalibration, and REACT warns when a model's scores sit far below the climatology.
- **Indexing.** The issue frame is the last observed frame and lead h targets issue + h ([I/O formats](io.md)). An off-by-one forecast scores as a lagging one.
- **Demonstrated information.** Pass your models to `ReactScorer.fit(..., forecasts=...)`; `scorer.diagnostics[lead]["calm_mean_score"]` is each model's mean realized gain over climatology on calm stretches. A model far below the others there is likely miscalibrated or misaligned.
