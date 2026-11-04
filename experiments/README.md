# Experiments

Scripts behind the paper's results, run against this package. They are not tests and not part
of the installed package. The paper's full reproduction, with its data, trained models and
outputs, is [react-reproducibility](https://github.com/calvin-breseman/react-reproducibility).

- `exp10_information_react.py`: the paper's REACT run on Eindhoven or Day in the Life: fits
  the scorer, scores the kinematic baselines, the GRUs and Trajectron++, and writes per-event
  records and a summary. `tests/parity/` reruns it on a few tracks and checks every row against
  the paper's run.
- `exp11_react_vs_calibration.py`: the covariance-rescaling analysis (paper section 4.4).
- `realized_behind.py`: how often a forecast is behind the climatology on realized scores.
- `trajectron_forecaster.py`: the Trajectron++ adapter (needs a patched Trajectron++ clone; see
  react-reproducibility's `third_party/`).
- `day_in_the_life.py`: a loader for the Day in the Life dataset.

Run them from this directory:

```bash
cd experiments
python exp10_information_react.py --help
```
