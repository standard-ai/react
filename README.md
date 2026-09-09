# reactmetric

**REACT (Regime Event Adaptive Correction Time)** measures *how quickly a trajectory forecast becomes genuinely useful again after a disruptive regime change* — a turn, a stop, a sudden acceleration. It complements accuracy metrics like ADE/FDE (which average over time and so wash out exactly the moments that matter) with an **adaptation-speed** axis.

> ADE/FDE tell you how good a forecast is *on average*. REACT tells you how fast it *recovers* when the world changes.

## The idea in one line

For a forecaster `m`, define the per-frame **information score**

```
s_m(f) = nll_reference(f) − nll_model(f)      (nats)
```

the pointwise log Bayes factor between the model and a *climatological reference* ("what you'd expect knowing nothing about this person beyond population statistics"). After a regime change at `t0`:

- **onset**: `s < 0` sustained — the forecast has fallen below the reference,
- **recovery**: `s ≥ ε` sustained — it beats the reference by the task margin again,

and `REACT = (recovery frame) − t0`. Recovery times are right-censored, so the headline summary is a **Kaplan-Meier** survival curve over events, with cluster-bootstrap confidence intervals.

## Install

```bash
pip install reactmetric                 # core
pip install "reactmetric[all]"          # + viz, datasets, ruptures
pip install "reactmetric[torch]"        # + the GRU example baseline
```

## Quickstart

The runnable quickstart defaults to a tiny Eindhoven-format pedestrian trajectory
sample and writes a JSON report plus plots:

```bash
python examples/quickstart.py
```

For the full Eindhoven Centraal Zenodo dataset (CC-BY-4.0, DOI
`10.5281/zenodo.13784588`), download/cache one 10-day parquet shard explicitly:

```bash
python examples/quickstart.py --dataset eindhoven --download --days 01_10
```

Programmatic usage looks like:

```python
import reactmetric as rm

# 1. Bring your trajectories.
data = rm.datasets.load_eindhoven_sample()
calib, test = data.split(fraction=0.2, seed=0)

# 2. ... and your model's predictive distributions.
my_predictives = {tid: rm.Predictive.gaussian(issue_frames, means, covs) for tid in test.ids}

# 3. Benchmark against baselines + an oracle recovery floor.
report = rm.benchmark(
    forecasts={"MyModel": my_predictives},
    data=test, calibration=calib,
    baselines=["CTRV", "IMM", "ConstantVelocity", "GRUPosition", "GRUOrientation"],
    oracle="B", lead=10, epsilon=0.0,
)
print(report.table())
report.to_json("react_results.json")
```

## What's in the box

- **Core metric** — onset/recovery, Kaplan-Meier survival, cluster-bootstrap CIs, per-event-type breakdowns, excess-REACT vs. an oracle floor.
- **Reference climatology** — `global`, `kinematic` (speed/heading-conditioned), and `position` modes, with a robust floor component.
- **Pluggable event detection** — a built-in kinematic changepoint detector, a `ruptures` adapter, and a bring-your-own-timestamps path.
- **Calibration diagnostics** — REACT scores your forecast *as delivered*; the calibration report (coverage χ², PIT, sharpness) tells you *why* a model is slow without silently "fixing" it.
- **Baselines** — Constant-Velocity, CTRV, IMM, plus **pretrained `GRUPosition` and `GRUOrientation`** (extras=`[torch]`). These GRUs were trained on a large corpus of retail pedestrian motion for the original REACT research; checkpoints ship with the package and are available wherever you pick comparison models (`baselines=[...]`, `rm.baselines.build(...)`, quickstart).
- **Calibration-robust scoring** — CRPS / energy-score REACT variants for models without trustworthy covariances.

## Calibration: call out, never correct

REACT is built on the log score, a strictly proper scoring rule — a miscalibrated forecast is *deservedly* penalized. A forecast that fails to recover utility because its uncertainty is wrong has genuinely failed, and REACT reports that. The calibration report makes the cause **visible** (so you can tell genuine slow adaptation from static over/under-confidence) but never modifies your forecast. An opt-in `counterfactual_react` answers the separate "what if my covariances were scalar-calibrated?" question, clearly labeled and never on the headline number.

## License

Apache-2.0.
