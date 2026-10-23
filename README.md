# reactmetric

**REACT (Regime Event Adaptive Correction Time)** evaluates probabilistic trajectory forecasts around changes in motion. After each detected stop, start, turn or speed change it reports:

- **anticipation**: whether the forecast stayed ahead of a climatological reference through the change;
- **REACT**: how long it took to become confidently ahead again;
- **censoring**: whether that happened within the window;
- **relapse**: whether the recovery held.

ADE, FDE and NLL average over time and wash these moments out. On the paper's two pedestrian datasets, learned forecasters lead on displacement and likelihood and recover more completely, yet simple kinematic predictors often recover sooner right after a change, especially at longer leads.

Method: Breseman and Woollard, *REACT: Measuring Regime Adaptation in Trajectories*, TGF 2026. Paper reproduction: [calvin-breseman/react-reproducibility](https://github.com/calvin-breseman/react-reproducibility).

## Install

```bash
pip install reactmetric                # core: numpy, scipy
pip install "reactmetric[viz]"         # + plots
pip install "reactmetric[torch]"       # + the GRU baselines
```

## Quickstart

```python
import reactmetric as rm

data = rm.datasets.load_eindhoven_sample()                # or rm.TrajectorySet.from_dataframe(...)
calib, test = data.split(fraction=0.3, seed=0)

scorer = rm.ReactScorer.fit(calib, info=(4.2,), leads=(5, 10), distortions=(0.1,))
result = scorer.evaluate(
    {"MyModel": my_forecaster,                            # predict(traj) -> rm.Predictive
     "CV": rm.baselines.ConstantVelocity.fit(calib)},
    test,
)
print(result.table(lead=10, info=4.2, distortion=0.1))   # REACT(10, 0.1 m; 4.2)
result.to_json("react.json")
```

`python examples/quickstart.py` runs this end to end on the bundled 36-track Eindhoven sample in under a minute.

## How it works

For a forecast at lead `h`, REACT compares its log score with a climatology (where a person is likely to be `h` frames ahead given only their last step), and decides frame by frame, with posterior probability at least 0.95, whether the forecast's true advantage is negative (onset) or positive (recovery). That advantage is judged against a latent truth whose sharpness is set by an assumed information level `I_traj` (nats) and a measurement-noise floor. Every model on a dataset shares the truth posterior, the events and the reference; the only model-specific input is its predictive density. Oracle B, the best forecast the assumptions allow, is scored alongside as the floor.

Results are written `REACT(h, d; I_traj)`: lead, optional target distance `d` (the forecast must also localize the person within `d` metres RMS), and information level. See [docs/concepts.md](docs/concepts.md).

## Bring your own model

- **Gaussian forecasts**: `rm.Predictive(issue_frames, means, covs)`, means `(n, H, 2)`, covs `(n, H, 2, 2)`.
- **Mixtures** (Trajectron++ and similar): `rm.MixturePredictive(issue_frames, weights, means, covs)`, scored natively.
- **Conventions**: the issue frame is the last observed frame and lead `h` targets `issue + h`, as in ETH/UCY, Argoverse and trajdata. Forecasts are needed at every frame of long tracks at about 10 Hz, in metres.
- **Files**: `rm.Forecasts.from_dataframe(...)` for long tables of `(mu_x, mu_y, sigma_x, sigma_y, rho)`; `reactmetric evaluate` on the command line. See [docs/io.md](docs/io.md).

## Defaults (the paper's)

Leads 1, 5, 10, 15 frames; information levels 3, 4.2 and 6 nats (headline 4.2); target distances ∞, 0.1 and 0.3 m; decisions at posterior probability 0.95; onset window `h + 5` frames; event windows up to 150 frames (15 s) or the next event; three-frame relapse within 10 frames of recovery; Bayesian changepoint events (`rm.events.BayesianChangepoints.tuned_sg()`).

## License

Apache-2.0.
