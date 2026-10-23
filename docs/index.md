# reactmetric

**REACT (Regime Event Adaptive Correction Time)** evaluates probabilistic trajectory forecasts around changes in motion. After each detected stop, start, turn or speed change it reports whether a forecast stayed ahead of a climatological reference (**anticipation**), how long it took to become confidently ahead again (**REACT**), whether it got there within the window (**censoring**), and whether that held (**relapse**). Aggregate metrics such as ADE, FDE and NLL average these moments away.

The method is that of Breseman and Woollard, *REACT: Measuring Regime Adaptation in Trajectories* (TGF 2026). Code that reproduces the paper is at [calvin-breseman/react-reproducibility](https://github.com/calvin-breseman/react-reproducibility); this library computes the same numbers (`tests/parity/` checks it).

## Install

```bash
pip install reactmetric              # core: numpy, scipy
pip install "reactmetric[viz]"       # + plots
pip install "reactmetric[datasets]"  # + parquet dataset loaders
pip install "reactmetric[torch]"     # + the GRU baselines
```

## Example

```python
import reactmetric as rm

data = rm.TrajectorySet.from_dataframe(df, id="track_id", coords=["x", "y"], dt=0.1)
calib, test = data.split(fraction=0.3, seed=0)

scorer = rm.ReactScorer.fit(calib, info=(3.0, 4.2, 6.0))     # shared by every model
result = scorer.evaluate({"MyModel": my_forecaster,           # predict(traj) -> rm.Predictive
                          "CV": rm.baselines.ConstantVelocity.fit(calib)}, test)
print(result.table(lead=10, info=4.2, distortion=0.1))       # REACT(10, 0.1 m; 4.2)
rm.viz.km_curves(result, lead=10, info=4.2, save="km.png")
```

Every number carries its lead, target distance and assumed information level. See [The REACT metric](concepts.md) for the method and defaults, [I/O formats](io.md) for the forecast conventions (the issue frame is the last observed frame), and `examples/quickstart.py` for a runnable example on the bundled Eindhoven sample.

## What it is for

REACT needs forecasts issued at every frame of long, continuous tracks of people (pedestrians, shoppers) at about 10 Hz, as densities: Gaussian or Gaussian-mixture. It is not a single leaderboard number: anticipation, recovery time, completeness and relapse can favour different models, and that is the point.
