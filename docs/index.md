# reactmetric

**REACT (Regime Event Adaptive Correction Time)** measures *how quickly a trajectory forecast becomes genuinely useful again after a disruptive regime change*. It complements ADE/FDE with an **adaptation-speed** axis.

> ADE/FDE tell you how good a forecast is *on average*. REACT tells you how fast it *recovers* when the world changes.

## Install

```bash
pip install reactmetric            # core (numpy/scipy/sklearn)
pip install "reactmetric[all]"     # + viz, datasets, ruptures
pip install "reactmetric[torch]"   # + pretrained GRUPosition / GRUOrientation
```

Pretrained **`GRUPosition`** and **`GRUOrientation`** checkpoints ship with the
package (extras=`[torch]`). They were trained on a large retail pedestrian
motion corpus for the original REACT research and are available wherever you
pick comparison models (`baselines=[...]`, `rm.baselines.build(...)`, quickstart).

## 60-second example

```python
import reactmetric as rm

data = rm.TrajectorySet.from_dataframe(df, id="track_id", coords=["x", "y"], dt=0.1)
calib, test = data.split(fraction=0.2, seed=0)

report = rm.benchmark(
    forecasts={"MyModel": my_predictives},      # dict[id -> rm.Predictive]
    data=test, calibration=calib,
    baselines=["CTRV", "IMM", "ConstantVelocity", "GRUPosition", "GRUOrientation"],
    oracle="B", lead=10, epsilon=0.0,
)
print(report.table())
report.to_json("react_results.json")
report.plot_km("km.png")
```

See [The REACT metric](concepts.md) for the math, [Calibration policy](calibration.md) for the "call out, never correct" stance, and the [API reference](api.md) for the full surface. A runnable end-to-end example lives in `examples/quickstart.py` and `examples/tutorial.ipynb`.
