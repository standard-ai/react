# Choosing events

REACT measures recovery from changes in motion, so which changes count is as much a part of the
result as the forecasts. A detector that fires on noise inflates anticipation (there was nothing
to anticipate); one that misses changes leaves them inside other events' windows, where they
show up as relapses. Decide what you are measuring before reading the numbers.

## The default: the paper's detector

`rm.events.BayesianChangepoints()` finds changepoints offline in each track's smoothed speed,
turn rate and velocity, then types each one from the motion before and after:

| Type | Meaning |
| --- | --- |
| `stop` | moving, then standing |
| `start` | standing, then moving |
| `speed_change` | speed changes by at least 0.2 m/s while moving |
| `turn` | heading changes by at least 25° while moving |
| `both` | a turn and a speed change together |
| `turn_change` | a change in turn rate the heading rule did not classify (rare) |

Its settings (hazard, minimum segment length, smoothing) are counted in observations and were
tuned on 10 Hz pedestrian data; see [Other frame rates](concepts.md#other-frame-rates).

Look at what it finds before scoring:

```python
det = rm.events.BayesianChangepoints()
events = det.detect_set(test)                       # {track id: [Event, ...]}
from collections import Counter
print(Counter(e.event_type for evs in events.values() for e in evs))
```

## Measuring some events only

Two ways, with different meanings:

- **Filter the result.** Score every event, then keep some. Windows stay as detected: every
  change still ends the window of the change before it.

  ```python
  stops = result.subset(event_type="stop")
  stops.table(lead=10)
  result.table(lead=10, by="event_type")            # or one row per type
  ```

- **Select at scoring time.** Only the chosen events exist, so windows run until the next chosen
  event (or 150 frames). Use this when the other changes are not changes in your setting.

  ```python
  scorer = rm.ReactScorer.fit(calib, events=rm.events.select(det, types=("stop", "start")))
  ```

  `select(det, where=lambda e: abs(e.d_speed_level) > 0.5)` filters on an event's measured
  effect instead.

## Your own events

If you have labels (annotated manoeuvres, signal changes, a door opening), pass them:

```python
labels = {"track_7": [120, 410], "track_9": [88]}     # frames of each change
scorer = rm.ReactScorer.fit(calib, events=rm.events.from_times(labels, event_type="door"))
```

The same events must be defined for the calibration tracks: REACT leaves the frames around them
out when it fits the truth model's correlations.

## Checking sensitivity

The paper reports how its conclusions change with the detector's hazard (section 4.7). A quick
version for your data: score with two settings and compare the profiles.

```python
loose = rm.events.BayesianChangepoints(hazard=3e-3)
for name, ev in [("default", det), ("loose", loose)]:
    sc = rm.ReactScorer.fit(calib, events=ev, info=(4.2,), leads=(10,))
    print(name, sc.evaluate(models, test).table(lead=10))
```

If the model ranking flips between settings, report both.
