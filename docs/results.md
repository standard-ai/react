# Reading results

`scorer.evaluate(...)` returns a `ReactResult`: one record per (lead, information level, target
distance, model, event), plus the settings they were made under.

```python
result      # ReactResult(596 events, models=['ConstantVelocity', 'IMM', 'GRUOrientation', 'oracle_B'], ...)
result.table(lead=10)                   # REACT(10, ∞; headline level), one row per model
result.table(lead=10, distortion=0.1)   # REACT(10, 0.1 m; ...)
```

## The table

| Column | Meaning |
| --- | --- |
| `events` | events scored for this model (shared by every model unless `event_set="all"`) |
| `anticipated` | share of events where the forecast never fell confidently behind the climatology within `lead + 5` frames of the event |
| `recovered` | share of events with an onset and then a confident recovery within the window |
| `censored` | share of events with an onset and no recovery before the window ended (never recovered) |
| `median_react` | median frames from the event to recovery, given onset (Kaplan–Meier; NaN when fewer than half the onsets recover) |
| `rmst` | restricted mean frames to recovery over all events, anticipated ones counting as 0, up to the window; lower is better |
| `relapsed` | share of recovered events that fall behind again: 3 consecutive frames starting within 10 frames of recovery |

`anticipated + recovered + censored = 1`. `columns="all"` adds every summary field; their
meanings are in `rm.results.GLOSSARY`. The restricted mean is the most stable single number for
comparing models: the median is undefined when most onsets never recover, and anticipation and
censoring pull in opposite directions.

REACT is a profile, not a leaderboard: a model can recover fastest and still fail to recover on
more events, as constant velocity does in the README's figure. Read the columns together.

## Target distance

`distortion=None` (the default) is d = ∞: recovery means confidently more useful than the
climatology. A number `d` in metres adds a margin: recovery also requires the forecast to place
the person within about `d` metres. `0`, `float("inf")` and `"clim"` are accepted for d = ∞;
`distortion="all"` shows every target. Asking for a lead, level, target or model the result does
not have raises an error that lists what it does have.

## Information level

Every REACT number depends on the assumed information level `I` (nats): how much of a person's
next position is knowable at all. `table()` uses the paper's rule by default
(`info="auto"`): the lowest level evaluated at which Oracle B recovers on at least 95% of events
at every lead, judged at the smallest finite target distance. `table().attrs["info_reason"]`
says which level was chosen and why, and `result.headline_info()` returns the same.

Ask for a level explicitly with `table(info=4.2)`, or `info="all"` for every level. If the
conclusion changes between levels, report that; the paper reports 3, 4.2 and 6 nats.

## Confidence intervals

Events on the same track are not independent, so intervals resample tracks: each bootstrap draw
takes test tracks with replacement, with all their events (paper section 4.6).

From the README's 400-track Eindhoven run (189 test tracks):

```python
result.ci("rmst", model="GRUOrientation", lead=10)
# {'estimate': 13.4, 'low': 11.9, 'high': 14.8, 'events': 596, 'tracks': 189}

result.compare("GRUOrientation", "ConstantVelocity", "rmst", lead=10)
# {'a': 13.4, 'b': 25.2, 'difference': -11.7, 'low': -16.2, 'high': -7.1, 'share_a_lower': 1.0, ...}

result.table(lead=10, ci=True)          # adds _low / _high columns for the main statistics
```

`compare` is paired: both models are evaluated on the same draws of the same shared events,
which is much tighter than comparing two separate intervals. For `rmst` and `median_react`, a
negative difference means the first model recovers sooner. When the median is undefined in a
draw it counts as beyond the window, so its interval can be open-ended (`inf`).

These intervals cover the sampling of test tracks. They are conditional on the fitted
climatology and truth model, and they do not cover the choice of information level, target or
event detector; vary those separately.

## Breakdowns

```python
result.table(lead=10, by="event_type")          # one row per model and event type
result.subset(event_type="stop").table(lead=10) # stops only
result.summary(by="event_type")                 # the same as a dict
result.to_dataframe()                           # every record, for your own analysis
```

## Saving and reloading

```python
result.to_json("react.json")                    # settings, summary, diagnostics and records
again = rm.ReactResult.load("react.json")       # tables, intervals and plots work as before
result.to_csv("records.csv")                    # records only; reload with settings="react.json"

scorer.save("react_fit/")                       # the fitted scorer and its overlap tables
scorer = rm.ReactScorer.load("react_fit/")      # score more models against the same fit
```

Overlap tables are the slow part of fitting. `ReactScorer.fit(..., table_dir="tables/")` keeps
them on disk and reuses any table whose fingerprint (climatology, level, noise floor, grid)
matches, so refitting the same calibration data is fast.

## Plots

```python
rm.viz.km_curves(result, lead=10, save="km.png")             # share not yet recovered
rm.viz.outcome_profile(result, lead=10, save="profile.png")  # anticipation, censoring, relapse
```
