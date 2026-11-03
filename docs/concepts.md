# The REACT metric

REACT (Regime Event Adaptive Correction Time) follows a probabilistic trajectory forecast through a detected change in a person's motion (a stop, a start, a turn, a change of speed). It asks four questions per event: did the forecast stay ahead of a climatological reference through the change (**anticipation**), how long did it take to become confidently ahead again (**REACT**), did that happen within the window (**censoring**), and did it hold (**relapse**). The method is that of Breseman and Woollard, *REACT: Measuring Regime Adaptation in Trajectories* (TGF 2026); this page summarizes it and names the library objects that implement each part.

Every REACT number depends on two choices and one assumption, and carries them:

$$\mathrm{REACT}(h, d;\ I_{\text{traj}})$$

the forecast lead $h$ (observations), an optional target distance $d$ (metres; $d = \infty$ is the plain comparison with climatology), and the assumed information level $I_{\text{traj}}$ (nats). The decision threshold, onset window and noise floor are fixed per dataset and reported once.

## Usefulness against a latent truth

At frame $t$ a forecaster issued at $t-h$ gives a density $p_t$ for the position $y_t$; the climatology gives $r_t$. The realized score is

$$s_t = \log p_t(y_t) - \log r_t(y_t)$$

in nats, an unbiased draw of the forecast's true information advantage $A_t = \mathrm{KL}(q_t \| r_t) - \mathrm{KL}(q_t \| p_t)$, where $q_t$ is the true law of $y_t$. One draw is noisy, so REACT decides on $A_t$, not $s_t$. It models the truth locally as a Gaussian around an unknown centre whose prior is the climatology's own spread, per climatology regime $k$ (walking or stopped):

$$m_t \mid k \sim \mathcal N\big(\mu_{k,t},\ (1-c)\Sigma_{k,t} - \sigma^2 I\big), \qquad q_t = \mathcal N\big(m_t,\ c\,\Sigma_{k,t} + \sigma^2 I\big), \qquad c = e^{-I_{\text{traj}}}$$

so before any observation $y_t$ is distributed exactly as the climatology. $I_{\text{traj}}$ is how much of the next position is assumed knowable beyond climatology; $\sigma$ is the measurement-noise floor, which keeps the modelled truth from being sharper than the sensor. The posterior of $m_t$ is inferred from the whole observed track and is the same for every model; the model enters only through its density.

Library: `reactmetric.information.truth_posterior`, `RegimeTests`, `estimate_noise_floor`.

## Decisions and outcomes

For each frame of an event window REACT computes the posterior probabilities that the forecast is behind, $P(A_t < 0 \mid y)$, and ahead by a margin, $P(A_t > \varepsilon_t \mid y)$, exactly (no Monte Carlo). A verdict needs probability at least $\gamma = 0.95$. For an event at $t_0$ with window end $T = \min(t_0 + 150,\ t_{\text{next}})$:

$$t_{\text{on}} = \min\{t_0 < t \le t_0 + h + 5 : P(A_t < 0 \mid y) \ge \gamma\}, \qquad t_{\text{rec}} = \min\{t_{\text{on}} < t \le T : P(A_t > \varepsilon_t \mid y) \ge \gamma\}$$

and $\mathrm{REACT} = t_{\text{rec}} - t_0$.

| Outcome | Meaning |
| --- | --- |
| anticipated | no onset within the onset window $t_0 + h + 5$ |
| recovered | onset, then a confident recovery within the window |
| censored | onset, no recovery before the window ends |
| relapsed | after recovery, confidently behind on three consecutive frames starting within 10 frames |

Recovery times are right-censored, so they are summarized with Kaplan–Meier curves: over all events (anticipated ones count as recovered at frame 0) and given onset. Median REACT is the median given onset; the restricted mean (area under the curve to 150 frames) stays informative when most events are anticipated. REACT is a profile, not one leaderboard number: anticipation, recovery time, completeness and relapse can favour different models.

Library: `ReactResult.table()` and `ReactResult.summary()`; [Reading results](results.md) explains every column.

## Distortion targets

Beating climatology is a relative advantage, not a spatial precision. With a target distance $d$, recovery requires beating climatology by enough information to localize the person within RMS distance $d$:

$$\varepsilon_t(d) = \max\big(0,\ H_{r,t} - (1 + \log 2\pi + \log(d^2/2))\big)$$

with $H_{r,t}$ the entropy of a Gaussian with the climatology's full covariance. The paper's headline target is $d = 0.1$ m. A target finer than the truth's own spread is unreachable; Oracle B's censoring at each $d$ is the check.

Library: `reactmetric.information.distortion_margin`.

## Events

Events come from the trajectory alone: an offline Bayesian changepoint model (Fearnhead 2006) on Savitzky–Golay-smoothed speed, turn rate and velocity components, with a tuned hazard and minimum segment length, then typed as stop, start, speed change or turn from the motion before and after. Detector errors change which transitions are evaluated (false events inflate anticipation; missed changes can show up as relapses). Labelled or externally defined events can be supplied instead.

Library: `reactmetric.events.BayesianChangepoints()`; `reactmetric.events.from_times` for your own events. [Choosing events](events.md) covers the types and how to select them.

## Climatology

The reference is the distribution of a person's position $h$ frames ahead given only their most recent one-frame displacement: a speed-conditioned mixture of a moving regime (in the body frame of the current speed and heading) and a stopped regime around the current position, fitted by EM on calibration tracks. It is not an uninformative null: a strong forecaster beats it on average and can still fall behind it around a change.

Library: `reactmetric.reference.fit_climatology`.

## Oracle B and the information level

Oracle B is the hindsight posterior predictive under the same truth model: the most any forecast could extract given the assumptions. Its REACT is the earliest time a forecast's advantage could be confirmed, and its never-recovered rate shows whether a target is reachable.

$I_{\text{traj}}$ is an assumption, not an estimate. A forecaster's mean gain on calm stretches bounds it from below (the *demonstrated information*). In simulation, under-assuming information causes false onsets for sharp forecasters while over-assuming is comparatively benign, so the paper reports 3, 4.2 and 6 nats and picks the headline by a stated rule: the lowest level at which Oracle B recovers on at least 95% of events at every lead. The paper does not say at which target distance; on Eindhoven the rule gives its 4.2 nats at d = 0.1 m (3 nats at d = ∞), so the library judges it at the smallest finite target.

Library: Oracle B is scored automatically as `oracle_B`; `ReactResult.headline_info()` applies the rule and `table()` uses it by default; `ReactScorer.diagnostics[lead]["calm_mean_score"]` holds the demonstrated information when models are passed to `fit`.

## Defaults

The library's defaults are the paper's. Leads and windows are counted in observations; the
paper's data were 10 Hz, so the right-hand column gives the time each default stands for there.

| Setting | Argument | Default | At 10 Hz |
| --- | --- | --- | --- |
| leads $h$ | `leads` | 1, 5, 10, 15 | 0.1 to 1.5 s |
| information levels | `info` | 3, 4.2, 6 nats | |
| target distances | `distortions` | $\infty$, 0.1 m, 0.3 m | |
| decision probability $\gamma$ | `p` | 0.95 | |
| onset window | `onset_margin` | $h + 5$ frames | $h$ + 0.5 s |
| event window | `window` | 150 frames or the next event | 15 s |
| shortest event window | `min_window` | 25 frames | 2.5 s |
| relapse | `relapse_run`, `relapse_window` | 3 consecutive frames starting within 10 of recovery | 0.3 s within 1 s |
| calm stretches | `calm_exclusion` | 20 frames before to 60 after each event left out | 2 s, 6 s |
| noise floor | `sigma` | estimated from stationary calibration frames (51-frame windows) | 5.1 s |
| events | `events` | `BayesianChangepoints()`: hazard 9e-4 per frame, segments of at least 30 frames, 5-frame smoothing | 3 s, 0.5 s |

## Other frame rates

Nothing in REACT assumes 10 Hz: the climatology, the noise floor and the event kinematics use
each track's own time step. The counts above, though, are counts. At another frame rate, decide
what time spans you want and set the counts to match. For 5 Hz data, halving them keeps the
paper's time spans:

```python
fps_ratio = 0.5                                     # your rate / 10 Hz
n = lambda frames: max(1, round(frames * fps_ratio))
det = rm.events.BayesianChangepoints(hazard=9e-4 / fps_ratio, min_size=n(30),
                                     temper=3.0 * fps_ratio, smooth_window=3)
scorer = rm.ReactScorer.fit(
    calib, leads=(n(10),), events=det, window=n(150), min_window=n(25), onset_margin=n(5),
    relapse_window=n(10), relapse_run=max(2, n(3)), calm_exclusion=(n(20), n(60)),
    pre_window=n(30), sigma=rm.information.estimate_noise_floor(calib, window=n(51) | 1,
                                                                 quad=max(3, n(9) | 1))[0],
)
```

On 300 Eindhoven tracks downsampled to 5 Hz this keeps the 10 Hz ordering of constant velocity
and IMM on anticipation and censoring, with median recovery 0.2–0.4 s later: fewer observations
per second mean less evidence per second, so confident decisions take a little longer. Compare REACT numbers only between data at
the same frame rate.
