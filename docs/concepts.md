# The REACT metric

## Information score

For a forecaster $m$ producing a predictive density $p_m$ over future positions, and a reference (climatology) density $p_\text{ref}$, the per-frame **information score** is

$$ s_m(f) = \mathrm{nll}_\text{ref}(f) - \mathrm{nll}_m(f) \quad (\text{nats}), $$

a pointwise log Bayes factor. $s_m(f) > 0$ means the model is more informative than the climatology at frame $f$; $s_m(f) < 0$ means it is *worse than knowing nothing*.

Using the strictly proper log score is deliberate: a miscalibrated forecast is *deservedly* penalized (see [Calibration policy](calibration.md)). CRPS / energy-score variants are available via `scoring="crps"|"energy"` for users who cannot supply trustworthy covariances.

## Onset, recovery, and REACT

After a regime change (a "commitment event") at frame $t_0$:

- **onset**: $s_m < 0$ sustained for `onset_consecutive` frames — the forecast has fallen below the reference;
- **recovery**: $s_m \ge \epsilon$ sustained for `recovery_consecutive` frames — it beats the reference by the task margin $\epsilon$ again.

$$ \mathrm{REACT}(e) = f_\text{recovery} - t_0. $$

If there is no sustained onset after $t_0$, the model **anticipated** the event ($\mathrm{REACT}=0$). If it never recovers within the window it is **right-censored** (and **superseded** if the window was cut short by the next event).

Target-frame indexing keeps the x-axis as "frames since the maneuver was observed": the lead-$h$ forecast of frame $f$ is issued at $f-h+1$, so a maneuver at $t_0$ elevates the trace at $t_0$, not $h$ frames before any evidence exists.

## Survival summary

Recovery times are right-censored, so REACT is summarized by the **Kaplan-Meier** survival function over events, with **cluster-bootstrap** confidence intervals (resampling tracks, since events within a track are not independent). The atom at zero is the anticipation rate.

## The reference climatology

The reference conditions only on event-continuous quantities so it is not itself surprised by the maneuvers REACT measures:

- **`global`** — the marginal law of lead-step displacements (a Gaussian mixture).
- **`kinematic`** (default, 2D) — conditioned on causal speed and heading (body-frame), so trivially-predictable persistence is conditioned out while genuine maneuvers still surprise it.
- **`position`** — environment structure via position-binned mixtures, shrunk toward the global law.

A fixed broad **floor component** bounds the reference surprise so rare large displacements cannot mechanically flatter a heavy-tailed model.

## The oracle ladder

REACT separates *intrinsic event difficulty* from *model deficiency*:

- **Oracle A (hindsight)** — acausal smoother; events where even it is surprised are tracking artifacts (`filter_artifacts`).
- **Oracle B (climatology-seeded)** — best causal forecast given only post-change information; its REACT is the irreducible re-estimation **floor**.
- **Oracle C** — Oracle B plus the quickest-detection delay (`detection_floor_frames`).

**Excess REACT** of model $m$ on event $e$ is $\mathrm{REACT}_m(e) - \mathrm{REACT}_\text{oracle}(e)$ — how far the model is from the best possible causal recovery.

## Rate-distortion task anchors

For a concrete localization task (RMS error $d$), the necessary information is the Gaussian rate $\epsilon^\* = \tfrac{1}{2}\log\!\frac{\det\Sigma_\text{ref}}{\det\Sigma_d}$. `rm.task_react(...)` sets $\epsilon = \epsilon^\*$ so REACT becomes "time to recover enough information to localize at $d$".
