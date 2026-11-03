"""The README's headline figure, from real Eindhoven tracks.

    python docs/figures/make_headline.py .data/eindhoven_3000tracks.csv

Left: one stop, constant velocity's realized score against the climatology frame by frame,
with REACT's onset and recovery. Right: Kaplan-Meier curves of time to recovery at lead 10
(d = infinity, 4.2 nats) for three forecasters and Oracle B. Writes docs/assets/react_headline.png (and the
result it was drawn from, docs/figures/headline_result.json).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

import reactmetric as rm
from reactmetric import information as ri
from reactmetric.core.trajectory import TrajectorySet
from reactmetric.metric.survival import kaplan_meier
from reactmetric.scorer import MIN_FRAMES, model_frames

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "react_headline.png")
LEAD, D = 10, 0.1  # scored at d = infinity and 0.1 m; drawn at d = infinity (DRAW)
DRAW = 0.0
# reference palette (light): blue, orange, aqua; Oracle B in a recessive grey
COLORS = {"ConstantVelocity": "#2a78d6", "IMM": "#eb6834", "GRUOrientation": "#1baf7a",
          "oracle_B": "#8a8984"}
LABELS = {"ConstantVelocity": "Constant velocity", "IMM": "IMM", "GRUOrientation": "GRU",
          "oracle_B": "Oracle B (the floor)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def score(csv, n_tracks=400):
    data = rm.datasets.load_eindhoven(path=csv)
    data = TrajectorySet({t: data[t] for t in list(data.ids)[:n_tracks]})
    calib, test = data.split(fraction=0.3, seed=0)
    scorer = rm.ReactScorer.fit(calib, info=(4.2,), leads=(LEAD,), distortions=(D,), n_jobs=8)
    models = {"ConstantVelocity": rm.baselines.ConstantVelocity.fit(calib),
              "IMM": rm.baselines.IMM(), "GRUOrientation": rm.baselines.GRUOrientation()}
    return scorer, test, models, scorer.evaluate(models, test, n_jobs=8)


def pick_event(scorer, test, models, result):
    """A clear event: constant velocity ahead before it, behind from the onset, then
    confidently ahead again and staying ahead for 30 frames."""
    rows = [r for r in result.records if r["model"] == "ConstantVelocity"
            and r["distortion"] == DRAW and r["status"] == "recovered" and not r["relapse3"]
            and 12 <= r["react"] <= 40 and r["onset"] is not None]
    ids = {str(t): t for t in test.ids}
    best, best_q = None, -np.inf
    for r in rows:
        t, sc = realized(scorer, test[ids[r["track"]]], models["ConstantVelocity"], r["t0"])
        sm = np.convolve(sc, np.ones(5) / 5, mode="same")
        before, during = sm[(t < -2) & (t >= -25)], sm[(t > r["onset"] + 2) & (t < r["react"] - 2)]
        after = sm[(t >= r["react"] + 2) & (t < r["react"] + 32)]
        if len(before) < 15 or len(after) < 25 or len(during) < 3 or t.max() < r["react"] + 32:
            continue
        q = min(before.min(), after.min(), -during.max())
        if q > best_q:
            best, best_q = r, q
    return best


def realized(scorer, traj, model, t0):
    ref = scorer.climatology
    target, y, wts, mus, covs = ref.mixture_frames(traj, LEAD, MIN_FRAMES)
    ref_lp = ri.mixture_logpdf(y[:, None], ri.mixture_prep(wts, mus, covs))[:, 0]
    tm, mu, S = model_frames(model.predict(traj), len(traj.positions), LEAD, ref)
    _, ia, ib = np.intersect1d(target, tm, return_indices=True)
    s = ri.gaussian_logpdf(y[ia], mu[ib], S[ib]) - ref_lp[ia]
    t = target[ia] - t0
    keep = (t >= -30) & (t <= 90)
    return t[keep], s[keep]


def draw(scorer, test, models, result, ev):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # left: one event
    tid = next(t for t in test.ids if str(t) == ev["track"])  # records hold ids as strings
    t, sc = realized(scorer, test[tid], models["ConstantVelocity"], ev["t0"])
    smooth = np.convolve(sc, np.ones(5) / 5, mode="same")
    a.axhline(0, color=INK2, lw=1)
    a.plot(t, sc, color=COLORS["ConstantVelocity"], lw=1, alpha=0.35)
    a.plot(t, smooth, color=COLORS["ConstantVelocity"], lw=2)
    a.set_xlim(-30, min(90, ev["react"] + 45))
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + 0.45 * (hi - lo))
    lo, hi = a.get_ylim()
    what = {"stop": "person stops", "start": "person starts walking",
            "speed_change": "speed changes", "turn": "person turns",
            "both": "person turns and slows"}.get(ev["event_type"],
                                                                         "motion changes")
    if ev["event_type"] == "both" and (ev["heading_change"] or 0) > 150:
        what = (f"person turns around ({ev['heading_change']:.0f}°), "
                f"{ev['speed_before']:.1f} → {ev['speed_after']:.1f} m/s")
    for x, name, dy in ((0, what, 0.97), (ev["onset"], "onset: confidently behind", 0.88),
                        (ev["react"], "recovered: confidently ahead again", 0.79)):
        a.axvline(x, color=INK if x == 0 else INK2, lw=1, ls=(0, (2, 2)) if x == 0 else "-")
        a.text(x + 1.5, lo + dy * (hi - lo), name, fontsize=8.5, color=INK2, va="top")
    y_br = lo + 0.66 * (hi - lo)
    a.annotate("", xy=(ev["react"], y_br), xytext=(0, y_br),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    a.text(ev["react"] + 2, y_br, f"REACT = {ev['react']} frames", ha="left", va="center",
           fontsize=10, color=INK, fontweight="bold")
    a.set_xlabel("frames after the event")
    a.set_ylabel("realized score vs climatology (nats)\nabove 0: more useful than climatology")
    a.set_title("One regime change, constant velocity, lead 10", loc="left", fontsize=11,
                color=INK)

    # right: Kaplan-Meier over events with an onset
    names = ["ConstantVelocity", "IMM", "GRUOrientation", "oracle_B"]
    ends = {}
    for m in names:
        rs = [r for r in result.records if r["model"] == m and r["distortion"] == DRAW]
        st = np.array([r["status"] for r in rs])
        dur = np.array([np.nan if r["duration"] is None else r["duration"] for r in rs], float)
        onset = np.isin(st, ("recovered", "censored"))
        tt, ss = kaplan_meier(np.where(onset, dur, 0.0), (st == "recovered") | ~onset)
        tt, ss = np.r_[0.0, tt, 150.0], np.r_[1.0, ss, ss[-1]]
        b.step(tt, ss, where="post", color=COLORS[m], lw=2, label=m,
               ls=(0, (4, 2)) if m == "oracle_B" else "-")
        ends[m] = ss[-1]
    b.set_xlim(0, 150)
    b.legend(handles=b.lines, labels=[f"{LABELS[m]}: {100 * ends[m]:.0f}% never recover"
                                      for m in names],
             frameon=False, fontsize=9, loc="upper right")
    b.set_xlim(0, 150)
    b.set_ylim(0, 1.02)
    b.set_xlabel("frames after the event")
    b.set_ylabel("share of events not yet recovered")
    n_ev = len({(r["track"], r["t0"]) for r in result.records})
    b.set_title(f"REACT({LEAD}, ∞; 4.2): {n_ev} events on Eindhoven", loc="left",
                fontsize=11, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}")


def main():
    warnings.simplefilter("ignore")
    cache = os.path.join(HERE, "_headline_cache")
    if os.path.exists(os.path.join(cache, "scorer.pkl")):
        scorer = rm.ReactScorer.load(cache)
        result = rm.ReactResult.load(os.path.join(HERE, "headline_result.json"))
        data = rm.datasets.load_eindhoven(path=sys.argv[1])
        data = TrajectorySet({t: data[t] for t in list(data.ids)[:400]})
        calib, test = data.split(fraction=0.3, seed=0)
        models = {"ConstantVelocity": rm.baselines.ConstantVelocity.fit(calib)}
    else:
        scorer, test, models, result = score(sys.argv[1])
        scorer.save(cache)
        result.to_json(os.path.join(HERE, "headline_result.json"))
    ev = pick_event(scorer, test, models, result)
    print("event", ev["track"], ev["t0"], "onset", ev["onset"], "react", ev["react"])
    print(result.table(lead=LEAD).to_string(index=False))
    draw(scorer, test, models, result, ev)


if __name__ == "__main__":
    main()
