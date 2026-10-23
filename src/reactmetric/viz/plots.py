"""Plots of a ReactResult (extras=[viz]). Every function takes ``save`` and returns the Figure.

``km_curves``: the share of events not yet recovered, frames after the event (paper Eq. 7), over
all events (anticipated ones recover at frame 0) or given onset. ``outcome_profile``: each
model's anticipation, censoring given onset, and relapse rates side by side.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..metric.survival import kaplan_meier

# Okabe-Ito: distinguishable under the common colour-vision deficiencies
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
           "#000000")
ORACLE_COLOR = "#7f7f7f"


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install 'reactmetric[viz]'") from e
    return plt


def react_label(lead, distortion=None, info=None) -> str:
    """REACT(h, d; I) in the paper's notation; d = infinity for the plain comparison."""
    d = "∞" if not distortion else f"{distortion:g} m"
    return f"REACT({lead}, {d}; {info:g})" if info is not None else f"REACT({lead}, {d})"


def _group(records, lead, info, distortion):
    out = {}
    for r in records:
        if (r["lead"] == lead and np.isclose(r["info"], info)
                and np.isclose(r["distortion"], distortion)):
            out.setdefault(r["model"], []).append(r)
    return out


def km_curves(result, lead: int, info: float, distortion: float = 0.0, given_onset: bool = True,
              models: Optional[Sequence[str]] = None, max_frames: int = 150, ax=None,
              save: Optional[str] = None):
    """Kaplan-Meier curves of time to recovery, one per model (Oracle B in grey)."""
    plt = _require_mpl()
    groups = _group(result.records, lead, info, distortion)
    names = [m for m in (models or sorted(groups, key=lambda m: (m == "oracle_B", m)))
             if m in groups]
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.2))
    for i, m in enumerate(names):
        rs = groups[m]
        st = np.array([r["status"] for r in rs])
        dur = np.array([np.nan if r["duration"] is None else float(r["duration"]) for r in rs])
        onset = np.isin(st, ("recovered", "censored"))
        if given_onset:
            if not onset.any():
                continue
            t, s = kaplan_meier(dur[onset], st[onset] == "recovered")
        else:
            t, s = kaplan_meier(np.where(onset, dur, 0.0), (st == "recovered") | ~onset)
        t = np.r_[0.0, t]
        s = np.r_[1.0, s]
        color = ORACLE_COLOR if m == "oracle_B" else PALETTE[i % len(PALETTE)]
        n = int(onset.sum()) if given_onset else len(rs)
        ax.step(t, s, where="post", color=color, lw=1.8,
                ls="--" if m == "oracle_B" else "-", label=f"{m} (n={n})")
    ax.set_xlim(0, max_frames)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("frames after the event")
    ax.set_ylabel("share not yet recovered" + (" (given onset)" if given_onset else ""))
    ax.set_title(react_label(lead, distortion, info))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def outcome_profile(result, lead: int, info: float, distortion: float = 0.0,
                    models: Optional[Sequence[str]] = None, save: Optional[str] = None):
    """Anticipation, censoring given onset and three-frame relapse per model (bars)."""
    plt = _require_mpl()
    summary = result.summary()
    rows = {k[3]: v for k, v in summary.items()
            if k[0] == lead and np.isclose(k[1], info) and np.isclose(k[2], distortion)}
    names = [m for m in (models or sorted(rows)) if m in rows and m != "oracle_B"]
    measures = [
        ("anticipation", lambda v: v["anticipation"]),
        ("censored given onset",
         lambda v: v["censored"] / (v["recovered"] + v["censored"])
         if (v["recovered"] + v["censored"]) > 0 else np.nan),
        ("relapse (3 frames)", lambda v: v["relapse3"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.8 / max(len(names), 1)
    x = np.arange(len(measures))
    for i, m in enumerate(names):
        vals = [f(rows[m]) for _, f in measures]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=m,
               color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, [name for name, _ in measures])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of events")
    ax.set_title(react_label(lead, distortion, info))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig
