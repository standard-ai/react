"""Plotting suite (extras=[viz]).

Kaplan-Meier recovery curves, information-score histograms, REACT-by-event-type,
excess-REACT distributions, a per-event detail view, the three-part summary
figure, dataset trajectory overview, task-REACT KM curves, forecast performance
summary, and a calibration diagnostic plot. All functions accept ``save`` and
return the matplotlib Figure.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np

SAI_GREEN = "#00B894"
SAI_LIGHT_BLUE = "#4FC3F7"
SAI_PINK = "#FF5DA2"
ORACLE_COLOR = "#7f8c8d"

_MODEL_COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#e67e22", "#1abc9c"]
_MODEL_COLORS_DARK = [SAI_GREEN, SAI_LIGHT_BLUE, "#6C5CE7", SAI_PINK, "#F2B705", "#16C79A"]

_DATASET_BG = "#0B1622"
_DATASET_FG = "#E6ECF2"
_DATASET_SPINE = "#22303C"
_DATASET_MUTED = "#5A6B78"
_DATASET_LABEL = "#8A99A6"

_PLOT_THEMES = {
    "light": {
        "fig_bg": "white",
        "ax_bg": "white",
        "panel_bg": "#fafafa",
        "fg": "#1a1a1a",
        "muted": "#5A6672",
        "spine": "#cccccc",
        "grid": "#b0b0b0",
        "axis_line": "#333333",
        "accent_line": "#2c3e50",
        "box_bg": "white",
        "cycle": _MODEL_COLORS,
        "model_cycle": _MODEL_COLORS,
    },
    "dark": {
        "fig_bg": "#0B1622",
        "ax_bg": "#0B1622",
        "panel_bg": "#0E1A28",
        "fg": "#E6ECF2",
        "muted": "#7C8C99",
        "spine": "#22303C",
        "grid": "#1B2A38",
        "axis_line": "#8A99A6",
        "accent_line": SAI_LIGHT_BLUE,
        "box_bg": "#0E1A28",
        "cycle": _MODEL_COLORS_DARK,
        "model_cycle": _MODEL_COLORS_DARK,
    },
}


def _plot_theme() -> str:
    name = os.environ.get("REACT_PLOT_THEME", "dark").strip().lower()
    return name if name in _PLOT_THEMES else "dark"


def _theme_color(key: str) -> str:
    return _PLOT_THEMES[_plot_theme()][key]


def _model_colors() -> Sequence[str]:
    return _PLOT_THEMES[_plot_theme()]["model_cycle"]


def _apply_plot_theme(plt) -> None:
    """Apply the paper-style light/dark theme to matplotlib rcParams."""
    from cycler import cycler

    theme = _PLOT_THEMES[_plot_theme()]
    updates = {
        "figure.facecolor": theme["fig_bg"],
        "figure.edgecolor": theme["fig_bg"],
        "axes.facecolor": theme["ax_bg"],
        "axes.edgecolor": theme["spine"],
        "axes.labelcolor": theme["fg"],
        "axes.titlecolor": theme["fg"],
        "text.color": theme["fg"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "xtick.labelcolor": theme["fg"],
        "ytick.labelcolor": theme["fg"],
        "grid.color": theme["grid"],
        "legend.facecolor": theme["box_bg"],
        "legend.edgecolor": theme["spine"],
        "savefig.facecolor": theme["fig_bg"],
        "savefig.edgecolor": theme["fig_bg"],
        "axes.prop_cycle": cycler(color=theme["cycle"]),
    }
    for key, value in updates.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value


def _sai_cyclic_cmap():
    """Cyclic brand colormap (endpoints match so headings wrap seamlessly)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "sai_cyclic",
        [SAI_GREEN, SAI_LIGHT_BLUE, "#6C5CE7", SAI_PINK, "#F2B705", SAI_GREEN],
    )


def _require_mpl():
    try:
        import matplotlib.pyplot as plt

        _apply_plot_theme(plt)
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install 'reactmetric[viz]'") from e


def _lead_from_results(results: Dict[str, object], lead: Optional[int] = None) -> Optional[int]:
    if lead is not None:
        return int(lead)
    for res in results.values():
        rlead = getattr(res, "lead", None)
        if rlead is not None:
            return int(rlead)
    return None


def km_recovery_curves(
    results: Dict[str, object],
    save: Optional[str] = None,
    conditional: bool = True,
    lead: Optional[int] = None,
):
    from ..constants import react_at_lead, react_frames_ylabel

    plt = _require_mpl()
    lead = _lead_from_results(results, lead)
    label = react_at_lead(lead)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in results.items():
        times, surv = res.km_curve(conditional=conditional)
        ax.step(times, surv, where="post", label=f"{name} (median {res.km_median():.0f})")
    ax.set_xlabel(f"{label} (frames since event)")
    ax.set_ylabel("P(not yet recovered)")
    ax.set_title(f"Kaplan-Meier recovery curves ({label})")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def info_score_histograms(traces: Dict[str, np.ndarray], save: Optional[str] = None):
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, s in traces.items():
        s = np.asarray(s)
        ax.hist(s[np.isfinite(s)], bins=60, histtype="step", density=True, label=name)
    ax.axvline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel("information score s (nats)")
    ax.set_ylabel("density")
    ax.set_title("Steady-state information score")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def react_by_event_type(result, save: Optional[str] = None):
    from ..constants import react_at_lead, react_frames_ylabel

    plt = _require_mpl()
    by = result.by_event_type()
    types = list(by.keys())
    medians = [by[t]["median_react_frames"] for t in types]
    label = react_at_lead(getattr(result, "lead", None))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(types, medians)
    ax.set_ylabel(react_frames_ylabel(getattr(result, "lead", None)))
    ax.set_title(f"{label} by event type -- {result.model_name}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def excess_react_distribution(results: Dict[str, object], save: Optional[str] = None):
    from ..constants import react_at_lead

    plt = _require_mpl()
    lead = _lead_from_results(results)
    label = react_at_lead(lead)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in results.items():
        vals = [m.excess_react for m in res.measurements if np.isfinite(m.excess_react)]
        if vals:
            ax.hist(vals, bins=40, histtype="step", density=True, label=name)
    ax.axvline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel(f"excess {label} vs oracle floor (frames)")
    ax.set_ylabel("density")
    ax.set_title(f"Excess {label} (model - oracle)")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def event_detail(result, event, traj, ref=None, lead: int = 10, save: Optional[str] = None):
    from ..constants import react_at_lead

    plt = _require_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    pos = traj.positions
    ax1.plot(pos[:, 0], pos[:, 1], "-", color="0.7")
    ax1.scatter([pos[event.t0, 0]], [pos[event.t0, 1]], color="r", zorder=5, label="event t0")
    ax1.set_title(f"track {traj.id} -- {event.event_type}")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend()
    m = next(
        (m for m in result.measurements if m.id == traj.id and int(m.t0) == int(event.t0)),
        None,
    )
    rlead = getattr(result, "lead", None)
    if rlead is None:
        rlead = lead
    label = react_at_lead(rlead)
    txt = "no measurement"
    if m is not None:
        txt = (f"{label}={m.react_frames:.0f} frames\nanticipated={m.anticipated}\n"
               f"censored={m.censored}")
    ax2.axis("off")
    ax2.text(0.05, 0.5, txt, fontsize=12, va="center")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def dataset_trajectory_overview(
    data,
    events=None,
    max_tracks: int = 40,
    save: Optional[str] = None,
):
    """Overview of trajectories with detected/known event locations.

    Useful as a quick sanity check that the dataset and event detector are
    producing plausible disruptions before interpreting REACT numbers.
    """
    plt = _require_mpl()
    from matplotlib.collections import LineCollection

    ids = data.ids[:max_tracks]
    tracks = [data[tid] for tid in ids if data[tid].T >= 2 and data[tid].D >= 2]
    if not tracks:
        raise ValueError("dataset_trajectory_overview requires at least one 2D trajectory")

    cyc = _sai_cyclic_cmap()
    segments = []
    colors = []
    all_points = []
    for traj in tracks:
        pos = traj.positions[:, :2]
        segments.append(pos)
        all_points.append(pos)
        displacement = pos[-1] - pos[0]
        hue = (np.arctan2(displacement[1], displacement[0]) + np.pi) / (2.0 * np.pi)
        colors.append(cyc(hue)[:3])

    pts = np.vstack(all_points)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    pad = 0.03 * (mx - mn) + 0.5

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    fig.patch.set_facecolor(_DATASET_BG)
    ax.set_facecolor(_DATASET_BG)
    ax.add_collection(
        LineCollection(
            segments,
            colors=colors,
            linewidths=2.2,
            alpha=0.035,
            capstyle="round",
        )
    )
    ax.add_collection(
        LineCollection(
            segments,
            colors=colors,
            linewidths=0.65,
            alpha=0.22,
            capstyle="round",
        )
    )
    for tid in ids:
        traj = data[tid]
        pos = traj.positions
        if events is None or traj.D < 2:
            continue
        for event in events.get(tid, []):
            if 0 <= event.t0 < traj.T:
                ax.scatter(
                    pos[event.t0, 0],
                    pos[event.t0, 1],
                    s=54,
                    facecolors="none",
                    edgecolors=_DATASET_FG,
                    linewidths=1.2,
                    alpha=0.85,
                    zorder=4,
                )
    ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
    ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", color=_DATASET_LABEL, fontsize=9)
    ax.set_ylabel("y (m)", color=_DATASET_LABEL, fontsize=9)
    ax.set_title(
        f"Synthetic regime-change trajectories · {len(tracks):,} tracks",
        color=_DATASET_FG,
        fontsize=14,
        pad=12,
    )
    ax.tick_params(colors=_DATASET_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_DATASET_SPINE)
    if events is not None:
        ax.scatter([], [], s=54, facecolors="none", edgecolors=_DATASET_FG, label="event t0")
        leg = ax.legend(loc="best")
        leg.get_frame().set_facecolor("#0E1A28")
        leg.get_frame().set_edgecolor(_DATASET_SPINE)
        for text in leg.get_texts():
            text.set_color(_DATASET_FG)
    sm = plt.cm.ScalarMappable(cmap=cyc, norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08)
    cbar.set_label("direction of travel", color=_DATASET_FG, fontsize=10)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["W", "S", "E", "N", "W"])
    cbar.ax.xaxis.set_tick_params(color=_DATASET_MUTED, labelcolor=_DATASET_FG)
    cbar.outline.set_edgecolor(_DATASET_SPINE)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=170, facecolor=_DATASET_BG, bbox_inches="tight")
    return fig


def task_react_km_curves(
    forecasters: Dict[str, object],
    data,
    ref,
    events,
    lead: int = 10,
    distortion_m: float = 0.5,
    config=None,
    save: Optional[str] = None,
):
    """KM curves for task-specific REACT.

    Sets epsilon to the rate-distortion threshold required to localize at RMS
    error ``distortion_m`` and evaluates each model under that stricter task.
    """
    from ..config import ReactConfig
    from ..constants import react_at_lead
    from ..forecasters.adapters import as_forecaster
    from ..metric.evaluate import evaluate

    plt = _require_mpl()
    config = config or ReactConfig(lead=lead)
    eps_star = float(ref.eps_star(lead, distortion_m))
    results = {}
    for name, model in forecasters.items():
        results[name] = evaluate(
            as_forecaster(model, name=name),
            data,
            ref,
            events,
            lead=lead,
            epsilon=eps_star,
            config=config,
            name=name,
        )

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (name, res) in enumerate(results.items()):
        times, surv = res.km_curve(conditional=False)
        if len(times) == 0:
            continue
        color = _model_colors()[i % len(_model_colors())]
        ax.step(
            np.append(times, times[-1] + 5),
            np.append(surv, surv[-1]),
            where="post",
            label=f"{name} (anticipated {100 * res.anticipation_rate:.0f}%)",
            color=color,
            linewidth=2,
        )
    ax.set_xlabel("Forecast issue frames since maneuver onset $T_0$", fontsize=12)
    ax.set_ylabel("P(not yet task-recovered)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"{react_at_lead(lead)}({distortion_m:g}m)  [ε={eps_star:.2f} nats] "
        "(Kaplan-Meier, censoring-aware)\n"
        "x-axis is forecast issue time; target is issue + h - 1; "
        "S(0) < 1 is anticipation",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def three_part_summary(report, accuracy: Optional[Dict] = None, save: Optional[str] = None):
    """The motivating figure: accuracy, recovery curves, and excess-REACT together."""
    from ..constants import react_at_lead, react_frames_ylabel

    plt = _require_mpl()
    lead = getattr(report, "lead", None)
    label = react_at_lead(lead)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (1) ADE/FDE accuracy
    ax = axes[0]
    if accuracy:
        names = list(accuracy.keys())
        ade = [accuracy[n]["ade_m"] for n in names]
        ax.bar(names, ade)
        ax.set_ylabel("ADE (m)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Accuracy (ADE)")

    # (2) KM recovery
    ax = axes[1]
    for name, res in report.results.items():
        times, surv = res.km_curve(conditional=True)
        ax.step(times, surv, where="post", label=name)
    ax.set_xlabel(react_frames_ylabel(lead))
    ax.set_ylabel("P(not recovered)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Adaptation ({label})")
    ax.legend()

    # (3) excess REACT
    ax = axes[2]
    for name, res in report.results.items():
        vals = [m.excess_react for m in res.measurements if np.isfinite(m.excess_react)]
        if vals:
            ax.hist(vals, bins=30, histtype="step", density=True, label=name)
    ax.axvline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel(f"excess {label} (frames)")
    ax.set_title(f"Excess vs oracle floor ({label})")
    fig.suptitle(f"{label} summary", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def forecast_performance_summary(
    accuracy: Dict[str, Dict[str, object]],
    leads: Optional[Sequence[int]] = None,
    save: Optional[str] = None,
):
    """ADE/FDE and by-lead displacement/NLL summary (not REACT).

    Mirrors the old research-pipeline diagnostic so quickstart can show the
    conventional accuracy axis next to REACT's adaptation axis.
    """
    plt = _require_mpl()
    names = list(accuracy.keys())
    if leads is None and names:
        leads = sorted(accuracy[names[0]].get("displacement_error_by_lead_m", {}).keys())
    leads = tuple(leads or ())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    x = np.arange(len(names))

    aggregate_metrics = [
        ("ade_m", "Mean ADE (m)"),
        ("fde_m", "Mean FDE (m)"),
        ("nll_mean", "Mean NLL (nats)"),
    ]
    for ax, (key, title) in zip(axes[0], aggregate_metrics):
        if key == "nll_mean":
            vals = [
                float(np.nanmean(list(accuracy[name].get("nll_by_lead", {}).values())))
                for name in names
            ]
        else:
            vals = [accuracy[name][key] for name in names]
        colors = [_model_colors()[i % len(_model_colors())] for i in range(len(names))]
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    lead_metrics = [
        ("displacement_error_by_lead_m", "ADE@h (m)"),
        ("displacement_error_by_lead_m", "FDE@h (m)"),
        ("nll_by_lead", "NLL@h (nats)"),
    ]
    for ax, (key, title) in zip(axes[1], lead_metrics):
        for i, name in enumerate(names):
            by_lead = accuracy[name].get(key, {})
            vals = [by_lead.get(lead, np.nan) for lead in leads]
            ax.plot(
                leads,
                vals,
                marker="o",
                linewidth=1.8,
                color=_model_colors()[i % len(_model_colors())],
                label=name,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("lead h (frames)")
        ax.grid(True, alpha=0.25)
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Held-Out Rolling Forecast Accuracy", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def plot_calibration(report, save: Optional[str] = None):
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    ratio = report.coverage_chi2 / max(report.dim, 1)
    ax.bar(["coverage ratio"], [ratio], color="C1" if not report.is_calibrated else "C0")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="calibrated")
    ax.set_ylabel("mean Mahalanobis^2 / D")
    ax.set_title(f"Calibration: {report.model_name} "
                 f"({'OK' if report.is_calibrated else 'MISCALIBRATED'})")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig
