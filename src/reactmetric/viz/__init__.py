"""Visualization suite (extras=[viz])."""

from .plots import (
    dataset_trajectory_overview,
    event_detail,
    excess_react_distribution,
    forecast_performance_summary,
    info_score_histograms,
    km_recovery_curves,
    plot_calibration,
    react_by_event_type,
    task_react_km_curves,
    three_part_summary,
)

__all__ = [
    "km_recovery_curves",
    "dataset_trajectory_overview",
    "info_score_histograms",
    "react_by_event_type",
    "excess_react_distribution",
    "event_detail",
    "task_react_km_curves",
    "forecast_performance_summary",
    "three_part_summary",
    "plot_calibration",
]
