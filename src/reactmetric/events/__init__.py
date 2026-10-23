"""Events: the regime changes REACT measures recovery from."""

from .base import (
    Event,
    EventSource,
    FixedEvents,
    events_by_track,
    flatten_events,
    from_times,
)

__all__ = [
    "Event",
    "EventSource",
    "FixedEvents",
    "from_times",
    "events_by_track",
    "flatten_events",
]

# Optional detectors are imported lazily to avoid hard dependencies.
try:  # built-in kinematic changepoint detector
    from .kinematic import (  # noqa: F401
        EVENT_SENSITIVITY_PRESETS,
        KinematicChangepoints,
        SegmentationConfig,
        segmentation_config,
    )

    __all__ += [
        "KinematicChangepoints",
        "SegmentationConfig",
        "EVENT_SENSITIVITY_PRESETS",
        "segmentation_config",
    ]
except Exception:  # pragma: no cover
    pass

try:  # Bayesian changepoints: the default event source (paper section 3.4)
    from .bayes import (  # noqa: F401
        TUNED_SG,
        BayesianChangepoints,
        fit_hazard,
        relabel_from_headings,
    )

    __all__ += ["BayesianChangepoints", "TUNED_SG", "fit_hazard", "relabel_from_headings"]
except Exception:  # pragma: no cover
    pass

try:  # ruptures plugin (extras=[cpd])
    from .plugins import RupturesDetector  # noqa: F401

    __all__ += ["RupturesDetector"]
except Exception:  # pragma: no cover
    pass
