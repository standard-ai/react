"""Events: the regime changes REACT measures recovery from (paper section 3.4).

The default source is ``BayesianChangepoints.tuned_sg()``. Any object with
``detect(traj) -> List[Event]`` can supply events instead: ``from_times`` for labelled or
externally defined events, ``RupturesDetector`` for a ruptures pipeline.
"""

from .base import (
    Event,
    EventSource,
    FixedEvents,
    events_by_track,
    flatten_events,
    from_times,
)
from .bayes import TUNED_SG, BayesianChangepoints, fit_hazard, relabel_from_headings
from .kinematic import KinematicFeatures, SegmentationConfig
from .plugins import RupturesDetector

__all__ = [
    "Event",
    "EventSource",
    "FixedEvents",
    "from_times",
    "BayesianChangepoints",
    "RupturesDetector",
    "fit_hazard",
]
