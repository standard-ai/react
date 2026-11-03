"""Events: the regime changes REACT measures recovery from (paper section 3.4).

The default source is ``BayesianChangepoints()``, the paper's detector. Any object with
``detect(traj) -> List[Event]`` can supply events instead: ``from_times`` for labelled or
externally defined events, ``RupturesDetector`` for a ruptures pipeline, ``select`` to keep
only some types. Event types: "stop", "start", "speed_change", "turn", "both" (a turn and a
speed change together), and rarely "turn_change".
"""

from .base import (
    Event,
    EventSource,
    FixedEvents,
    SelectedEvents,
    events_by_track,
    flatten_events,
    from_times,
    select,
)
from .bayes import TUNED_SG, BayesianChangepoints, fit_hazard, relabel_from_headings
from .kinematic import KinematicFeatures, SegmentationConfig
from .plugins import RupturesDetector

__all__ = [
    "Event",
    "EventSource",
    "FixedEvents",
    "from_times",
    "select",
    "BayesianChangepoints",
    "RupturesDetector",
    "fit_hazard",
]
