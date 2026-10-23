import numpy as np

import reactmetric as rm
from reactmetric.events import BayesianChangepoints, SegmentationConfig
from reactmetric.events.base import EventSource, from_times


def test_detector_is_event_source():
    assert isinstance(BayesianChangepoints.tuned_sg(), EventSource)


def test_default_config_is_the_low_preset_the_paper_used():
    cfg = SegmentationConfig()
    assert (cfg.speed_change_z, cfg.turn_rate_z, cfg.sharp_turn_rate) == (2.5, 2.0, 1.25)
    assert BayesianChangepoints.tuned_sg().config.velocity_window == 1


def test_from_times_event_source():
    src = from_times({0: [10, 20], 1: [15]})
    ts = rm.TrajectorySet.from_arrays({0: np.zeros((30, 2)), 1: np.zeros((30, 2))})
    detected = src.detect_set(ts)
    assert [e.t0 for e in detected[0]] == [10, 20]
    assert [e.t0 for e in detected[1]] == [15]
