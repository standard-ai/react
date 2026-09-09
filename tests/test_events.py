import numpy as np

import reactmetric as rm
from reactmetric.events import KinematicChangepoints, SegmentationConfig, segmentation_config
from reactmetric.events.base import EventSource, from_times


def test_detector_is_event_source():
    det = KinematicChangepoints()
    assert isinstance(det, EventSource)


def test_detector_finds_a_sharp_turn():
    # straight, then a 90-degree turn at frame 40
    seg1 = np.cumsum(np.tile([0.1, 0.0], (40, 1)), axis=0)
    start = seg1[-1]
    seg2 = start + np.cumsum(np.tile([0.0, 0.1], (40, 1)), axis=0)
    pos = np.vstack([seg1, seg2])
    traj = rm.Trajectory(pos, dt=0.1, id=0)
    det = KinematicChangepoints(SegmentationConfig(penalty=5.0, min_size=8))
    events = det.detect(traj)
    assert len(events) >= 1
    assert any(abs(e.t0 - 40) <= 10 for e in events)
    assert all(np.isfinite(e.detection_floor_frames) for e in events)


def test_from_times_event_source():
    src = from_times({0: [10, 20], 1: [15]})
    ts = rm.TrajectorySet.from_arrays({0: np.zeros((30, 2)), 1: np.zeros((30, 2))})
    detected = src.detect_set(ts)
    assert [e.t0 for e in detected[0]] == [10, 20]
    assert [e.t0 for e in detected[1]] == [15]


def test_calibrate_penalty_reduces_detections():
    data, _ = rm.datasets.synthetic_regime_changes(n=40, seed=4)
    det = KinematicChangepoints(SegmentationConfig(penalty=2.0))
    n_low = sum(len(det.detect(t)) for t in data)
    det.calibrate_penalty(data, target_false_rate=0.005)
    n_cal = sum(len(det.detect(t)) for t in data)
    assert n_cal <= n_low


def test_sensitivity_presets_reduce_detections():
    data, _ = rm.datasets.synthetic_regime_changes(n=30, seed=5)
    high = KinematicChangepoints.from_preset("high")
    low = KinematicChangepoints.from_preset("low")
    n_high = sum(len(high.detect(t)) for t in data)
    n_low = sum(len(low.detect(t)) for t in data)
    assert n_low <= n_high
    assert segmentation_config("low").min_size > segmentation_config("high").min_size


def test_calibrate_events_per_track_reduces_detections():
    data, _ = rm.datasets.synthetic_regime_changes(n=30, seed=6)
    det = KinematicChangepoints.from_preset("high")
    n_high = sum(len(det.detect(t)) for t in data)
    det.calibrate_events_per_track(data, target_events_per_track=0.5)
    n_cal = sum(len(det.detect(t)) for t in data)
    assert n_cal <= n_high
