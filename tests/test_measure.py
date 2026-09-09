import numpy as np

from reactmetric.config import ReactConfig
from reactmetric.events.base import Event
from reactmetric.metric.measure import first_sustained_run, measure_event_on_trace


def test_first_sustained_run():
    mask = np.array([0, 1, 1, 0, 1, 1, 1], dtype=bool)
    assert first_sustained_run(mask, 3, 0, len(mask)) == 4
    assert first_sustained_run(mask, 2, 0, len(mask)) == 1
    assert first_sustained_run(mask, 4, 0, len(mask)) is None


def _measure(trace, t0=20, cfg=None):
    cfg = cfg or ReactConfig(
        onset_consecutive=3, recovery_consecutive=3, min_post_coverage=10, max_react_frames=100
    )
    frames = np.arange(len(trace))
    ev = Event(id=0, t0=t0, event_type="x")
    # trace is the info score s; the measurer takes -s with tau_onset=0
    return measure_event_on_trace(frames, -np.asarray(trace, float), ev, "m", 0.0, 0.0, cfg)


def test_recovery_detected():
    s = np.zeros(80)
    s[:21] = 2.0          # informative before event
    s[21:31] = -1.0       # onset (drops below ref) for 10 frames
    s[31:] = 2.0          # recovers
    m = _measure(s, t0=20)
    assert m is not None
    assert not m.anticipated
    assert not m.censored
    assert m.react_frames == 11  # recovers at frame 31 (t0=20)


def test_anticipated_when_no_onset():
    s = np.full(80, 2.0)
    m = _measure(s, t0=20)
    assert m.anticipated
    assert m.react_frames == 0.0


def test_censored_when_never_recovers():
    s = np.full(80, 2.0)
    s[21:] = -1.0
    m = _measure(s, t0=20)
    assert not m.anticipated
    assert m.censored
