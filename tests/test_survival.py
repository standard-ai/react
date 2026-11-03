import numpy as np
import pytest

from reactmetric.metric.survival import kaplan_meier, km_median


def test_km_no_censoring_is_ecdf_complement():
    durations = np.array([1.0, 2.0, 2.0, 3.0])
    observed = np.array([True, True, True, True])
    times, surv = kaplan_meier(durations, observed)
    # S(1)=3/4, S(2)=1/4, S(3)=0
    assert surv[np.searchsorted(times, 1.0)] == 0.75
    assert surv[np.searchsorted(times, 2.0)] == 0.25
    assert surv[-1] == 0.0


def test_km_with_censoring_keeps_mass():
    durations = np.array([1.0, 2.0, 3.0])
    observed = np.array([True, False, True])  # middle censored
    times, surv = kaplan_meier(durations, observed)
    # at t=1: 1 event of 3 at risk -> S=2/3; t=2 censored (no drop);
    # t=3: 1 of 1 at risk -> S=0
    assert surv[np.searchsorted(times, 1.0)] == pytest.approx(2.0 / 3.0)
    assert surv[-1] == 0.0


def test_km_median():
    times, surv = kaplan_meier(np.arange(1, 11.0), np.ones(10, dtype=bool))
    assert km_median(times, surv) == 5.0


def test_weighted_km_matches_kaplan_meier():
    from reactmetric.results import _wkm

    rng = np.random.default_rng(0)
    d = rng.integers(0, 40, 200).astype(float)
    obs = rng.random(200) < 0.7
    w = rng.integers(0, 3, 200)
    t1, s1 = kaplan_meier(np.repeat(d, w), np.repeat(obs, w))
    t2, s2 = _wkm(d, obs, w.astype(float))
    assert np.allclose(t1, t2) and np.allclose(s1, s2)
