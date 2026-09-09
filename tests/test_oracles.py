import numpy as np

from reactmetric.oracles import (
    ClimatologySeeded,
    Hindsight,
    analytic_reset_recovery_frames,
    filter_artifacts,
    linear_extrapolation_variance,
    oracle_c_react,
)


def test_oracle_b_info_score_zero_at_event(synthetic):
    """Oracle B knows nothing post-change at t0, so s == 0 there."""
    ob = ClimatologySeeded(horizon=15, history=10, sigma_r=0.05)
    tid = synthetic["test"].ids[0]
    traj = synthetic["test"][tid]
    evs = synthetic["events"][tid]
    if not evs:
        return
    frames, s = ob.info_trace(traj.positions, synthetic["ref"], lead=10, t0=evs[0].t0)
    if len(frames) == 0:
        return
    # at the issue frame == t0, the oracle has < 2 post-change samples -> s == 0
    at_t0 = np.isclose(frames, evs[0].t0)
    if at_t0.any():
        assert np.allclose(s[at_t0], 0.0)


def test_oracle_c_react_adds_floor():
    assert oracle_c_react(5.0, 3.0) == 8.0
    assert oracle_c_react(5.0, np.inf) == 5.0


def test_linear_extrapolation_variance_decreases_with_samples():
    v_few = linear_extrapolation_variance(3, 10, 0.1)
    v_many = linear_extrapolation_variance(30, 10, 0.1)
    assert v_many < v_few
    assert analytic_reset_recovery_frames(0.1, 15, nll_band=2.0) >= 2


def test_filter_artifacts_splits(synthetic):
    genuine, artifacts = filter_artifacts(
        synthetic["events"], synthetic["test"], synthetic["ref"],
        oracle=Hindsight(horizon=15), lead=10,
    )
    n_g = sum(len(v) for v in genuine.values())
    n_a = sum(len(v) for v in artifacts.values())
    # synthetic data is coherent, so most events should be genuine
    assert n_g > 0
    assert n_g >= n_a
