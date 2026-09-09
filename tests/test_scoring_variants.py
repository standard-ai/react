import numpy as np

import reactmetric as rm
from reactmetric.scoring.crps import gaussian_crps, gaussian_energy_score


def test_crps_minimized_at_zero_residual():
    cov = np.eye(2)
    c0 = gaussian_crps(np.zeros((1, 2)), cov)[0]
    c1 = gaussian_crps(np.array([[2.0, 0.0]]), cov)[0]
    assert c0 < c1


def test_energy_score_increases_with_error():
    cov = np.eye(2)
    e0 = gaussian_energy_score(np.zeros((1, 2)), cov, n_samples=400, seed=0)[0]
    e1 = gaussian_energy_score(np.array([[3.0, 0.0]]), cov, n_samples=400, seed=0)[0]
    assert e0 < e1


def test_react_runs_under_all_scorings(synthetic):
    cv = rm.baselines.ConstantVelocity()
    out = {}
    for sc in ("nll", "crps", "energy"):
        res = rm.evaluate(cv, synthetic["test"], synthetic["ref"], synthetic["events"],
                          lead=10, scoring=sc)
        assert res.n_measured > 0
        out[sc] = res.anticipation_rate
    assert all(0.0 <= v <= 1.0 for v in out.values())
