import warnings

import numpy as np

import reactmetric as rm


def test_benchmark_runs_with_baselines_and_oracle(synthetic):
    cv = rm.baselines.ConstantVelocity()
    preds = {tid: cv.predict(synthetic["test"][tid]) for tid in synthetic["test"].ids}
    preds = {k: v for k, v in preds.items() if v is not None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = rm.benchmark(
            forecasts={"CV-pre": preds}, data=synthetic["test"], calibration=synthetic["calib"],
            baselines=["CTRV"], oracle="C", lead=10,
        )
    rows = rep.rows()
    names = {r["model"] for r in rows}
    assert {"CV-pre", "CTRV"} <= names
    # excess REACT defined for at least some events
    assert any(np.isfinite(r["mean_excess_react"]) for r in rows)
    d = rep.to_dict()
    assert "oracle_B" in d["models"] or "oracle_C" in d["models"]


def test_lead_and_epsilon_sweeps(synthetic):
    cv = rm.baselines.ConstantVelocity()
    ls = rm.lead_sweep(cv, synthetic["test"], synthetic["ref"], synthetic["events"],
                       leads=(5, 10, 15))
    assert set(ls["ConstantVelocity"].keys()) == {5, 10, 15}
    es = rm.epsilon_sweep(cv, synthetic["test"], synthetic["ref"], synthetic["events"],
                          epsilons=(0.0, 0.5), lead=10)
    assert set(es["ConstantVelocity"].keys()) == {0.0, 0.5}
    # higher epsilon is weakly harder -> failure rate non-decreasing
    f0 = es["ConstantVelocity"][0.0]["failure_rate"]
    f1 = es["ConstantVelocity"][0.5]["failure_rate"]
    assert f1 >= f0 - 1e-9


def test_rate_distortion_and_task_react(synthetic):
    anchors = rm.rate_distortion_anchors(synthetic["ref"], lead=10, distortion_targets_m=(0.9, 0.3))
    assert anchors[0.3] >= anchors[0.9]  # tighter task needs more information
    cv = rm.baselines.ConstantVelocity()
    tr = rm.task_react(cv, synthetic["test"], synthetic["ref"], synthetic["events"],
                       lead=10, distortion_m=0.5)
    assert "eps_star_nats" in tr["ConstantVelocity"]


def test_forecast_accuracy(synthetic):
    cv = rm.baselines.ConstantVelocity()
    acc = rm.forecast_accuracy(cv, synthetic["test"], leads=(1, 5, 10))
    a = acc["ConstantVelocity"]
    assert np.isfinite(a["ade_m"]) and np.isfinite(a["fde_m"])
    # error grows with lead
    e = a["displacement_error_by_lead_m"]
    assert e[10] >= e[1] - 1e-9
