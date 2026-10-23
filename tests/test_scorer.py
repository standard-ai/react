"""ReactScorer end to end on small synthetic walk-stop-walk tracks (one lead, one level)."""

import numpy as np
import pytest

from reactmetric import baselines
from reactmetric.core.trajectory import Trajectory, TrajectorySet
from reactmetric.scorer import ReactScorer


def _tracks(n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        v = np.r_[np.full(200, rng.uniform(0.9, 1.4)), np.zeros(150), np.full(200, 1.1)]
        ang = rng.uniform(0, 2 * np.pi)
        step = v[:, None] * 0.1 * np.array([np.cos(ang), np.sin(ang)])
        pos = np.cumsum(step, 0) + 0.01 * rng.standard_normal((len(v), 2))
        out[f"s{seed}_{i}"] = Trajectory(id=f"s{seed}_{i}", positions=pos, dt=0.1)
    return TrajectorySet(out)


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    calib = _tracks(30, seed=1)
    return ReactScorer.fit(calib, info=(4.2,), leads=(5,), distortions=(0.3,),
                           table_dir=str(tmp_path_factory.mktemp("tables")))


def test_fit_records_its_assumptions(scorer):
    assert scorer.noise_floor["floor"] == "A" and 0.005 < scorer.sigma < 0.02
    assert [lv.info for lv in scorer.levels[5]] == [4.2]
    cal = scorer.calibration[5]
    assert cal["calm_frames"] > 0 and 0.0 <= cal["fits"]["4.200"]["scale"] <= 1.0


def test_evaluate_scores_models_and_oracle(scorer):
    test = _tracks(3, seed=2)
    result = scorer.evaluate({"CV": baselines.ConstantVelocity()}, test)
    models = {r["model"] for r in result.records}
    assert models == {"CV", "oracle_B"}
    assert {r["distortion"] for r in result.records} == {0.0, 0.3}
    assert all(r["lead"] == 5 and r["info"] == 4.2 for r in result.records)
    statuses = {r["status"] for r in result.records if r["model"] == "CV"}
    assert statuses <= {"recovered", "censored", "better", "indistinguishable"}
    # Oracle B is never behind the climatology
    assert all(r["onset"] is None for r in result.records if r["model"] == "oracle_B")
    summary = result.summary()
    row = summary[(5, 4.2, 0.0, "CV")]
    assert row["events"] >= 3 and 0.0 <= row["anticipation"] <= 1.0


def test_precomputed_forecasts_with_unknown_ids_raise(scorer):
    test = _tracks(2, seed=3)
    cv = baselines.ConstantVelocity()
    preds = {f"other_{t}": cv.predict(test[t]) for t in test.ids}
    with pytest.raises(ValueError, match="match no test track"):
        scorer.evaluate({"CV": preds}, test)


def test_precomputed_forecasts_match_a_forecaster(scorer):
    test = _tracks(2, seed=4)
    cv = baselines.ConstantVelocity()
    a = scorer.evaluate({"CV": cv}, test).records
    b = scorer.evaluate({"CV": {t: cv.predict(test[t]) for t in test.ids}}, test).records
    assert a == b


def test_one_component_mixture_scores_like_its_gaussian(scorer):
    from reactmetric.core.predictive import MixturePredictive

    test = _tracks(2, seed=5)
    cv = baselines.ConstantVelocity()
    gauss, mix = {}, {}
    for t in test.ids:
        p = cv.predict(test[t])
        gauss[t] = p
        mix[t] = MixturePredictive(p.issue_frames, np.ones((len(p), 1)), p.means[:, :, None],
                                   p.covs[:, :, None])
    a = scorer.evaluate({"m": gauss}, test).records
    b = scorer.evaluate({"m": mix}, test).records
    key = ("model", "track", "t0", "distortion", "status", "onset", "react", "relapse3")
    assert [tuple(r[k] for k in key) for r in a] == [tuple(r[k] for k in key) for r in b]
