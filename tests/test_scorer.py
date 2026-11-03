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
    cal = scorer.diagnostics[5]
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


def test_benchmark_adds_baselines_and_serializes(scorer, tmp_path):
    import json

    import reactmetric as rm

    calib, test = _tracks(10, seed=6), _tracks(2, seed=7)
    cv = baselines.ConstantVelocity()
    mine = {t: cv.predict(test[t]) for t in test.ids}
    result = rm.benchmark({"Mine": mine}, test, calib, baselines=["ConstantVelocity", "CTRV"],
                          scorer=scorer)
    assert {r["model"] for r in result.records} == {"Mine", "ConstantVelocity", "CTRV",
                                                    "oracle_B"}
    result.to_json(tmp_path / "r.json")
    payload = json.load(open(tmp_path / "r.json"))
    assert payload["settings"]["info_levels"] == {"5": [4.2]}
    assert any(row["model"] == "Mine" for row in payload["summary"])
    result.to_csv(tmp_path / "r.csv")
    assert (tmp_path / "r.csv").read_text().startswith("lead,info,distortion")
    with pytest.raises(ValueError, match="clash"):
        rm.benchmark({"CTRV": mine}, test, calib, baselines=["CTRV"], scorer=scorer)


def test_plots(scorer, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from reactmetric.viz import km_curves, outcome_profile, react_label

    result = scorer.evaluate({"CV": baselines.ConstantVelocity()}, _tracks(2, seed=8))
    km_curves(result, lead=5, info=4.2, save=str(tmp_path / "km.png"))
    km_curves(result, lead=5, info=4.2, given_onset=False)
    outcome_profile(result, lead=5, info=4.2, save=str(tmp_path / "bars.png"))
    assert (tmp_path / "km.png").exists() and (tmp_path / "bars.png").exists()
    assert react_label(10, 0.1, 4.2) == "REACT(10, 0.1 m; 4.2)"


# -- results: tables, intervals, files ---------------------------------------------------------


@pytest.fixture(scope="module")
def result(scorer):
    return scorer.evaluate({"CV": baselines.ConstantVelocity(), "CTRV": baselines.CTRV()},
                           _tracks(6, seed=9))


def test_table_spellings_and_loud_filters(result):
    base = result.table(lead=5)
    for d in (0, 0.0, float("inf"), "clim", "inf"):
        assert result.table(lead=5, distortion=d).equals(base)
    assert list(base["model"]) == ["CV", "CTRV", "oracle_B"]
    assert {"anticipated", "median_react", "rmst", "relapsed"} <= set(base.columns)
    assert base["distortion"].isna().all()
    assert set(result.table(lead=5, distortion="all")["distortion"].fillna(0)) == {0.0, 0.3}
    with pytest.raises(ValueError, match="no records with lead"):
        result.table(lead=7)
    with pytest.raises(ValueError, match="no records with distortion"):
        result.table(distortion=0.5)
    assert "rmst_cond" in result.table(columns="all").columns
    assert "only level" in result.table().attrs["info_reason"]


def test_models_repr_settings_and_breakdown(result):
    assert result.models == ["CV", "CTRV", "oracle_B"]
    assert "models=['CV', 'CTRV', 'oracle_B']" in repr(result)
    st = result.settings()
    assert st["window_frames"] == 150 and st["event_set"] == "shared"
    by = result.summary(by="event_type")
    assert all(len(k) == 5 for k in by)
    assert sum(v["events"] for k, v in by.items() if k[2] == 0.0 and k[3] == "CV") == \
        result.summary()[(5, 4.2, 0.0, "CV")]["events"]
    assert "event_type" in result.table(by="event_type").columns


def test_shared_event_set(result):
    events = {m: {(r["track"], r["t0"]) for r in result.records if r["model"] == m}
              for m in result.models}
    assert events["CV"] == events["CTRV"] == events["oracle_B"]
    assert result.coverage[5]["shared"] == len(events["CV"])


def test_ci_and_compare(result):
    out = result.ci("rmst", model="CV", lead=5, n_boot=200)
    assert out["low"] <= out["estimate"] <= out["high"]
    assert out["estimate"] == pytest.approx(result.table(lead=5).set_index("model").loc["CV",
                                                                                        "rmst"])
    same = result.compare("CV", "CV", lead=5, n_boot=100)
    assert same["difference"] == 0 and same["low"] == same["high"] == 0
    diff = result.compare("CV", "oracle_B", "anticipated", lead=5, n_boot=100)
    assert diff["low"] <= diff["difference"] <= diff["high"]
    df = result.table(lead=5, ci=True, n_boot=50)
    assert {"rmst_low", "rmst_high", "median_react_low"} <= set(df.columns)
    with pytest.raises(ValueError, match="statistic"):
        result.ci("speed", model="CV", lead=5)


def test_results_reload(result, tmp_path):
    from reactmetric import ReactResult

    result.to_json(tmp_path / "r.json")
    result.to_csv(tmp_path / "r.csv")
    for back in (ReactResult.load(str(tmp_path / "r.json")),
                 ReactResult.load(str(tmp_path / "r.csv"), settings=str(tmp_path / "r.json"))):
        assert back.window == 150 and back.models == result.models
        a, b = back.table(distortion="all"), result.table(distortion="all")
        assert a.drop(columns="median_react").equals(b.drop(columns="median_react"))
        assert np.allclose(a["median_react"], b["median_react"], equal_nan=True)
    assert ReactResult([]).table().empty


# -- scorer behaviour ----------------------------------------------------------------------------


def test_scorer_save_load_and_table_reuse(scorer, tmp_path):
    from reactmetric.scorer import ReactScorer

    scorer.save(str(tmp_path / "saved"))
    again = ReactScorer.load(str(tmp_path / "saved"))
    test = _tracks(2, seed=10)
    cv = {"CV": baselines.ConstantVelocity()}
    assert scorer.evaluate(cv, test).records == again.evaluate(cv, test).records
    refit = ReactScorer.fit(_tracks(30, seed=1), info=(4.2,), leads=(5,), distortions=(0.3,),
                            table_dir=scorer.table_dir)
    assert set(refit.table_seconds.values()) == {0.0}  # same fingerprint: reused


def test_nan_forecasts_count_as_missing(scorer):
    from reactmetric.core.predictive import Predictive

    test = _tracks(2, seed=11)
    cv = baselines.ConstantVelocity()
    preds = {}
    for t in test.ids:
        p = cv.predict(test[t])
        means = p.means.copy()
        means[::7] = np.nan
        preds[t] = Predictive(p.issue_frames, means, p.covs)
    recs = scorer.evaluate({"CV": preds}, test, event_set="all").records
    assert recs and all(np.isfinite(r["s_sum"]) for r in recs)


def test_errors_and_degenerate_forecasts_warn(scorer):
    class Broken:
        def predict(self, traj):
            raise RuntimeError("boom")

    test = _tracks(2, seed=12)
    with pytest.warns(RuntimeWarning, match="Broken.predict raised on 2 of 2 tracks.*boom"):
        scorer.evaluate({"Broken": Broken(), "CV": baselines.ConstantVelocity()}, test)


def test_fit_rejects_bad_settings():
    from reactmetric.reference import Climatology
    from reactmetric.scorer import ReactScorer

    calib = _tracks(6, seed=13)
    with pytest.raises(ValueError, match="positive"):
        ReactScorer.fit(calib, info=(0.0,), leads=(5,))
    with pytest.raises(ValueError, match="positive metres"):
        ReactScorer.fit(calib, info=(4.2,), leads=(5,), distortions=(0.0,))
    aniso = Climatology.fit(calib, leads=(5,), mode="kinematic")
    aniso.stop_cov[5] = np.diag([1e-4, 2e-4])
    with pytest.raises(ValueError, match="isotropic"):
        ReactScorer.fit(calib, info=(4.2,), leads=(5,), climatology=aniso)


def test_windows_are_settings(tmp_path_factory):
    from reactmetric.scorer import ReactScorer

    sc = ReactScorer.fit(_tracks(30, seed=1), info=(4.2,), leads=(5,), distortions=(),
                         window=60, min_window=10, onset_margin=2,
                         table_dir=str(tmp_path_factory.mktemp("w")))
    res = sc.evaluate({"CV": baselines.ConstantVelocity()}, _tracks(3, seed=2))
    assert res.settings()["window_frames"] == 60
    assert all(r["duration"] is None or r["duration"] <= 60 for r in res.records)
    assert all(r["onset"] is None or r["onset"] <= 5 + 2 for r in res.records)


def test_shared_events_warn_only_when_models_disagree(scorer):
    import warnings

    from reactmetric.core.predictive import Predictive

    class Late:  # forecasts only from frame 400, so misses the first event (frame 200)
        def predict(self, traj):
            p = baselines.ConstantVelocity().predict(traj)
            keep = p.issue_frames >= 400
            return Predictive(p.issue_frames[keep], p.means[keep], p.covs[keep])

    test = _tracks(3, seed=14)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scorer.evaluate({"CV": baselines.ConstantVelocity(), "CTRV": baselines.CTRV()}, test)
    with pytest.warns(RuntimeWarning, match="lack a forecast window from every model"):
        res = scorer.evaluate({"CV": baselines.ConstantVelocity(), "Late": Late()}, test)
    assert res.coverage[5]["Late"] < res.coverage[5]["CV"]


def test_event_selection(scorer, result):
    import reactmetric as rm

    stops = result.subset(event_type="stop")
    assert {r["event_type"] for r in stops.records} == {"stop"}
    assert stops.table(lead=5)["events"].iloc[0] <= result.table(lead=5)["events"].iloc[0]
    with pytest.raises(ValueError, match="no records"):
        result.subset(event_type="nothing")
    only = rm.events.select(scorer.events, types="stop")
    test = _tracks(2, seed=15)
    assert all(e.event_type == "stop" for t in test.ids for e in only.detect(test[t]))
