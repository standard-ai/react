import warnings

import numpy as np
import pytest

import reactmetric as rm

pd = pytest.importorskip("pandas")


def test_trajectoryset_from_dataframe_roundtrip():
    rows = []
    for tid in range(3):
        for t in range(20):
            rows.append({"track_id": tid, "timestep": t,
                         "position_x": 0.1 * t, "position_y": -0.05 * t})
    df = pd.DataFrame(rows)
    ts = rm.TrajectorySet.from_dataframe(df, id="track_id", coords=["position_x", "position_y"],
                                         time="timestep", dt=0.1)
    assert len(ts) == 3
    assert ts[0].T == 20 and ts[0].D == 2


def test_forecasts_from_dataframe_sigma_rho():
    rows = []
    for tid in range(2):
        for issue in range(5):
            for lead in range(1, 4):
                rows.append({"track_id": tid, "issue_frame": issue, "lead": lead,
                             "mu_x": float(issue + lead), "mu_y": 0.0,
                             "sigma_x": 0.5, "sigma_y": 0.3, "rho": 0.0})
    df = pd.DataFrame(rows)
    fc = rm.Forecasts.from_dataframe(df)
    assert set(fc.keys()) == {0, 1}
    pred = fc[0]
    assert pred.horizon == 3
    np.testing.assert_allclose(pred.covs[0, 0], np.diag([0.25, 0.09]), atol=1e-9)


def test_forecasts_npz_roundtrip(tmp_path):
    issue = np.arange(4)
    means = np.zeros((4, 3, 2))
    covs = np.tile(np.eye(2), (4, 3, 1, 1))
    fc = {7: rm.Predictive(issue, means, covs)}
    path = str(tmp_path / "f.npz")
    rm.Forecasts.to_npz(fc, path)
    loaded = rm.Forecasts.from_npz(path)
    assert "7" in loaded
    np.testing.assert_array_equal(loaded["7"].means, means)


def test_calibration_report_flags_overconfidence(synthetic):
    bad = rm.baselines.ConstantVelocity(measurement_noise=0.002, name="Bad")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = rm.calibration.report(bad, synthetic["data"], synthetic["ref"],
                                    synthetic["events"], lead=10, emit_warning=False)
    assert rep.coverage_chi2 > 2  # overconfident in 2D
    assert not rep.is_calibrated
    assert any("OVERCONFIDENT" in w for w in rep.warnings)


def test_counterfactual_is_labeled_hypothetical(synthetic):
    bad = rm.baselines.ConstantVelocity(measurement_noise=0.002, name="Bad")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = rm.calibration.counterfactual_react(bad, synthetic["test"], synthetic["ref"],
                                                  synthetic["events"], lead=10)
    assert cf.temperature > 1.0
    assert "HYPOTHETICAL" in cf.summary()
