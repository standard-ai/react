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


def test_forecasts_from_dataframe_reads_leads_by_value():
    rows = [{"track_id": 0, "issue_frame": issue, "lead": lead,
             "mu_x": float(100 * issue + lead), "mu_y": 0.0,
             "sigma_x": 0.5, "sigma_y": 0.3, "rho": 0.0}
            for issue in range(3) for lead in (1, 5, 10, 15)]
    pred = rm.Forecasts.from_dataframe(pd.DataFrame(rows))[0]
    assert pred.horizon == 15
    target, mu, _ = pred.at_lead(5)
    np.testing.assert_array_equal(target, np.arange(3) + 5)
    np.testing.assert_array_equal(mu[:, 0], 100 * np.arange(3) + 5)
    # a lead absent from the file is all-NaN, not the next lead's forecast
    assert np.isnan(pred.at_lead(2)[1]).all()


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

