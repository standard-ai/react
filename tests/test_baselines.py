import numpy as np
import pytest

import reactmetric as rm
from reactmetric.baselines import CTRV, IMM, ConstantVelocity


def test_baselines_available():
    avail = rm.baselines.available()
    assert "ConstantVelocity" in avail and "CTRV" in avail and "IMM" in avail


def test_baselines_produce_valid_predictives(synthetic):
    traj = synthetic["test"][synthetic["test"].ids[0]]
    for f in (ConstantVelocity(), CTRV(), IMM()):
        pred = f.predict(traj)
        assert pred is not None
        assert pred.means.shape[0] == len(pred.issue_frames)
        assert pred.covs.shape[1] == pred.horizon
        nll = pred.nll_matrix(traj.positions)
        assert np.isfinite(nll[~np.isnan(nll)]).all()


def test_cv_covariance_grows_with_lead(synthetic):
    traj = synthetic["test"][synthetic["test"].ids[0]]
    pred = ConstantVelocity().predict(traj)
    var = np.einsum("nhii->nh", pred.covs)  # trace per (issue, lead)
    assert np.all(np.diff(var.mean(axis=0)) >= -1e-9)


def test_constant_velocity_is_accurate_on_straight_line():
    pos = np.cumsum(np.tile([0.1, 0.05], (60, 1)), axis=0)
    traj = rm.Trajectory(pos, dt=0.1, id=0)
    pred = ConstantVelocity(history=10, horizon=15, measurement_noise=1e-4).predict(traj)
    target, mean, _ = pred.at_lead(10)
    valid = target < traj.T
    err = np.linalg.norm(traj.positions[target[valid]] - mean[valid], axis=1)
    assert err.mean() < 1e-2


def test_pretrained_grus_load_and_predict():
    pytest.importorskip("torch")
    pos = np.cumsum(np.tile([0.08, 0.02], (120, 1)), axis=0)
    traj = rm.Trajectory(pos, dt=0.1, id=0)
    for cls_name in ("GRUPosition", "GRUOrientation"):
        assert cls_name in rm.baselines.available()
        model = rm.baselines.build(cls_name)
        pred = model.predict(traj)
        assert pred is not None
        assert pred.horizon == 15
        assert pred.means.shape[1] == 15
        assert np.isfinite(pred.means).all()
        assert np.isfinite(pred.covs).all()
    from reactmetric.baselines.gru import default_checkpoint_path

    assert default_checkpoint_path("position").endswith("gru_position_best.pt")
    assert default_checkpoint_path("orientation").endswith("gru_orientation_best.pt")
