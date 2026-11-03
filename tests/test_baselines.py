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


@pytest.mark.parametrize("cls", [CTRV, IMM])
def test_ctrv_imm_need_history_of_four(cls):
    # both initialise from observation 3, which a forecast issued before frame 3 must not see
    with pytest.raises(ValueError, match="history >= 4"):
        cls(history=3)
    pos = np.cumsum(np.tile([0.1, 0.05], (40, 1)), axis=0)
    assert cls(history=4).predict(rm.Trajectory(pos, dt=0.1, id=0)).issue_frames[0] == 3


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


@pytest.mark.parametrize("variant", ["position", "orientation"])
@pytest.mark.parametrize("heading_deg", [0.0, 60.0, -135.0])
def test_gru_covariance_is_rotated_to_world_frame(variant, heading_deg):
    """Network variances are along-/cross-track; the world covariance must follow the heading."""
    torch = pytest.importorskip("torch")
    from reactmetric.baselines import GRUOrientation, GRUPosition

    class _Stub:
        """Fixed heading-frame output: mean (1, 0) along-track, var (4, 0.25) along/cross."""

        def __call__(self, x):
            n, h = x.shape[0], 15
            mean = torch.zeros(n, h, 2)
            mean[:, :, 0] = 1.0
            log_var = torch.log(torch.tensor([4.0, 0.25])).expand(n, h, 2).contiguous()
            return mean, log_var

    theta = np.deg2rad(heading_deg)
    along = np.array([np.cos(theta), np.sin(theta)])
    cross = np.array([-np.sin(theta), np.cos(theta)])
    pos = np.array([3.0, -2.0]) + 0.1 * np.arange(60)[:, None] * along
    traj = rm.Trajectory(pos, dt=0.1, id=0)

    model = (GRUPosition if variant == "position" else GRUOrientation)()
    model.model = _Stub()
    pred = model.predict(traj)
    assert pred is not None

    cov = pred.covs[0, 0]
    np.testing.assert_allclose(cov @ along, 4.0 * along, atol=1e-6)
    np.testing.assert_allclose(cov @ cross, 0.25 * cross, atol=1e-6)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    # Means use the same map: 1 m ahead of the history window's first frame, along the heading.
    np.testing.assert_allclose(pred.means[0, 0], pos[0] + along, atol=1e-6)
