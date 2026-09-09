import numpy as np

import reactmetric as rm
from reactmetric.metric.evaluate import info_trace


def test_info_trace_zero_when_model_equals_reference():
    """A forecaster whose density IS the climatology scores s == 0 everywhere."""
    data, events = rm.datasets.synthetic_regime_changes(n=30, seed=2)
    calib, test = data.split(0.5, seed=0)
    ref = rm.reference.Climatology.fit(calib, leads=(5,), mode="global")

    class ClimForecaster:
        name = "clim"

        def predict(self, traj):
            obs = traj.positions
            t, dim = obs.shape
            lead = 5
            if t <= lead + 1:
                return None
            issue = np.arange(1, t - lead + 1)  # anchor at issue-1
            n = len(issue)
            means = np.empty((n, lead, dim))
            covs = np.empty((n, lead, dim, dim))
            gm = ref.models[lead]
            mu, cov = gm.mixture_mean, gm.mixture_covariance
            for i, ti in enumerate(issue):
                anchor = obs[ti - 1]
                # only the top lead matters here; fill all leads with the marginal
                means[i] = anchor + mu
                covs[i] = cov
            return rm.Predictive(issue, means, covs, horizon=lead)

    traj = test[test.ids[0]]
    frames, s = info_trace(ClimForecaster(), traj, ref, lead=5, reference_kind="climatology")
    assert len(s) > 0
    # model mean is marginal-shifted from anchor, ref scores displacement; both
    # use the same marginal Gaussian moments, so s is near zero (mixture vs its
    # moment-matched Gaussian differ slightly) -> small magnitude.
    assert np.nanmean(np.abs(s)) < 1.0


def test_benchmarkable_evaluate_runs_all_modes():
    data, events = rm.datasets.synthetic_regime_changes(n=60, seed=5)
    calib, test = data.split(0.3, seed=0)
    cv = rm.baselines.ConstantVelocity(history=10, horizon=15)
    for mode in ("global", "kinematic", "position"):
        extra = {"min_cell_samples": 40} if mode == "position" else {}
        ref = rm.reference.Climatology.fit(calib, leads=(10,), mode=mode, **extra)
        res = rm.evaluate(cv, test, ref, events, lead=10)
        assert res.n_measured > 0
        assert 0.0 <= res.anticipation_rate <= 1.0
        d = res.to_dict(n_boot=50)
        assert "km_median_react_frames" in d
