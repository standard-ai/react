import numpy as np

from reactmetric.baselines import IMM, ConstantVelocity
from reactmetric.baselines._kinematics import stack_tracks
from reactmetric.constants import COV_REGULARIZATION
from reactmetric.scoring.nll import _LOG_2PI


def _cv_tracks(n_tracks, T, accel, noise, dt=0.1, seed=0):
    """Tracks drawn from the Kalman filter's own model: white-noise acceleration."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_tracks):
        x = np.zeros((T, 2))
        v = rng.normal(0.0, 1.0, 2)
        for t in range(1, T):
            a = rng.normal(0.0, accel, 2)
            x[t] = x[t - 1] + v * dt + a * dt**2 / 2
            v = v + a * dt
        out[i] = x + rng.normal(0.0, noise, (T, 2))
    return out


def test_cv_fit_recovers_noise_levels():
    tracks = _cv_tracks(60, 200, accel=0.8, noise=0.03)
    cv = ConstantVelocity.fit(tracks)
    assert abs(cv.accel_noise / 0.8 - 1) < 0.15
    assert abs(cv.measurement_noise / 0.03 - 1) < 0.1
    assert cv.fit_result_["converged"]


def test_cv_fitted_forecast_is_calibrated_on_its_own_model():
    tracks = _cv_tracks(40, 200, accel=0.8, noise=0.03, seed=1)
    cv = ConstantVelocity(measurement_noise=0.03, accel_noise=0.8)
    import reactmetric as rm

    z2 = []
    for i, x in tracks.items():
        pred = cv.predict(rm.Trajectory(x, dt=0.1, id=i))
        target, mean, cov = pred.at_lead(10)
        ok = target < len(x)
        d = x[target[ok]] - mean[ok]
        z2.append(np.einsum("ni,nij,nj->n", d, np.linalg.inv(cov[ok]), d) / 2)
    assert abs(np.concatenate(z2).mean() - 1) < 0.1


def _imm_scalar_loglik(imm, o, dt):
    """Step-by-step IMM innovation likelihood, written like IMM.predict's filter."""
    M = imm._M
    Fs, Qs = imm._mode_matrices(dt)
    m = len(Fs)
    sr = imm.measurement_noise
    Rs = [imm._R[j] + np.eye(2) * COV_REGULARIZATION for j in range(m)]
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    v0 = (o[3] - o[0]) / (3 * dt)
    x0 = np.array([*o[0], *v0])
    P0 = np.diag([sr**2, sr**2, 0.25, 0.25])
    xs = [x0.copy() for _ in range(m)]
    Ps = [P0.copy() for _ in range(m)]
    mu = np.full(m, 1.0 / m)
    total = 0.0
    for t in range(1, len(o)):
        cbar = np.maximum(M.T @ mu, 1e-12)
        w = (M * mu[:, None]) / cbar[None, :]
        xm = [sum(w[i, j] * xs[i] for i in range(m)) for j in range(m)]
        Pm = [
            sum(w[i, j] * (Ps[i] + np.outer(xs[i] - xm[j], xs[i] - xm[j])) for i in range(m))
            for j in range(m)
        ]
        xs = [Fs[j] @ xm[j] for j in range(m)]
        Ps = [Fs[j] @ Pm[j] @ Fs[j].T + Qs[j] for j in range(m)]
        lik = np.empty(m)
        for j in range(m):
            S = H @ Ps[j] @ H.T + Rs[j]
            Si = np.linalg.inv(S)
            r = o[t] - H @ xs[j]
            K = Ps[j] @ H.T @ Si
            xs[j] = xs[j] + K @ r
            Ps[j] = (np.eye(4) - K @ H) @ Ps[j]
            lik[j] = -0.5 * r @ Si @ r - 0.5 * np.linalg.slogdet(S)[1] - _LOG_2PI
        lm = np.log(cbar) + lik
        pk = lm.max()
        total += pk + np.log(np.exp(lm - pk).sum())
        mu = np.exp(lm - pk)
        mu /= mu.sum()
    return total


def test_imm_batched_likelihood_matches_scalar_filter():
    t = np.arange(300) * 0.1
    heading = np.cumsum(0.3 * np.sin(t / 3)) * 0.1
    v = 1.2 * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    walk = np.cumsum(v * 0.1, axis=0) + np.random.default_rng(0).normal(0, 0.02, (300, 2))
    tracks = {0: walk + [30.0, -12.0], 1: walk[:120], 2: walk[:40] * 0.1}
    imm = IMM(measurement_noise=0.03, measurement_noise_stop=0.012, p_stay_stop=0.995)
    obs, lengths, dt = stack_tracks(tracks)
    batch = imm.innovation_loglik(obs, lengths, dt)
    scalar = [_imm_scalar_loglik(imm, x, dt) for x in tracks.values()]
    np.testing.assert_allclose(batch, scalar, rtol=1e-8)


def test_imm_fit_improves_likelihood():
    tracks = _cv_tracks(20, 120, accel=0.8, noise=0.02, seed=2)
    obs, lengths, dt = stack_tracks(tracks)
    before = IMM().innovation_loglik(obs, lengths, dt).sum()
    fitted = IMM.fit(tracks, maxiter=2)
    after = fitted.innovation_loglik(obs, lengths, dt).sum()
    assert after > before


def test_imm_fit_respects_bounds():
    tracks = _cv_tracks(10, 80, accel=0.8, noise=0.02, seed=3)
    box = {"turn_rate": (0.4, 0.8), "sigma_a_ct": (0.5, 0.9), "p_stay": (0.97, 0.97)}
    fitted = IMM.fit(tracks, bounds=box, maxiter=1)
    assert 0.4 <= fitted.turn_rate <= 0.8 and 0.5 <= fitted.sigma_a_ct <= 0.9
    assert fitted.p_stay == 0.97


def test_imm_stop_mode_defaults_match_shared_values():
    a, b = (
        IMM(measurement_noise=0.03),
        IMM(measurement_noise=0.03, measurement_noise_stop=0.03, p_stay_stop=0.97),
    )
    np.testing.assert_array_equal(a._M, b._M)
    np.testing.assert_array_equal(a._R, b._R)
    c = IMM(p_stay=0.97, p_stay_stop=0.995)
    np.testing.assert_allclose(c._M.sum(axis=1), 1.0)
    assert c._M[0, 0] == 0.995 and c._M[1, 1] == 0.97
