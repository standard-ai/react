"""The latent truth: correlation structure and the posterior of where the person was headed.

Paper section 3.2. Every quantity here comes from the
observed track and the climatology alone, so the posterior is shared by every model evaluated.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def info_to_c(info: float, dim: int = 2) -> float:
    """Truth spread as a fraction of climatology's: Omega = c Sigma_ref, c = exp(-2 I / d)."""
    return float(np.exp(-2.0 * info / dim))


def overlap_corr(lead: int, n_lags: int) -> np.ndarray:
    """rho_b: correlation of the unforeseeable part between targets `lag` frames apart.

    Targets lag frames apart share lead - lag frames of motion the earlier forecast could not
    see; a velocity innovation i frames before a target moves it by i steps.
    """
    rho = np.zeros(n_lags + 1)
    rho[0] = 1.0
    den = sum(i * i for i in range(1, lead + 1))
    for lag in range(1, min(lead, n_lags + 1)):
        rho[lag] = sum(i * (i + lag) for i in range(1, lead - lag + 1)) / den
    return rho


def knowable_corr(n_lags: int, scale: float, tau: float, omega: float) -> np.ndarray:
    """rho_a: nugget plus damped cosine, rho(l) = scale * exp(-l/tau) cos(omega l) for l >= 1."""
    lags = np.arange(n_lags + 1, dtype=float)
    rho = scale * np.exp(-lags / tau) * np.cos(omega * lags)
    rho[0] = 1.0
    return rho


def fit_knowable_corr(rho_w, rho_b, c, max_lag=20, noise=0.0):
    """Fit knowable_corr to (rho_w - c rho_b) / (1 - c - noise) at lags 1..max_lag by least squares.

    noise: the share of the whitened variance that is white measurement noise (truth-spread
    floor), which has no correlation at lags >= 1 and is not knowable.

    Returns (scale, tau, omega). The scale for each (tau, omega) is solved in closed form and
    clipped to [0, 1], which keeps the correlation positive definite.
    """
    lags = np.arange(1, max_lag + 1, dtype=float)
    rho_w, rho_b = np.asarray(rho_w), np.asarray(rho_b)
    target = (rho_w[1 : max_lag + 1] - c * rho_b[1 : max_lag + 1]) / max(1.0 - c - noise, 1e-6)
    ok = np.isfinite(target)
    lags, target = lags[ok], target[ok]
    taus = np.geomspace(0.3, 300.0, 120)
    omegas = np.linspace(0.0, np.pi / 2, 91)
    basis = np.exp(-lags[None, None, :] / taus[:, None, None]) * np.cos(
        omegas[None, :, None] * lags[None, None, :]
    )
    scale = np.clip((basis * target).sum(-1) / np.maximum((basis * basis).sum(-1), 1e-12), 0.0, 1.0)
    sse = ((scale[..., None] * basis - target) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    return float(scale[i, j]), float(taus[i]), float(omegas[j])


def pooled_autocorr(runs, n_lags):
    """Autocorrelation of whitened residual runs, pooled over runs and axes, without demeaning.

    Returns (rho[0..n_lags], mean square per coordinate). The mean square is 1 when the
    reference is calibrated; the residuals are not demeaned because under the reference they
    have mean zero, and demeaning short runs biases every lag downward.
    """
    tot = sum(float((r * r).sum()) for r in runs)
    cnt = sum(r.size for r in runs)
    ms = tot / max(cnt, 1)
    rho = np.full(n_lags + 1, np.nan)
    rho[0] = 1.0
    for lag in range(1, n_lags + 1):
        num, n = 0.0, 0
        for r in runs:
            if len(r) > lag:
                num += float((r[:-lag] * r[lag:]).sum())
                n += r[:-lag].size
        if n:
            rho[lag] = num / n / ms
    return rho, ms


def knowable_share(c, noise=0.0):
    """Share of the whitened variance that is knowable: 1 - c_t - noise_t (floored at 0.03 when
    there is noise). c may be per frame; noise is the whitened white-noise variance."""
    c = np.asarray(c, float)
    noise = np.broadcast_to(np.asarray(noise, float), c.shape)
    if noise.any():
        return np.clip(1.0 - c - noise, 0.03, None)
    return 1.0 - c


def truth_posterior(frames, w, rho_a, rho_b, c, noise=0.0):
    """Posterior of the knowable part a_t, filtering (w up to t) and smoothing (whole window).

    Returns mean_f (n, 2), var_f (n,), mean_s (n, 2), var_s (n,); variances are per axis and
    equal across axes. One Cholesky factor serves every t: the leading t x t block of chol(K)
    is chol of the leading block, so (L^-1 x)[:t] only depends on x[:t].

    c: per-frame truth fraction c_t (entropic information level).

    noise: per-frame variance (whitened units) of white measurement noise on the observation,
    the truth-spread floor sigma^2 I whitened; the knowable share is then 1 - c_t - noise_t
    (knowable_share). noise = 0: no floor."""
    c = np.broadcast_to(np.asarray(c, float), (len(frames),))
    noise = np.broadcast_to(np.asarray(noise, float), (len(frames),))
    va = knowable_share(c, noise)
    frames = np.asarray(frames)
    n = len(frames)
    lag = np.abs(frames[:, None] - frames[None, :])
    n_l = len(rho_a) - 1
    idx = np.minimum(lag, n_l)
    sa, sb = np.sqrt(va), np.sqrt(c)
    Ka = np.outer(sa, sa) * np.where(lag <= n_l, rho_a[idx], 0.0)
    Kw = (Ka + np.outer(sb, sb) * np.where(lag <= n_l, rho_b[idx], 0.0) + np.diag(noise)
          + 1e-9 * np.eye(n))
    L = cholesky(Kw, lower=True)
    V = solve_triangular(L, w, lower=True)
    U = solve_triangular(L, Ka, lower=True)
    Ut = np.triu(U)
    mean_f = Ut.T @ V
    var_f = np.maximum(va - (Ut * Ut).sum(0), 1e-12)
    mean_s = U.T @ V
    var_s = np.maximum(va - (U * U).sum(0), 1e-12)
    return mean_f, var_f, mean_s, var_s


def to_world(mean_w, var_w, mu_r, S_r):
    """Whitened posterior of a_t -> posterior of m_t in world coordinates: (mean, cov)."""
    Lr = np.linalg.cholesky(S_r)
    return mu_r + np.einsum("nij,nj->ni", Lr, mean_w), var_w[:, None, None] * S_r
