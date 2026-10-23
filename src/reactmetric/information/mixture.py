"""Gaussian-mixture forecasts (e.g. Trajectron++'s 25 modes), scored natively.

Paper section 3.2: the log density is evaluated exactly and the expectation over the truth uses a
sparse Gauss-Hermite residual on the moment-matched Gaussian. With one component this reduces
exactly to the Gaussian tests.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .densities import _EYE2, gh_nodes_2d, mixture_logpdf, mixture_moments, mixture_prep
from .regime_tests import RegimeTests, _polar_tails, advantage_quadratic

LOG2PI = np.log(2.0 * np.pi)


def interp_polar(g, sa, ir, A, radii, kind="cubic"):
    """Interpolate a polar-grid field known on angle indices 0, sa, 2 sa, ... and radius indices ir
    to the full (A, len(radii)) grid. g: (n, na, len(ir)). kind "linear": linear in radius,
    periodic linear in angle;
    "cubic": cubic spline in radius, trigonometric (FFT) interpolation in angle."""
    n, na, nr = g.shape
    R = len(radii)
    if kind == "cubic" and nr >= 4:
        g = CubicSpline(radii[ir], g, axis=2)(radii)
    elif nr < R:
        g = np.stack([np.interp(radii, radii[ir], row) for row in g.reshape(-1, nr)]).reshape(
            n, na, R
        )
    if na < A:
        if kind == "cubic" and na >= 4:
            F = np.fft.rfft(g, axis=1)
            Fp = np.zeros((n, A // 2 + 1, g.shape[2]), complex)
            k = F.shape[1]
            Fp[:, :k] = F
            if na % 2 == 0:
                Fp[:, na // 2] *= 0.5  # split the Nyquist term
            g = np.fft.irfft(Fp, n=A, axis=1) * (A / na)
        else:
            frac = (np.arange(A) % sa) / sa
            lo = np.arange(A) // sa
            hi = (lo + 1) % na
            g = (1 - frac)[None, :, None] * g[:, lo] + frac[None, :, None] * g[:, hi]
    return g


class MixtureForecast:
    """Per-frame Gaussian-mixture forecast for the frames of a track at one lead.

    weights (n, J), means (n, J, 2), covs (n, J, 2, 2); every component is kept (no pruning or
    merging). Also holds the exact moment-matched Gaussian. logpdf() is the one realized-score
    path (mixture_logpdf) shared with the climatology."""

    def __init__(self, weights, means, covs):
        w = np.asarray(weights, float)
        w = w / w.sum(1, keepdims=True)
        self.w, self.mu = w, np.asarray(means, float)
        self.S = np.asarray(covs, float) + 1e-12 * _EYE2
        self.raw = (w, self.mu, self.S)
        self.mm_mu, self.mm_S = mixture_moments(w, self.mu, self.S)
        self.mm_S = self.mm_S + 1e-12 * _EYE2
        self.P = np.linalg.inv(self.S)
        self.ld = np.linalg.slogdet(self.S)[1]
        self.logw = np.log(np.clip(w, 1e-300, None))
        self.M = w.shape[1]
        self._prep = None

    def take(self, idx):
        out = object.__new__(MixtureForecast)
        for name in ("w", "mu", "S", "mm_mu", "mm_S", "P", "ld", "logw"):
            setattr(out, name, getattr(self, name)[idx])
        out.raw = (out.w, out.mu, out.S)
        out.M, out._prep = self.M, None
        return out

    def logpdf(self, y):
        """Log density at one point per frame, y (n, 2)."""
        if self._prep is None:
            self._prep = mixture_prep(*self.raw)
        return mixture_logpdf(y[:, None], self._prep)[:, 0]


def _features(z):
    return np.stack(
        [z[:, 0] ** 2, 2.0 * z[:, 0] * z[:, 1], z[:, 1] ** 2, z[:, 0], z[:, 1], np.ones(len(z))], 1
    )


def expected_logn_on_grid(loc, Lc, Om, mu, P, ld, feats):
    """E_{y ~ N(m, Om)} log N_j(y) at m = loc + Lc z for the grid z (Pz): loc (n,2), Lc (n,2,2),
    Om (n,2,2),
    component mu (n,J,2), P (n,J,2,2), ld (n,J). Returns (n, J, Pz)."""
    d0 = loc[:, None] - mu  # (n,J,2)
    PL = np.einsum("njik,nkl->njil", P, Lc)
    Mq = np.einsum("nki,njkl->njil", Lc, PL)  # Lc' P Lc
    lin = 2.0 * np.einsum("nki,njk->nji", Lc, np.einsum("njkl,njl->njk", P, d0))
    const = np.einsum("nji,njik,njk->nj", d0, P, d0)
    tr = np.einsum("njik,nki->nj", P, Om)
    coef = np.empty(d0.shape[:2] + (6,))
    coef[..., 0] = Mq[..., 0, 0]
    coef[..., 1] = Mq[..., 0, 1]
    coef[..., 2] = Mq[..., 1, 1]
    coef[..., 3] = lin[..., 0]
    coef[..., 4] = lin[..., 1]
    coef[..., 5] = const + tr + ld
    coef = -0.5 * coef
    coef[..., 5] -= LOG2PI
    return np.einsum("njf,pf->njp", coef, feats)


class RegimeTestsMixture(RegimeTests):
    """RegimeTests for a Gaussian-mixture model forecast.

    The advantage of regime k at truth location m, with q = N(m, c S_k), is
    I_k(m) = E_q[log p_model] - E_q[log w_k N_k] - G_k(m). Write E_q[log p_model] = F_g(m) + Res(m),
    where F_g is the exact quadratic of the moment-matched Gaussian g (what advantage_quadratic
    returns) and Res the non-Gaussian remainder. Res is Gauss-Hermite (order gh) on q of log p at a
    coarse polar subset (stride = angle, radius steps) and interpolated; regimes carrying less than
    heavy_weight of the mass use the closed-form variational bound instead. With one component
    Res = 0 and this reduces to RegimeTests. Use as

        X = RegimeTestsMixture.from_exact(exact, gh=5, stride=(8, 2))
        v = X.test_full_mixture(idx, fc.take(rows), eps, margins=margins)   # same dict as test_full
    """

    def __init__(self, *args, gh=5, stride=(8, 2), interp="cubic", heavy_weight=1e-3, **kw):
        super().__init__(*args, **kw)
        self._setup(gh, stride, interp, heavy_weight)

    def _setup(self, gh, stride, interp, heavy_weight):
        self.gh, self.stride, self.interp, self.heavy_weight = gh, stride, interp, heavy_weight
        self._terms = None
        self._feats = _features((self.dirs[:, None, :] * self.radii[None, :, None]).reshape(-1, 2))
        tw = np.gradient(self.radii) * self.radii * np.exp(-0.5 * self.radii**2)
        self._tw = tw / tw.sum()

    @classmethod
    def from_exact(cls, X, gh=5, stride=(8, 2), interp="cubic", heavy_weight=1e-3):
        """Wrap an existing RegimeTests (shares its regime posteriors and overlap terms)."""
        out = object.__new__(cls)
        out.__dict__.update(X.__dict__)
        out._setup(gh, stride, interp, heavy_weight)
        return out

    def _set(self, k, which, idx):
        if which == "post":
            return self.post_mean[idx, k], self.post_cov[idx, k], self.post_w[idx, k]
        return self.means[idx, k], self.V[k][idx], self.w[idx, k]

    def model_terms(self, idx, fc, min_weight=1e-9):
        """Per (regime, posterior/prior): the Gaussian quadratic (M, g, h) of the moment-matched
        Gaussian and the mixture residual Res on the polar grid (n, A, R)."""
        n, A, R = len(idx), len(self.dirs), len(self.radii)
        terms = {}
        for k in range(self.K):
            mu_k, S_k = self.means[idx, k], self.covs[idx, k]
            Om = self.Om[k][idx]
            for which in ("post", "prior"):
                loc, C, wk = self._set(k, which, idx)
                Mq, gq, hq = advantage_quadratic(
                    loc, C, fc.mm_mu, fc.mm_S, mu_k, S_k, self.c, self.sig2
                )
                Lc = np.linalg.cholesky(C + 1e-15 * _EYE2)
                res = np.zeros((n, A * R))
                live = np.flatnonzero(wk > min_weight)
                heavy = live[wk[live] > self.heavy_weight]
                light = np.setdiff1d(live, heavy)
                if len(light) and fc.M > 1:
                    res[light] = self._lse_residual(loc, Lc, Om, fc, light)
                if len(heavy):
                    Pm = np.linalg.inv(fc.mm_S[heavy])
                    ldm = np.linalg.slogdet(fc.mm_S[heavy])[1][:, None]
                    res[heavy] = self._gh_residual(
                        loc[heavy], Lc[heavy], Om[heavy], fc.take(heavy), Pm, ldm
                    )
                res = res.reshape(n, A, R)
                terms[(k, which)] = (Mq, gq, hq, res, res.mean(1) @ self._tw)
        return terms

    def _lse_residual(self, loc, Lc, Om, fc, rows):
        """Variational lower bound minus the Gaussian's value on the grid (len(rows), A*R)."""
        r_ = rows
        Pm = np.linalg.inv(fc.mm_S[r_])
        ldm = np.linalg.slogdet(fc.mm_S[r_])[1][:, None]
        Fm = expected_logn_on_grid(
            loc[r_], Lc[r_], Om[r_], fc.mu[r_], fc.P[r_], fc.ld[r_], self._feats
        )
        lse_ = Fm + fc.logw[r_][..., None]
        mx = lse_.max(1, keepdims=True)
        Fa = (mx + np.log(np.exp(lse_ - mx).sum(1, keepdims=True)))[:, 0]
        Fg = expected_logn_on_grid(
            loc[r_], Lc[r_], Om[r_], fc.mm_mu[r_][:, None], Pm[:, None], ldm, self._feats
        )[:, 0]
        return Fa - Fg

    def _gh_residual(self, loc, Lc, Om, fc, Pm, ldm):
        """Res(m) = E_q log p_M - E_q log g at a coarse (angle, radius) subset of the polar points
        (Gauss-Hermite on q of log p_M), then interpolated. E_q log g is the exact quadratic,
        so Res is the smooth non-Gaussian remainder."""
        n, A, R = len(loc), len(self.dirs), len(self.radii)
        sa, sr = self.stride
        ia = np.arange(0, A, sa)
        ir = np.unique(np.append(np.arange(0, R, sr), R - 1))
        z = (self.dirs[ia][:, None, :] * self.radii[ir][None, :, None]).reshape(-1, 2)
        feats = _features(z)
        Fg = expected_logn_on_grid(loc, Lc, Om, fc.mm_mu[:, None], Pm[:, None], ldm, feats)[
            :, 0
        ]  # (n, Ps)
        nodes, gw = gh_nodes_2d(self.gh)
        LO = np.linalg.cholesky(Om + 1e-15 * _EYE2)
        Ys = np.einsum("nij,gj->ngi", LO, nodes)
        Ps = len(z)
        gh = np.empty((n, Ps))
        f3 = np.stack([z[:, 0] ** 2, 2.0 * z[:, 0] * z[:, 1], z[:, 1] ** 2], 1)  # (Ps, 3)
        step = max(1, int(3e6 // (Ps * len(nodes) * fc.M)))
        for s0 in range(0, n, step):
            sl = slice(s0, s0 + step)
            P_, mu_, Lc_, loc_ = fc.P[sl], fc.mu[sl], Lc[sl], loc[sl]
            # (y - mu_j)' P_j (y - mu_j), y = loc + Lc z + Ys_g, is quadratic in z
            A_ = np.einsum("nki,njkl,nlm->njim", Lc_, P_, Lc_)  # Lc' P Lc
            quad = np.einsum(
                "njf,pf->njp", np.stack([A_[..., 0, 0], A_[..., 0, 1], A_[..., 1, 1]], -1), f3
            )
            d0 = loc_[:, None, None] + Ys[sl][:, :, None] - mu_[:, None]  # (b, G, M, 2)
            Pd0 = np.einsum("njkl,ngjl->ngjk", P_, d0)
            lin = 2.0 * np.einsum("nki,ngjk->ngji", Lc_, Pd0)  # (b, G, M, 2)
            const = np.einsum("ngji,ngji->ngj", d0, Pd0)
            qq = (
                quad[:, None] + np.einsum("ngjf,pf->ngjp", lin, z) + const[..., None]
            )  # (b, G, M, Ps)
            lc = (
                fc.logw[sl][:, None, :, None]
                - 0.5 * qq
                - 0.5 * fc.ld[sl][:, None, :, None]
                - LOG2PI
            )
            mx = lc.max(2, keepdims=True)
            lp = (mx + np.log(np.exp(lc - mx).sum(2, keepdims=True)))[:, :, 0]  # (b, G, Ps)
            gh[sl] = np.einsum("ngp,g->np", lp, gw)
        g = (gh - Fg).reshape(n, len(ia), len(ir))
        g = interp_polar(g, sa, ir, A, self.radii, self.interp)
        return g.reshape(n, A * R)

    def _tails(self, idx, mu_m, S_m, eps, which, need_lt=True):
        if self._terms is None:
            return super()._tails(idx, mu_m, S_m, eps, which, need_lt)
        p_lt = np.zeros(len(idx))
        p_gt = np.zeros(len(idx))
        mean_I = np.zeros(len(idx))
        r = self.radii
        eps = np.broadcast_to(np.asarray(eps, float), (len(idx),))
        thr = eps[:, None, None]
        for k in range(self.K):
            _, _, wk = self._set(k, which, idx)
            M, g, h, res, mres = self._terms[(k, which)]
            h = h - np.log(np.clip(self.w[idx, k], 1e-300, None))
            A = np.einsum("ai,nij,aj->na", self.dirs, M, self.dirs)
            B = np.einsum("ai,ni->na", self.dirs, g)
            Q = A[..., None] * (r * r) + B[..., None] * r + h[:, None, None]
            G = self.G[(k, which)][idx]
            I = Q + res - G
            gt, lt = _polar_tails(I, thr, r, A, B)
            if need_lt and eps.any():
                _, lt = _polar_tails(I, 0.0, r, A, B)
            p_lt += wk * lt
            p_gt += wk * gt
            mean_I += wk * (np.einsum("nii->n", M) + h - self.mean_G[(k, which)][idx] + mres)
        return p_lt, p_gt, mean_I

    def test_full_mixture(self, idx, fc, eps=0.0, margins=(), base=True):
        """test_full for a mixture forecast fc (MixtureForecast aligned to rows idx)."""
        self._terms = self.model_terms(idx, fc)
        try:
            return self.test_full(idx, None, None, eps, margins, base)
        finally:
            self._terms = None
