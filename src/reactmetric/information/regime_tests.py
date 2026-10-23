"""Onset and recovery tests: posterior probabilities that a model is behind or ahead.

Paper sections 3.2-3.3 / docs/react-spec.md section 6. The climatology is a two-regime mixture
(moving, stopped); the regime is made explicit in the truth model, so given the regime the
advantage is a quadratic in the truth location minus the overlap term G_k, and its tail
probabilities are integrated exactly on a polar grid. There is no Monte Carlo.
"""

from __future__ import annotations

import numpy as np

from .densities import _EYE2, gh_nodes_2d
from .overlap import overlap_on_grid


def advantage_quadratic(mu, C, mu_m, S_m, mu_r, S_r, c, sig2=0.0):
    """Write I_t(m), m ~ N(mu, C), as z'Mz + g'z + h with z ~ N(0, I_2). Omega = c S_r + sig2 I.

    Coordinates are centred on the reference mean first; I is translation invariant and the
    raw quadratic terms in world coordinates would cancel catastrophically.
    """
    mu = mu - mu_r
    mu_m = mu_m - mu_r
    Ar = np.linalg.inv(S_r)
    Am = np.linalg.inv(S_m)
    D = 0.5 * (Ar - Am)
    d = np.einsum("nij,nj->ni", Am, mu_m)
    _, ldr = np.linalg.slogdet(S_r)
    _, ldm = np.linalg.slogdet(S_m)
    e = 0.5 * (
        c * np.einsum("nij,nji->n", Ar - Am, S_r)
        + sig2 * np.einsum("nii->n", Ar - Am)
        - np.einsum("ni,nij,nj->n", mu_m, Am, mu_m)
        + ldr
        - ldm
    )
    S = np.linalg.cholesky(C + 1e-15 * _EYE2)
    M = np.einsum("nji,njk,nkl->nil", S, D, S)
    g = np.einsum("nji,nj->ni", S, 2.0 * np.einsum("nij,nj->ni", D, mu) + d)
    h = np.einsum("ni,nij,nj->n", mu, D, mu) + np.einsum("ni,ni->n", d, mu) + e
    return M, g, h


def _quadratic_roots(a, b, c, tol=1e-14):
    """Real roots (r1 <= r2) of a u^2 + b u + c = 0, elementwise; NaN where absent."""
    with np.errstate(divide="ignore", invalid="ignore"):
        quad = np.abs(a) >= tol
        disc = b * b - 4.0 * a * c
        sq = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
        q1, q2 = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)
        lin = -c / b
        r1 = np.where(quad, np.minimum(q1, q2), np.where(np.abs(b) >= tol, lin, np.nan))
        r2 = np.where(quad, np.maximum(q1, q2), np.nan)
    return r1, r2


def _log_odds(p, floor=1e-12):
    p = np.clip(p, floor, 1.0 - floor)
    return np.log(p) - np.log1p(-p)


class RegimePosterior:
    """The shared posterior of the truth location split by the climatology's regime.

    post_mu, post_C: the shared posterior of m (n, 2), (n, 2, 2); mm_mu, mm_prior: the
    Gaussian prior it was computed under; weights (n, K), means (n, K, 2), covs (n, K, 2, 2):
    the climatology's regimes; c: truth fraction within a regime.
    """

    def __init__(self, post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, sig2=0.0):
        n, K = weights.shape
        self.K, self.c, self.sig2 = K, float(c), float(sig2)
        self.w, self.means, self.covs = weights, means, covs
        Ai = np.linalg.inv(post_C)
        Bi = np.linalg.inv(mm_prior)
        eta = np.einsum("nij,nj->ni", Ai, post_mu) - np.einsum("nij,nj->ni", Bi, mm_mu)
        Lam = Ai - Bi
        ev, V = np.linalg.eigh(0.5 * (Lam + np.swapaxes(Lam, 1, 2)))
        Lam = np.einsum("nij,nj,nkj->nik", V, np.clip(ev, 0.0, None), V)  # evidence precision
        self.post_mean = np.empty((n, K, 2))
        self.post_cov = np.empty((n, K, 2, 2))
        log_ev = np.empty((n, K))
        # truth spread Omega_k = c S_k + sig2 I; the regime's prior spread of m is what is left of
        # S_k: V_k = (1 - c) S_k - sig2 I, eigen-clipped to stay positive definite (sig2 = 0: as is)
        self.Om = [self.c * covs[:, k] + self.sig2 * _EYE2 for k in range(K)]
        self.V = []
        for k in range(K):
            Vk = (1.0 - self.c) * covs[:, k]
            if self.sig2 > 0.0:
                Vk = Vk - self.sig2 * _EYE2
                e_, U_ = np.linalg.eigh(Vk)
                fl = 0.02 * (1.0 - self.c) * 0.5 * np.trace(covs[:, k], axis1=1, axis2=2)
                Vk = np.einsum("nij,nj,nkj->nik", U_, np.maximum(e_, fl[:, None]), U_)
            self.V.append(Vk)
        for k in range(K):
            Vk = self.V[k]
            Qk = np.linalg.inv(Vk)
            Pk = Qk + Lam
            Pk_inv = np.linalg.inv(Pk)
            hk = np.einsum("nij,nj->ni", Qk, means[:, k]) + eta
            self.post_cov[:, k] = Pk_inv
            self.post_mean[:, k] = np.einsum("nij,nj->ni", Pk_inv, hk)
            log_ev[:, k] = (
                0.5 * np.einsum("ni,nij,nj->n", hk, Pk_inv, hk)
                - 0.5 * np.einsum("ni,nij,nj->n", means[:, k], Qk, means[:, k])
                - 0.5 * np.linalg.slogdet(Vk)[1]
                - 0.5 * np.linalg.slogdet(Pk)[1]
            )
        lp = np.log(np.clip(weights, 1e-300, None)) + log_ev
        lp -= lp.max(1, keepdims=True)
        self.post_w = np.exp(lp)
        self.post_w /= self.post_w.sum(1, keepdims=True)


def regime_predictive(post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, sig2=0.0):
    """Posterior predictive of y under the regime truth model, moment-matched to a Gaussian.

    Per regime k: m | k, data ~ N(post_mean_k, post_cov_k) (RegimePosterior's regime split of the
    shared posterior) and y | m, k ~ N(m, c S_k); the regimes are weighted by their posterior
    probabilities. With a hindsight (smoothed) posterior this is Oracle B, the most a forecast
    can extract at the assumed information level.
    """
    rt = RegimePosterior(post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, sig2=sig2)
    mean = np.einsum("nk,nki->ni", rt.post_w, rt.post_mean)
    d = rt.post_mean - mean[:, None]
    cov = np.einsum(
        "nk,nkij->nij", rt.post_w,
        rt.post_cov + np.stack(rt.Om, 1) + np.einsum("nki,nkj->nkij", d, d)
    )
    return mean, cov + 1e-12 * _EYE2


# ---------------------------------------------------------------------------
# Regime tests with the exact mixture score: overlap term per truth location
# ---------------------------------------------------------------------------
#
# Given the regime k, the advantage against the real mixture is
#
#   I_k(m) = Q_k(m) - G_k(m),   Q_k(m) = E_q[log p_model] - E_q[log w_k N_k]   (quadratic in m)
#   G_k(m) = E_{y ~ N(m, c S_k)} log(1 + sum_j w_j N_j(y) / (w_k N_k(y))) >= 0,
#
# G_k does not depend on the model, so it is computed once per frame and shared by every
# model. With y = m + chol(c S_k) x_g at the Gauss-Hermite nodes x_g, each log-ratio is a
# quadratic in m, so on a fixed set of points m = loc + chol(C) z it is a quadratic in z per node
# and G_k is evaluated exactly (to quadrature accuracy) with no table and no interpolation.
#
# Tail probabilities of I_k are integrated over m's standardised coordinates on a polar grid:
# along each direction the sign changes are solved from the local quadratic and the Gaussian
# mass on each side added in closed form (P(r > a) = exp(-a^2 / 2) for a standard 2D normal).
# Both tails are summed directly, so very small probabilities keep their precision.

POLAR_ANGLES = 64
POLAR_RADII = np.linspace(0.0, 8.0, 17)


def _polar_points(n_angles=POLAR_ANGLES, radii=POLAR_RADII):
    phi = 2.0 * np.pi * (np.arange(n_angles) + 0.5) / n_angles
    d = np.column_stack([np.cos(phi), np.sin(phi)])  # (A, 2)
    return d, radii


def _polar_tails(I, thr, radii, A, B):
    """P(I > thr) and P(I < thr) from I on a polar grid (n, A, R) over a standard 2D normal.

    I = A r^2 + B r + C(r) along each ray with C linear within each radial interval; each
    crossing is solved from that quadratic. Both tails are summed directly, interval by
    interval, so each keeps its precision.
    """
    e = np.exp(-0.5 * radii * radii)
    I0, I1 = I[..., :-1], I[..., 1:]
    p0, p1 = I0 > thr, I1 > thr
    r0, r1 = radii[:-1], radii[1:]
    a, b = A[..., None], B[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        C0 = I0 - a * r0 * r0 - b * r0
        sl = (I1 - a * r1 * r1 - b * r1 - C0) / (r1 - r0)
        q1, q2 = _quadratic_roots(np.broadcast_to(a, I0.shape), b + sl, C0 - sl * r0 - thr)
        rr = r0 + (thr - I0) / (I1 - I0) * (r1 - r0)
        tol = 1e-9
        rr = np.where((q1 >= r0 - tol) & (q1 <= r1 + tol), q1,
                      np.where((q2 >= r0 - tol) & (q2 <= r1 + tol), q2, rr))
        rr = np.clip(np.nan_to_num(rr, nan=0.5 * (r0 + r1)), r0, r1)
    er = np.exp(-0.5 * rr * rr)
    lo, hi = e[:-1] - er, er - e[1:]
    seg = e[:-1] - e[1:]
    pos = np.where(p0 & p1, seg, 0.0) + np.where(p0 & ~p1, lo, 0.0) + np.where(~p0 & p1, hi, 0.0)
    neg = np.where(~p0 & ~p1, seg, 0.0) + np.where(p0 & ~p1, hi, 0.0) + np.where(~p0 & p1, lo, 0.0)
    thr = np.asarray(thr, float)
    last = I[..., -1] > (thr[..., 0] if thr.ndim == I.ndim else thr)
    gt = pos.sum(-1) + np.where(last, e[-1], 0.0)
    lt = neg.sum(-1) + np.where(last, 0.0, e[-1])
    return gt.mean(-1), lt.mean(-1)


class RegimeTests(RegimePosterior):
    """Exact regime-conditioned onset/recovery tests for one event window at one level,
    scored against the real mixture: the overlap term per truth location.

    post_mu, post_C: the shared posterior of m (n, 2), (n, 2, 2); mm_mu, mm_prior: the
    Gaussian prior it was computed under; weights (n, K), means (n, K, 2), covs (n, K, 2, 2):
    the climatology's regimes; c: truth fraction within a regime. The overlap term G_k is
    evaluated once on each regime's polar grid (posterior and prior) and shared by every model
    tested on this window.

    With table (an OverlapTable for this lead and level) and geom = (speed, heading, anchor)
    per frame, G_k is looked up at every grid point and only points beyond the table are
    computed directly. Without a table it is computed directly on every g_stride = (angle,
    radius) grid point and interpolated between them.
    """

    def __init__(self, post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c,
                 n_angles=POLAR_ANGLES, radii=POLAR_RADII, overlap_gh=9, blob_scale=(4.0,),
                 g_stride=(1, 1), chunk=8, min_weight=1e-12, table=None, geom=None, sig2=0.0):
        super().__init__(
            post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, sig2=sig2
        )
        if table is not None and abs(getattr(table, "sig2", 0.0) - self.sig2) > 1e-15:
            raise ValueError("overlap table built for a different noise floor")
        self.dirs, self.radii = _polar_points(n_angles, radii)
        nodes, gw = gh_nodes_2d(overlap_gh)
        n = len(weights)
        self.G = {}
        self.n_off = 0
        if table is not None:
            z = (self.dirs[:, None, :] * self.radii[None, :, None]).reshape(-1, 2)
            speed, heading, anchor = geom
        else:
            sa, sr = g_stride
            ia = np.arange(0, n_angles, sa)
            ir = np.unique(np.append(np.arange(0, len(radii), sr), len(radii) - 1))
            z = (self.dirs[ia][:, None, :] * self.radii[ir][None, :, None]).reshape(-1, 2)
        for k in range(self.K):
            for which in ("post", "prior"):
                if which == "post":
                    loc, C, wk = self.post_mean[:, k], self.post_cov[:, k], self.post_w[:, k]
                else:
                    loc, C, wk = means[:, k], self.V[k], weights[:, k]
                g = np.zeros((n, len(z)))
                rows = np.flatnonzero(wk > min_weight)
                if table is not None and len(rows):
                    m = loc[rows, None] + np.einsum(
                        "nij,pj->npi", np.linalg.cholesky(C[rows] + 1e-15 * _EYE2), z
                    )
                    vals, off = table.lookup(k, speed[rows], heading[rows], anchor[rows], m)
                    if off.any():
                        ri_, pi_ = np.nonzero(off)
                        fr = rows[ri_]
                        vals[ri_, pi_] = overlap_on_grid(
                            m[ri_, pi_], np.broadcast_to(1e-12 * _EYE2, (len(fr), 2, 2)),
                            weights[fr], means[fr], covs[fr], k, self.c, np.zeros((1, 2)),
                            nodes, gw, blob_scale=blob_scale, sig2=self.sig2,
                        )[:, 0]
                        self.n_off += len(fr)
                    g[rows] = vals
                    self.G[(k, which)] = g.reshape(n, n_angles, len(radii))
                    continue
                for s0 in range(0, len(rows), chunk):
                    r = rows[s0 : s0 + chunk]
                    g[r] = overlap_on_grid(
                        loc[r], C[r], weights[r], means[r], covs[r], k, self.c, z, nodes, gw,
                        blob_scale=blob_scale, sig2=self.sig2,
                    )
                if table is not None:
                    self.G[(k, which)] = g.reshape(n, n_angles, len(radii))
                    continue
                g = g.reshape(n, len(ia), len(ir))
                if len(ir) < len(radii):  # linear in radius
                    g = np.stack(
                        [np.interp(self.radii, self.radii[ir], row)
                         for row in g.reshape(-1, len(ir))]
                    )
                    g = g.reshape(n, len(ia), len(radii))
                if sa > 1:  # periodic linear in angle
                    frac = (np.arange(n_angles) % sa) / sa
                    lo = np.arange(n_angles) // sa
                    hi = (lo + 1) % len(ia)
                    g = (1 - frac)[None, :, None] * g[:, lo] + frac[None, :, None] * g[:, hi]
                self.G[(k, which)] = g
        # E[G] over each regime's Gaussian, for the posterior mean advantage
        tw = np.gradient(self.radii) * self.radii * np.exp(-0.5 * self.radii**2)
        tw /= tw.sum()
        self.mean_G = {key: val.mean(1) @ tw for key, val in self.G.items()}

    def _tails(self, idx, mu_m, S_m, eps, which, need_lt=True):
        p_lt = np.zeros(len(idx))
        p_gt = np.zeros(len(idx))
        mean_I = np.zeros(len(idx))
        r = self.radii
        eps = np.broadcast_to(np.asarray(eps, float), (len(idx),))
        thr = eps[:, None, None]
        for k in range(self.K):
            mu_k, S_k = self.means[idx, k], self.covs[idx, k]
            if which == "post":
                m, C, wk = self.post_mean[idx, k], self.post_cov[idx, k], self.post_w[idx, k]
            else:
                m, C, wk = mu_k, self.V[k][idx], self.w[idx, k]
            corr = self.mean_G[(k, which)][idx]
            M, g, h = advantage_quadratic(m, C, mu_m, S_m, mu_k, S_k, self.c, self.sig2)
            h = h - np.log(np.clip(self.w[idx, k], 1e-300, None))
            A = np.einsum("ai,nij,aj->na", self.dirs, M, self.dirs)  # (n, A)
            B = np.einsum("ai,ni->na", self.dirs, g)
            Q = A[..., None] * (r * r) + B[..., None] * r + h[:, None, None]
            G = self.G[(k, which)][idx]
            gt, lt = _polar_tails(Q - G, thr, r, A, B)
            if need_lt and eps.any():  # onset is I < 0, recovery I > eps
                _, lt = _polar_tails(Q - G, 0.0, r, A, B)
            p_lt += wk * lt
            p_gt += wk * gt
            mean_I += wk * (np.einsum("nii->n", M) + h - corr)
        return p_lt, p_gt, mean_I

    def test_full(self, idx, mu_m, S_m, eps=0.0, margins=(), base=True):
        """Per-frame statistics for a Gaussian model on rows idx: posterior log-odds of onset
        (I < 0) and recovery (I > eps), their log Bayes factors against the climatology's
        prior, and the posterior mean advantage.

        margins: further recovery margins, each a per-frame array aligned with idx (e.g. the
        rate-distortion margin for a target distortion); for each, "post_rec_m{j}" and
        "lbf_rec_m{j}" are the recovery statistics for I > margin.
        base=False skips the onset / base-recovery statistics (and mean_I): only the margin
        recoveries are returned, for callers that need nothing else on these frames."""
        out = {}
        if base:
            q_lt, q_gt, mean_I = self._tails(idx, mu_m, S_m, eps, "post")
            p_lt, p_gt, _ = self._tails(idx, mu_m, S_m, eps, "prior")
            post_on, post_rec = _log_odds(q_lt), _log_odds(q_gt)
            out = {
                "post_on": post_on,
                "post_rec": post_rec,
                "lbf_on": post_on - _log_odds(p_lt),
                "lbf_rec": post_rec - _log_odds(p_gt),
                "mean_I": mean_I,
            }
        for j, mg in enumerate(margins):
            mg = np.broadcast_to(np.asarray(mg, float), (len(idx),))
            if base and not mg.any() and eps == 0.0:
                out[f"post_rec_m{j}"], out[f"lbf_rec_m{j}"] = out["post_rec"], out["lbf_rec"]
                continue
            _, q_m, _ = self._tails(idx, mu_m, S_m, mg, "post", need_lt=False)
            _, p_m, _ = self._tails(idx, mu_m, S_m, mg, "prior", need_lt=False)
            out[f"post_rec_m{j}"] = _log_odds(q_m)
            out[f"lbf_rec_m{j}"] = out[f"post_rec_m{j}"] - _log_odds(p_m)
        return out

    def test(self, idx, mu_m, S_m, eps=0.0, decision="posterior"):
        """Decision statistics (onset, recovery) and posterior mean advantage.

        decision "posterior": posterior log-odds, compared with log(p / (1 - p)); the
        climatology's prior counts as knowledge. "bayes-factor": log Bayes factors, compared
        with log k; only the movement from that prior counts."""
        v = self.test_full(idx, mu_m, S_m, eps)
        if decision == "posterior":
            return v["post_on"], v["post_rec"], v["mean_I"]
        return v["lbf_on"], v["lbf_rec"], v["mean_I"]
