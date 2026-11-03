"""The climatology's overlap term G_k: computed on a grid of truth locations, or tabulated.

Paper section 3.2. G_k does not depend on
the model, so it is computed once per frame and regime and shared by every model.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .densities import _EYE2, gh_nodes_2d, mixture_prep


def _quad_in_z(D0, Dz, P, feats):
    """(d)' P (d) with d = D0 + Dz z at points z: D0 (n, G, 2), Dz (n, 2, 2), P (n, 2, 2).

    feats = [z1^2, 2 z1 z2, z2^2, z1, z2] (Pz, 5). Returns (n, G, Pz)."""
    Mq = np.einsum("nji,njk,nkl->nil", Dz, P, Dz)
    quad = feats[:, :3] @ np.stack([Mq[:, 0, 0], Mq[:, 0, 1], Mq[:, 1, 1]], 0)  # (Pz, n)
    PD0 = np.einsum("nij,ngj->ngi", P, D0)
    lin = 2.0 * np.einsum("nji,ngj->ngi", Dz, PD0)
    const = np.einsum("ngi,ngi->ng", D0, PD0)
    return quad.T[:, None, :] + np.einsum("pf,ngf->ngp", feats[:, 3:], lin) + const[..., None]


def overlap_on_grid(loc, C, weights, means, covs, k, c, z, nodes, gw, blob_scale=(1.0,), sig2=0.0):
    """G_k at m = loc + chol(C) z for points z (P, 2); per-frame loc (n, 2), C (n, 2, 2).

    G_k(m) = E_{y ~ N(m, O)} log(1 + sum_j w_j N_j(y) / (w_k N_k(y))), O = c S_k + sig2 I. When
    another regime is much narrower than O (the stopped regime inside a moving truth), the integrand
    is a small blob inside a wide Gaussian that fixed nodes miss. So the expectation is taken
    with two Gauss-Hermite rules combined by the balance heuristic: one on q_0 = N(m, O), one
    per other regime on q_j proportional to N(y; m, O) N_j(y) (the blob), each node weighted by
    q_0 / sum_r q_r. Every node position is affine in z, so every log-density is a quadratic in
    z and the whole evaluation is exact algebra plus the quadrature.
    """
    n, K = weights.shape
    Lc = np.linalg.cholesky(C + 1e-15 * _EYE2)
    O = c * covs[:, k] + sig2 * _EYE2 + 1e-15 * _EYE2
    Oi = np.linalg.inv(O)
    _, inv, lognorm = mixture_prep(weights, means, covs)
    feats = np.stack([z[:, 0] ** 2, 2.0 * z[:, 0] * z[:, 1], z[:, 1] ** 2, z[:, 0], z[:, 1]], 1)
    others = [j for j in range(K) if j != k]
    # rules: (Y0 (n, G, 2), Yz (n, 2, 2), centre offset mu0 (n, 2), centre slope Mz, cov)
    rules = [
        (loc[:, None] + np.einsum("nij,gj->ngi", np.linalg.cholesky(O), nodes), Lc, loc, Lc, O)
    ]
    for j in others:
        Sj = np.linalg.inv(Oi + inv[:, j])
        A = np.einsum("nij,njk->nik", Sj, Oi)
        mu0 = np.einsum("nij,nj->ni", A, loc) + np.einsum(
            "nij,njk,nk->ni", Sj, inv[:, j], means[:, j]
        )
        Mz = np.einsum("nij,njk->nik", A, Lc)
        for bs in blob_scale:
            Sb = bs * Sj
            Y0 = mu0[:, None] + np.einsum("nij,gj->ngi", np.linalg.cholesky(Sb), nodes)
            rules.append((Y0, Mz, mu0, Mz, Sb))
    cov_inv = [np.linalg.inv(r[4]) for r in rules]
    cov_logdet = [np.linalg.slogdet(r[4])[1] for r in rules]
    total = 0.0
    for Y0, Yz, _, _, _ in rules:
        # log-ratios l_j = log w_j N_j(y) - log w_k N_k(y)
        Qk = _quad_in_z(Y0 - means[:, k][:, None], Yz, inv[:, k], feats)
        ls = [
            0.5 * Qk - 0.5 * _quad_in_z(Y0 - means[:, j][:, None], Yz, inv[:, j], feats)
            + (lognorm[:, j] - lognorm[:, k])[:, None, None]
            for j in others
        ]
        L = np.stack(ls, 0)
        peak = np.maximum(L.max(0), 0.0)
        gap = peak + np.log(np.exp(-peak) + np.exp(L - peak).sum(0))
        # balance-heuristic weight q_0 / sum_r q_r at these nodes
        lq = np.stack([
            -0.5 * _quad_in_z(Y0 - mu0[:, None], Yz - Mz, cov_inv[r], feats)
            - 0.5 * cov_logdet[r][:, None, None]
            for r, (_, _, mu0, Mz, _) in enumerate(rules)
        ], 0)
        wt = np.exp(lq[0] - (lq.max(0) + np.log(np.exp(lq - lq.max(0)).sum(0))))
        total = total + np.einsum("ngp,g->np", gap * wt, gw)
    return total


class OverlapTable:
    """The overlap term G_k tabulated once per lead and information level.

    In the mover's body frame (origin at the last position, x along the heading) the
    climatology depends on the speed alone, so G_k(m) is a function of (speed, m_body). The
    table stores it on a speed grid times a square body-frame grid that is uniform (spacing h)
    within +-core of the origin, where the stopped regime and the sharp part of G sit, and
    grows geometrically (ratio `growth`) out to +-extent, where G varies slowly. The position
    index is a closed-form function of m, so a lookup is arithmetic plus one C interpolation
    call (trilinear in speed and position). Points beyond the table are computed directly.

    Build with OverlapTable.build(body, c, ...), where body(v) returns the climatology's
    weights (V, K), means (V, K, 2) and covs (V, K, 2, 2) in the body frame at speeds v.
    """

    def __init__(self, v, core, h, growth, X, G, c, sig2=0.0):
        self.v, self.core, self.h, self.growth, self.X = np.asarray(v, float), core, h, growth, X
        self.G, self.c, self.sig2 = G, float(c), float(sig2)
        self.K = G.shape[0]

    def _index(self, m):
        am = np.abs(m)
        a, h, g = self.core, self.h, self.growth
        with np.errstate(invalid="ignore"):
            far = a / h + np.log1p(np.maximum(am - a, 0.0) * (g - 1.0) / h) / np.log(g)
        return np.sign(m) * np.where(am <= a, am / h, far) + self.X

    @staticmethod
    def axis(core, h, growth, extent):
        """Node positions of one axis and its half-length X in index units."""
        X = int(np.ceil(core / h + np.log1p((extent - core) * (growth - 1.0) / h) / np.log(growth)))
        x = np.arange(-X, X + 1, dtype=float)
        ax = np.abs(x)
        far = core + h * (growth ** (ax - core / h) - 1.0) / (growth - 1.0)
        m = np.where(ax <= core / h, ax * h, far)
        return np.sign(x) * m, X

    @classmethod
    def build(cls, body, c, v_grid, core, h, growth, extent, overlap_gh=9, blob_scale=(4.0,),
              chunk=4096, sig2=0.0):
        nodes_ax, X = cls.axis(core, h, growth, extent)
        M1, M2 = np.meshgrid(nodes_ax, nodes_ax, indexing="ij")
        pts = np.column_stack([M1.ravel(), M2.ravel()])
        w, mu, S = body(np.asarray(v_grid, float))
        V, K = w.shape
        nodes, gw = gh_nodes_2d(overlap_gh)
        G = np.empty((K, V, len(nodes_ax), len(nodes_ax)), np.float32)
        zero, eye = np.zeros((1, 2)), np.eye(2)[None]
        for k in range(K):
            for j in range(V):
                out = np.empty(len(pts))
                for s0 in range(0, len(pts), chunk):
                    out[s0 : s0 + chunk] = overlap_on_grid(
                        zero, eye, w[j : j + 1], mu[j : j + 1], S[j : j + 1], k, c,
                        pts[s0 : s0 + chunk], nodes, gw, blob_scale=blob_scale, sig2=sig2,
                    )[0]
                G[k, j] = out.reshape(len(nodes_ax), len(nodes_ax))
        return cls(v_grid, core, h, growth, X, G, c, sig2)

    def save(self, path):
        np.save(path + ".npy", self.G)
        np.savez(path + "_meta.npz", v=self.v, core=self.core, h=self.h, growth=self.growth,
                 X=self.X, c=self.c, sig2=self.sig2)

    @classmethod
    def load(cls, path, mmap=True):
        meta = np.load(path + "_meta.npz")
        G = np.load(path + ".npy", mmap_mode="r" if mmap else None)
        return cls(meta["v"], float(meta["core"]), float(meta["h"]), float(meta["growth"]),
                   int(meta["X"]), G, float(meta["c"]),
                   float(meta["sig2"]) if "sig2" in meta.files else 0.0)

    def lookup(self, k, speed, heading, anchor, m):
        """G_k at world points m (n, P, 2) given per-frame speed, heading and anchor (n,).

        Returns the values (n, P) and a mask of points beyond the table."""
        cs, sn = np.cos(heading)[:, None], np.sin(heading)[:, None]
        dx, dy = m[..., 0] - anchor[:, None, 0], m[..., 1] - anchor[:, None, 1]
        x1 = self._index(cs * dx + sn * dy)  # body frame: R' (m - anchor)
        x2 = self._index(-sn * dx + cs * dy)
        N = 2 * self.X
        off = (x1 < 0) | (x1 > N) | (x2 < 0) | (x2 > N)
        vi = np.interp(np.clip(speed, self.v[0], self.v[-1]), self.v, np.arange(len(self.v)))
        coords = np.stack([np.broadcast_to(vi[:, None], x1.shape), x1, x2]).reshape(3, -1)
        vals = map_coordinates(self.G[k], coords, order=1, mode="nearest", prefilter=False)
        return vals.reshape(x1.shape).astype(float), off
