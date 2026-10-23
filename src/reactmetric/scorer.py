"""REACT as published (TGF 2026, "REACT: Measuring Regime Adaptation in Trajectories").

Two steps, both shared by every model evaluated on a dataset::

    scorer = ReactScorer.fit(calib, info=(3.0, 4.2, 6.0))          # calibration tracks
    result = scorer.evaluate({"MyModel": my_forecaster}, test)       # test tracks
    result.summary()                                                 # one row per group

`fit` fits the climatology (section 3.5), the measurement-noise floor (3.7), the correlation
structure of the truth model at each lead and information level (3.2), and the overlap tables
of the exact tests. `evaluate` detects events (3.4), and for every event, lead, information
level and target distance decides onset, recovery and relapse (3.3) for each model and for
Oracle B (3.6). Every record carries the lead, the information level and the distortion, so a
REACT number always travels with its assumptions: REACT(h, d; I_traj).

Frames are 10 Hz frame counts throughout, as in the paper.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import information as ri
from .core.predictive import MixturePredictive
from .metric.survival import kaplan_meier, km_median
from .reference import fit_climatology

LEADS = (1, 5, 10, 15)
# label of the base case (distortion 0.0 in the records): no target, d = infinity
CLIM = "clim"
# events with fewer target frames before the next event (or the cap) are skipped; the shortest
# track stretch the per-track frame builders accept; the shortest calm run used for correlations
WINDOW_CAP, MIN_WINDOW, MIN_FRAMES, MIN_RUN = 150, 25, 15, 12
# calm stretches exclude [t0 - MASK_PRE, t0 + MASK_POST]
MASK_PRE, MASK_POST = 20, 60
# an onset counts only within lead + ONSET_MARGIN frames of the event
ONSET_MARGIN = 5
# Gauss-Hermite order for the mixture entropy; frames before an event used for the pre-event score
GH_NODES, PRE_WINDOW = 5, 30
# exact overlap term: Gauss-Hermite nodes per rule
OVERLAP = dict(overlap_gh=9)
# overlap tables: uniform core of +-TABLE_CORE stopped-regime sds at TABLE_STEP sds, then
# geometric growth by TABLE_GROWTH per node out to TABLE_REACH moving-regime sds
TABLE_CORE, TABLE_STEP, TABLE_GROWTH, TABLE_REACH = 8.0, 1.0 / 6.0, 1.06, 10.0
N_LAGS, FIT_LAGS = 30, 20
CORR_LEN = WINDOW_CAP + 1
# outcome fields also recorded under the mixture-entropy margin (suffix _H)
H_FIELDS = ("status", "react", "duration", "relapse", "relapse_after", "relapse3",
            "relapse3_after")

Forecasts = Union[Mapping[str, object], Callable[[], Mapping[str, object]]]


# ---------------------------------------------------------------------------
# Per-track frames
# ---------------------------------------------------------------------------


def model_frames(pred, T, lead, ref):
    """A forecast's target frames, means and covariances at one lead, or None."""
    if pred is None or len(pred) == 0:
        return None
    try:
        target, mu, S = pred.at_lead(lead)
    except ValueError:
        return None
    vw = int(getattr(ref, "velocity_window", 5))
    ok = (target < T) & (target - lead - vw >= 0)
    if ok.sum() < MIN_FRAMES:
        return None
    return target[ok].astype(int), mu[ok], S[ok] + 1e-9 * np.eye(2)


def mixture_frames(pred, T, lead, ref):
    """The full mixture of a MixturePredictive over the frames model_frames keeps (same mask
    and order), or None for a Gaussian forecast."""
    if not isinstance(pred, MixturePredictive):
        return None
    target, w, mu, cov = pred.mixture_at_lead(lead)
    ok = (target < T) & (target - lead - int(getattr(ref, "velocity_window", 5)) >= 0)
    if ok.sum() < MIN_FRAMES:
        return None
    return ri.MixtureForecast(w[ok], mu[ok], cov[ok])


def whiten(y, mu, S):
    return np.linalg.solve(np.linalg.cholesky(S), (y - mu)[..., None])[..., 0]


def calm_mask(target, events):
    mask = np.ones(len(target), bool)
    for ev in events:
        mask &= ~((target >= ev.t0 - MASK_PRE) & (target <= ev.t0 + MASK_POST))
    return mask


def runs_of(values, mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return [values[r] for r in np.split(idx, breaks + 1) if len(r) >= MIN_RUN]


def frame_logpdf(y, mu, S, fc=None):
    """Realized log density: Gaussian, or the full mixture when fc is given."""
    if fc is None:
        return ri.gaussian_logpdf(y, mu, S)
    return fc.logpdf(y)


def realized_score(y, mu_m, S_m, mu_r, S_r):
    return ri.gaussian_logpdf(y, mu_m, S_m) - ri.gaussian_logpdf(y, mu_r, S_r)


def event_kinematics(obs, t0, dt, gap=5, span=10):
    """Speed before and after an event and the heading change across it, from displacements
    over `span` frames ending `gap` frames before t0 and starting `gap` frames after it."""
    a0, a1, b0, b1 = t0 - gap - span, t0 - gap, t0 + gap, t0 + gap + span
    if a0 < 0 or b1 >= len(obs):
        return None, None, None
    va, vb = (obs[a1] - obs[a0]) / (span * dt), (obs[b1] - obs[b0]) / (span * dt)
    sa, sb = float(np.hypot(*va)), float(np.hypot(*vb))
    turn = None
    if sa > 0.2 and sb > 0.2:
        turn = float(np.degrees(abs(np.angle(complex(*vb) / complex(*va)))))
    return sa, sb, turn


def table_path(table_dir, lead, info):
    return os.path.join(table_dir, f"overlap_lead{lead}_info{info:g}")


def _build_table(job):
    ref, lead, info, c, path, sig2 = job
    t = time.time()
    v, core, h, growth, extent = ref.table_axes(
        lead, TABLE_CORE, TABLE_STEP, TABLE_GROWTH, TABLE_REACH
    )
    tab = ri.OverlapTable.build(
        ref.body_mixture(lead), c, v, core, h, growth, extent, sig2=sig2, **OVERLAP
    )
    tab.save(path)
    return lead, info, time.time() - t, tab.G.shape


# ---------------------------------------------------------------------------
# Forecasts: a mapping name -> Forecaster (``predict(traj)``) or {track id: Predictive}, or a
# zero-argument callable returning such a mapping (built once per worker process, for models
# that are expensive or awkward to pickle)
# ---------------------------------------------------------------------------

_WORKER: Dict[object, object] = {}  # per-process cache: models and overlap tables


def _cached(key, make):
    if key not in _WORKER:
        _WORKER[key] = make()
    return _WORKER[key]


def _resolve(forecasts: Forecasts):
    if callable(forecasts) and not isinstance(forecasts, Mapping):
        key = ("forecasts", hashlib.sha1(pickle.dumps(forecasts)).hexdigest())
        return _cached(key, forecasts)
    return forecasts


def _predict_all(models, traj):
    out = {}
    for name, f in models.items():
        if isinstance(f, Mapping):
            out[name] = f.get(traj.id)
            continue
        try:
            out[name] = f.predict(traj)
        except Exception:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _calib_chunk(tracks, ref, forecasts, detector, model_leads, leads):
    models = _resolve(forecasts) if forecasts is not None else {}
    names = list(models)
    nodes, gw = ri.gh_nodes_2d(GH_NODES)
    out = {
        lead: {
            "runs": [],
            "s_sum": dict.fromkeys(names, 0.0),
            "s_n": dict.fromkeys(names, 0),
            "scale_sum": 0.0,
            "scale_n": 0,
            "sn_sum": 0.0,
            "margin_samples": [],
        }
        for lead in leads
    }
    for traj in tracks.values():
        events = detector.detect(traj)
        preds = _predict_all(models, traj)
        T = len(traj.positions)
        for lead in leads:
            fr = ref.gaussian_frames(traj, lead, MIN_FRAMES)
            if fr is None:
                continue
            target, y, mu_r, S_r = fr
            calm = calm_mask(target, events)
            out[lead]["runs"].extend(runs_of(whiten(y, mu_r, S_r), calm))
            if calm.any():
                _, _, wts, mus, covs = ref.mixture_frames(traj, lead, MIN_FRAMES)
                h_mix = ri.mixture_entropy(wts[calm], mus[calm], covs[calm], nodes, gw)
                h_cov = ri.gaussian_entropy(S_r[calm])
                scale = np.exp(h_mix - h_cov)
                out[lead]["scale_sum"] += float(scale.sum())
                out[lead]["scale_n"] += int(calm.sum())
                # mean whitening scale of an isotropic unit noise, 1/2 tr S_r^-1 (floor A)
                out[lead]["sn_sum"] += float(
                    (0.5 * np.trace(np.linalg.inv(S_r[calm]), axis1=1, axis2=2)).sum()
                )
                out[lead]["margin_samples"].append(
                    np.column_stack([wts[calm][:, 1], h_mix, h_cov])[::5]
                )
            for m in names:
                ok = model_leads.get(m) is None or lead in model_leads[m]
                mf = model_frames(preds[m], T, lead, ref) if ok else None
                if mf is None:
                    continue
                tm, mu_m, S_m = mf
                _, ia, ib = np.intersect1d(target, tm, return_indices=True)
                keep = calm[ia]
                ia, ib = ia[keep], ib[keep]
                if len(ia) == 0:
                    continue
                s = realized_score(y[ia], mu_m[ib], S_m[ib], mu_r[ia], S_r[ia])
                out[lead]["s_sum"][m] += float(s.sum())
                out[lead]["s_n"][m] += len(s)
    return out


def _split_chunks(tracks, n):
    ids = list(tracks.ids) if hasattr(tracks, "ids") else list(tracks)
    return [{t: tracks[t] for t in ids[i::n]} for i in range(n) if ids[i::n]]


def _small_chunks(tracks, size):
    """Chunks of about `size` tracks (strided, so long and short tracks mix)."""
    n_ids = len(tracks.ids) if hasattr(tracks, "ids") else len(tracks)
    return _split_chunks(tracks, max(1, -(-n_ids // size)))


@dataclass
class Level:
    """One assumed information level at one lead: c = e^-I and the fitted correlations."""

    info: float
    c: float
    rho_a: np.ndarray
    rho_b: np.ndarray


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


@dataclass
class ReactScorer:
    """Everything REACT shares across the models evaluated on one dataset.

    Build with ReactScorer.fit on calibration tracks; then evaluate() or score_track()."""

    ref: object
    levels: Dict[int, List[Level]]
    sigma: float
    detector: object
    table_dir: str
    leads: Tuple[int, ...] = LEADS
    distortions: Tuple[float, ...] = (0.1, 0.3)
    p: float = 0.95
    eps: float = 0.0
    entropy_margin_info: Optional[Tuple[float, ...]] = None
    calibration: Dict = field(default_factory=dict)
    noise_floor: Dict = field(default_factory=dict)

    @property
    def log_k(self) -> float:
        return float(np.log(self.p / (1.0 - self.p)))

    # -- fitting -------------------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        calib,
        info: Sequence[float] = (3.0, 4.2, 6.0),
        *,
        leads: Sequence[int] = LEADS,
        sigma: Union[str, float] = "estimate",
        distortions: Sequence[float] = (0.1, 0.3),
        events=None,
        p: float = 0.95,
        eps: float = 0.0,
        lead_info: Optional[Mapping[int, Sequence[float]]] = None,
        entropy_margin_info: Optional[Sequence[float]] = None,
        climatology=None,
        forecasts: Optional[Forecasts] = None,
        model_leads: Optional[Mapping[str, Sequence[int]]] = None,
        table_dir: Optional[str] = None,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> ReactScorer:
        """Fit REACT's shared quantities on calibration tracks (a TrajectorySet).

        info: assumed information levels I_traj (nats); the paper reports 3, 4.2 and 6 with 4.2
            as the headline. lead_info: per-lead overrides (the paper ran lead 1 at 4.2 only).
        sigma: the measurement-noise floor, "estimate" (floor A from stationary calibration
            frames), 0 for none, or a value in metres.
        distortions: target distances d (m) besides d = infinity, which is always included.
        events: the event source; default BayesianChangepoints.tuned_sg() (paper section 3.4).
        climatology: a fitted KinematicClimatology; default fit_climatology(calib, leads).
        forecasts: optional models, scored on calm calibration stretches; their mean realized
            score (calibration[lead]["calm_mean_score"]) is the demonstrated information, a
            lower bound on I_traj.
        entropy_margin_info: levels at which outcomes are also recorded under the
            mixture-entropy distortion margin (the _H fields); default none.
        table_dir: where the overlap tables are written; default a temporary directory.
        """
        from .events import BayesianChangepoints

        leads = tuple(int(x) for x in leads)
        detector = events if events is not None else BayesianChangepoints.tuned_sg()
        model_leads = {k: tuple(v) for k, v in (model_leads or {}).items()}
        ref = climatology if climatology is not None else fit_climatology(
            calib, leads=leads, dt=calib.dt, seed=seed
        )

        if isinstance(sigma, str):
            if sigma.lower() not in ("estimate", "a"):
                raise ValueError(f"sigma must be 'estimate', 0 or metres, not {sigma!r}")
            sig, n_res, n_tr = ri.estimate_noise_floor(calib)
            noise_fit = {"floor": "A", "sigma_cm": 100 * sig, "residuals_per_axis": n_res,
                         "tracks": n_tr}
        elif float(sigma) == 0.0:
            sig, noise_fit = 0.0, {"floor": "none"}
        else:
            sig, noise_fit = float(sigma), {"floor": "given"}
        noise_fit["sigma_cm"] = 100 * sig
        sig2 = sig**2

        chunks = _split_chunks(calib, max(1, n_jobs))
        args = (ref, forecasts, detector, model_leads, leads)
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                parts = list(pool.map(_calib_chunk, chunks, *[[a] * len(chunks) for a in args]))
        else:
            parts = [_calib_chunk(ch, *args) for ch in chunks]
        names = list(parts[0][leads[0]]["s_sum"]) if parts else []

        calibration, levels = {}, {}
        for lead in leads:
            runs = [r for pt in parts for r in pt[lead]["runs"]]
            rho_w, ms = ri.pooled_autocorr(runs, N_LAGS)
            s_mean = {
                m: sum(pt[lead]["s_sum"][m] for pt in parts)
                / max(sum(pt[lead]["s_n"][m] for pt in parts), 1)
                for m in names
            }
            rho_b = ri.overlap_corr(lead, CORR_LEN)
            # entropic information level: the calm-frame truth fraction is c times the mean
            # entropy scale of the mixture reference
            s_bar = sum(pt[lead]["scale_sum"] for pt in parts) / max(
                sum(pt[lead]["scale_n"] for pt in parts), 1
            )
            # mean whitened variance of the isotropic measurement noise on calm frames
            sn_bar = sig2 * sum(pt[lead]["sn_sum"] for pt in parts) / max(
                sum(pt[lead]["scale_n"] for pt in parts), 1
            )
            levels[lead], fits = [], {}
            lv = info if lead_info is None or lead not in lead_info else lead_info[lead]
            for I in lv:
                c = ri.info_to_c(I)
                scale, tau, omega = ri.fit_knowable_corr(
                    rho_w, rho_b, c * s_bar, FIT_LAGS, noise=sn_bar
                )
                levels[lead].append(
                    Level(float(I), c, ri.knowable_corr(CORR_LEN, scale, tau, omega), rho_b)
                )
                fits[f"{I:.3f}"] = {"c": c, "scale": scale, "tau": tau, "omega": omega}
            ms_ = np.concatenate([x for pt in parts for x in pt[lead]["margin_samples"]])
            stand = ms_[:, 0] > 0.5
            margin_med = {}
            for d in distortions:
                row = {}
                for name, sel in (("standing", stand), ("moving", ~stand)):
                    row[name] = {
                        "n": int(sel.sum()),
                        "entropy_cov": float(np.median(ri.distortion_margin(ms_[sel, 2], d))),
                        "entropy_mix": float(np.median(ri.distortion_margin(ms_[sel, 1], d))),
                    }
                margin_med[f"{d:g}"] = row
            calibration[lead] = {
                "calm_mean_score": s_mean,
                "reference_mean_square": ms,
                "rho_w": rho_w[1:11].tolist(),
                "calm_frames": int(sum(len(r) for r in runs)),
                "entropy_scale_calm": s_bar,
                "noise_white_variance": sn_bar,
                "margin_median_nats": margin_med,
                "fits": fits,
            }

        table_dir = table_dir or tempfile.mkdtemp(prefix="react_tables_")
        os.makedirs(table_dir, exist_ok=True)
        jobs = [
            (ref, lead, lv.info, lv.c, table_path(table_dir, lead, lv.info), sig2)
            for lead in leads
            for lv in levels[lead]
        ]
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                built = list(pool.map(_build_table, jobs))
        else:
            built = [_build_table(j) for j in jobs]
        timing = {f"L{b[0]}/{b[1]:g}": b[2] for b in built}

        ent = None if entropy_margin_info is None else tuple(
            round(float(x), 4) for x in entropy_margin_info
        )
        out = cls(
            ref=ref, levels=levels, sigma=sig, detector=detector, table_dir=table_dir,
            leads=leads, distortions=tuple(float(d) for d in distortions), p=float(p),
            eps=float(eps), entropy_margin_info=ent, calibration=calibration,
            noise_floor=noise_fit,
        )
        out.table_seconds = timing
        return out

    # -- scoring -------------------------------------------------------------------------------

    def _tables(self):
        return {
            (lead, lv.info): _cached(
                ("table", table_path(self.table_dir, lead, lv.info)),
                lambda p_=table_path(self.table_dir, lead, lv.info): ri.OverlapTable.load(p_),
            )
            for lead in self.leads
            for lv in self.levels.get(lead, ())
        }

    def score_track(self, traj, predictives: Mapping[str, object], events=None,
                    model_leads: Optional[Mapping[str, Sequence[int]]] = None, clock=None):
        """REACT records for every event on one track: one row per (lead, level, distortion,
        model, event), Oracle B included. predictives: name -> Predictive (or None).

        events: the track's events (default: this scorer's detector)."""
        ref = self.ref
        model_leads = model_leads or {}
        events = sorted(self.detector.detect(traj) if events is None else events,
                        key=lambda e: e.t0)
        clock = clock if clock is not None else {}
        for k in ("tests shared", "tests per model", "off-table points", "events", "model rows"):
            clock.setdefault(k, 0)
        records = []
        if not events:
            return records
        tables = self._tables()
        nodes, gw = ri.gh_nodes_2d(GH_NODES)
        sig2 = float(self.sigma) ** 2
        eps, log_k = self.eps, self.log_k
        dd = tuple(self.distortions)
        nd = len(dd)
        names = list(predictives)
        T = len(traj.positions)
        tid = traj.id
        obs_all = np.asarray(traj.positions, float)
        kin = {ev.t0: event_kinematics(obs_all, ev.t0, traj.dt) for ev in events}
        for lead in self.leads:
            rm = ref.mixture_frames(traj, lead, MIN_FRAMES)
            if rm is None or not self.levels.get(lead):
                continue
            target, y, wts, mus, covs = rm
            speed, heading, anchor = ref.frame_geometry(traj, lead, target)
            mu_r, S_r = ri.mixture_moments(wts, mus, covs)
            w = whiten(y, mu_r, S_r)
            H_ref = ri.mixture_entropy(wts, mus, covs, nodes, gw)
            H_cov = ri.gaussian_entropy(S_r)
            scale = np.exp(H_ref - H_cov)
            # whitened variance of the isotropic measurement noise sigma^2 I (floor A)
            sn_fr = sig2 * 0.5 * np.trace(np.linalg.inv(S_r), axis1=1, axis2=2)
            # default margin: Gaussian entropy of the mixture's total covariance; "_H"
            # alternative: the mixture's own entropy
            margin = {d: ri.distortion_margin(H_cov, d) for d in dd}
            margin_H = {d: ri.distortion_margin(H_ref, d) for d in dd}
            ref_lp = ri.mixture_logpdf(y[:, None], ri.mixture_prep(wts, mus, covs))[:, 0]
            mfr = {
                m: model_frames(predictives[m], T, lead, ref)
                if model_leads.get(m) is None or lead in model_leads[m] else None
                for m in names
            }
            mixfc = {
                m: mixture_frames(predictives[m], T, lead, ref) if mfr[m] is not None else None
                for m in names
            }
            for k_ev, ev in enumerate(events):
                nxt = events[k_ev + 1].t0 if k_ev + 1 < len(events) else np.inf
                sel = (target > ev.t0) & (target < nxt) & (target <= ev.t0 + WINDOW_CAP)
                if sel.sum() < MIN_WINDOW:
                    continue
                fw, yw, mrw, Srw, ww = target[sel], y[sel], mu_r[sel], S_r[sel], w[sel]
                pre = (target >= ev.t0 - PRE_WINDOW) & (target < ev.t0)
                if k_ev > 0:
                    pre &= target > events[k_ev - 1].t0
                clock["events"] += 1
                for lv in self.levels[lead]:
                    info, c, rho_a, rho_b = lv.info, lv.c, lv.rho_a, lv.rho_b
                    with_h = (self.entropy_margin_info is None
                              or round(float(info), 4) in self.entropy_margin_info)
                    ct = c * scale[sel]
                    sn_t = sn_fr[sel]
                    va = ri.knowable_share(ct, sn_t)[:, None, None]
                    mf_, vf_, ms_, vs_ = ri.truth_posterior(fw, ww, rho_a, rho_b, ct, sn_t)
                    pmu, pC = ri.to_world(mf_, vf_, mrw, Srw)
                    smu, sC = ri.to_world(ms_, vs_, mrw, Srw)
                    t_ = time.time()
                    targs = (pmu, pC, mrw, va * Srw, wts[sel], mus[sel], covs[sel], c)
                    tests = ri.RegimeTests(
                        *targs, **OVERLAP, table=tables[(lead, info)],
                        geom=(speed[sel], heading[sel], anchor[sel]), sig2=sig2,
                    )
                    clock["off-table points"] += tests.n_off
                    clock["tests shared"] += time.time() - t_
                    # Oracle B: the hindsight posterior predictive under the regime truth model
                    ob_mu, ob_S = ri.regime_predictive(
                        smu, sC, mrw, va * Srw, wts[sel], mus[sel], covs[sel], c, sig2
                    )
                    mtests = (
                        ri.RegimeTestsMixture.from_exact(tests, gh=5, stride=(8, 2))
                        if any(f is not None for f in mixfc.values())
                        else None
                    )
                    rows = [("oracle_B", np.arange(len(fw)), ob_mu, ob_S, None, None)]
                    for m in names:
                        if mfr[m] is None:
                            continue
                        tm, mu_m, S_m = mfr[m]
                        _, ia, ib = np.intersect1d(fw, tm, return_indices=True)
                        if len(ia) >= MIN_WINDOW:
                            fcm = mixfc[m]
                            rows.append(
                                (
                                    m, ia, mu_m[ib], S_m[ib], (tm, mu_m, S_m, fcm),
                                    fcm.take(ib) if fcm is not None else None,
                                )
                            )
                    for m, ia, mu_x, S_x, full, fc_x in rows:
                        clock["model rows"] += 1
                        records.extend(self._event_rows(
                            ev, lead, info, m, ia, mu_x, S_x, full, fc_x, tests, mtests,
                            fw, yw, mrw, Srw, y, target, pre, ref_lp, sel, margin, margin_H,
                            with_h, kin, tid, eps, log_k, clock, dd, nd,
                        ))
        return records

    def _event_rows(self, ev, lead, info, m, ia, mu_x, S_x, full, fc_x, tests, mtests, fw,
                    yw, mrw, Srw, y, target, pre, ref_lp, sel, margin, margin_H, with_h, kin,
                    tid, eps, log_k, clock, dd, nd):
        # recovery channels: 0 the base, 1..nd one per distortion under the default
        # (covariance) margin, then the entropy ("_H") margin for those distortions where it
        # differs from it on this window (hch[j]: the channel behind distortion j's _H outcome)
        mg = [None] + [margin[d][sel][ia] for d in dd]
        hch = list(range(nd + 1))
        if with_h:
            for jd, d in enumerate(dd):
                mh = margin_H[d][sel][ia]
                if not np.allclose(mg[jd + 1], mh, atol=1e-3):
                    hch[jd + 1] = len(mg)
                    mg.append(mh)

        def evaluate(idx, chs, need_on):
            t_ = time.time()
            base = need_on or 0 in chs
            mch = [ch for ch in chs if ch != 0]
            margs = [mg[ch][idx] for ch in mch]
            if fc_x is None:
                v = tests.test_full(ia[idx], mu_x[idx], S_x[idx], eps, margins=margs, base=base)
            else:
                v = mtests.test_full_mixture(ia[idx], fc_x.take(idx), eps, margins=margs,
                                             base=base)
            clock["tests per model"] += time.time() - t_
            sfx = ["" if ch == 0 else f"_m{mch.index(ch)}" for ch in chs]
            out = {
                "rec": np.array([v[f"post_rec{s}"] for s in sfx]).reshape(len(chs), len(idx)),
                "lbf_rec": np.array([v[f"lbf_rec{s}"] for s in sfx]).reshape(
                    len(chs), len(idx)),
            }
            if base:
                out.update(on=v["post_on"], lbf_on=v["lbf_on"], mean_I=v["mean_I"])
            return out

        F = fw[ia]
        oracle = m == "oracle_B"
        outs, fas, st = ri.classify_lazy(
            evaluate, F, ev.t0, log_k, len(mg), lead + ONSET_MARGIN, oracle=oracle
        )
        k_on = st["k"]  # mean I over the onset-window frames, tested for all
        I_sum = float(np.nansum(st["mean_I"][:k_on]))
        lp = frame_logpdf(yw[ia], mu_x, S_x, fc_x)
        pre_sum, pre_n = 0.0, 0
        if full is not None and pre.any():
            tm, mu_m, S_m, fc_f = full
            _, pa, pb = np.intersect1d(target[pre], tm, return_indices=True)
            if len(pa):
                pre_sum = float(
                    (
                        frame_logpdf(
                            y[pre][pa],
                            mu_m[pb],
                            S_m[pb],
                            fc_f.take(pb) if fc_f is not None else None,
                        )
                        - ref_lp[pre][pa]
                    ).sum()
                )
                pre_n = int(len(pa))
        sb, sa, turn = kin[ev.t0]
        last = int(F[-1] - ev.t0)
        rows = []
        for j, dist in enumerate((0.0,) + dd):
            if oracle:
                # the inferred truth is never behind the climatology; its REACT is the earliest
                # frame its advantage can be confirmed (the floor)
                fa, fah = fas[j], fas[hch[j]]
                first_ahead = None if fa is None else int(F[fa] - ev.t0)
                first_ahead_H = None if fah is None else int(F[fah] - ev.t0)
                res = ri.oracle_outcome(first_ahead, last)
                res_h = ri.oracle_outcome(first_ahead_H, last) if with_h else None
                on_pd = None
                rec_pd = bool(st["lbf_rec"][j, fa] < 0) if fa is not None else None
            else:
                first_ahead = first_ahead_H = None  # only Oracle B needs them
                res = outs[j]
                res_h = outs[hch[j]] if with_h else None
                # prior-driven verdicts: the decision fired although the trajectory's evidence
                # pointed the other way (Bayes factor < 1)
                o_ = st["onset"]
                on_pd = bool(st["lbf_on"][o_] < 0) if o_ is not None else None
                r_ = st["react"][j]
                rec_pd = bool(st["lbf_rec"][j, r_] < 0) if r_ is not None else None
            rows.append(
                {
                    "lead": lead,
                    "info": round(float(info), 4),
                    "distortion": float(dist),
                    "distortion_label": f"{dist:g}m" if dist else CLIM,
                    "model": m,
                    "track": str(tid),
                    "t0": int(ev.t0),
                    "event_type": ev.event_type,
                    "speed_before": sb,
                    "speed_after": sa,
                    "heading_change": turn,
                    "effect_speed": float(ev.d_speed_level),
                    "margin_mean": float(margin[dist][sel][ia].mean()) if dist else 0.0,
                    "margin_mean_H": float(margin_H[dist][sel][ia].mean()) if dist else 0.0,
                    "n": int(len(ia)),
                    **res,
                    **{f"{k}_H": (res_h[k] if res_h else None) for k in H_FIELDS},
                    "s_mix_sum": float((lp - ref_lp[sel][ia]).sum()),
                    "s_sum": float((lp - ri.gaussian_logpdf(yw[ia], mrw[ia], Srw[ia])).sum()),
                    "I_sum": I_sum,
                    "I_n": int(k_on),
                    "first_ahead": first_ahead,
                    "first_ahead_H": first_ahead_H,
                    "onset_prior_driven": on_pd,
                    "recovery_prior_driven": rec_pd,
                    "pre_s_sum": pre_sum,
                    "pre_n": pre_n,
                }
            )
        return rows

    def evaluate(self, forecasts: Forecasts, data, *, model_leads=None, n_jobs: int = 1,
                 chunk_tracks: int = 5, progress: Optional[Callable[[int, int], None]] = None):
        """Score every model on every event of the test tracks (a TrajectorySet).

        forecasts: name -> Forecaster or {track id: Predictive}, or a zero-argument callable
        returning that mapping (built once per worker). model_leads: name -> leads at which
        that model is scored (default: all). Returns a ReactResult."""
        model_leads = {k: tuple(v) for k, v in (model_leads or {}).items()}
        if isinstance(forecasts, Mapping):
            _check_ids(forecasts, data)
        chunks = _small_chunks(data, chunk_tracks)
        results, clock = {}, {}
        t0 = time.time()
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futs = {pool.submit(_score_chunk, self, ch, forecasts, model_leads): i
                        for i, ch in enumerate(chunks)}
                for n_done, fut in enumerate(as_completed(futs), 1):
                    recs, clk = fut.result()
                    results[futs[fut]] = recs
                    for k_, v_ in clk.items():
                        clock[k_] = clock.get(k_, 0) + v_
                    if progress is not None:
                        progress(n_done, len(chunks))
        else:
            for i, ch in enumerate(chunks):
                recs, clk = _score_chunk(self, ch, forecasts, model_leads)
                results[i] = recs
                for k_, v_ in clk.items():
                    clock[k_] = clock.get(k_, 0) + v_
                if progress is not None:
                    progress(i + 1, len(chunks))
        clock["wall"] = time.time() - t0
        records = [r for i in sorted(results) for r in results[i]]
        return ReactResult(records=records, scorer=self, timing=clock)


def _check_ids(forecasts, data):
    """Precomputed forecasts keyed by ids that match no test track score nothing; say so."""
    ids = set(data.ids)
    for name, f in forecasts.items():
        if isinstance(f, Mapping) and f and not ids & set(f):
            raise ValueError(
                f"forecasts for {name!r} are keyed by ids that match no test track "
                f"(e.g. {next(iter(f))!r}; tracks look like {next(iter(ids))!r})"
            )


def _score_chunk(scorer, tracks, forecasts, model_leads):
    models = _resolve(forecasts)
    clock = {"predict": 0.0}
    t_chunk = time.time()
    records = []
    for traj in tracks.values():
        events = sorted(scorer.detector.detect(traj), key=lambda e: e.t0)
        if not events:
            continue
        t_ = time.time()
        preds = _predict_all(models, traj)
        clock["predict"] += time.time() - t_
        records.extend(scorer.score_track(traj, preds, events, model_leads, clock))
    clock["chunk total"] = time.time() - t_chunk
    return records, clock


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _rmst(t_, s_):
    t_ = np.append(np.minimum(t_, WINDOW_CAP), WINDOW_CAP)
    return float(np.sum(np.diff(t_) * s_))


def km_summary(rs, sfx=""):
    """Anticipation and REACT for one group under one outcome rule (paper section 3.3).

    Given onset: Kaplan-Meier over events with an onset (time to recovery given the forecast
    fell behind); median REACT is its median. Over all events: anticipated ones (no onset)
    recover at frame 0. Besides medians, the restricted mean (area under the Kaplan-Meier curve
    up to WINDOW_CAP frames; Royston & Parmar 2013), which stays informative when over half the
    events are anticipated."""
    if not rs:  # e.g. the "_H" outcomes at levels where the entropy margin was not computed
        return dict.fromkeys(
            ("recovered", "censored", "better", "indistinguishable", "anticipation", "km_react",
             "km_react_overall", "rmst", "rmst_cond"), float("nan"))
    st = np.array([r["status" + sfx] for r in rs])
    dur = np.array(
        [np.nan if r["duration" + sfx] in (None, "") else float(r["duration" + sfx]) for r in rs]
    )
    onset = np.isin(st, ("recovered", "censored"))
    out = {
        "recovered": float(np.mean(st == "recovered")),
        "censored": float(np.mean(st == "censored")),
        "better": float(np.mean(st == "better")),
        "indistinguishable": float(np.mean(st == "indistinguishable")),
        "anticipation": float(np.mean(~onset)),
        "km_react": float("nan"),
        "km_react_overall": float("nan"),
        "rmst": float("nan"),
        "rmst_cond": float("nan"),
    }
    if onset.any():
        t_, s_ = kaplan_meier(dur[onset], st[onset] == "recovered")
        out["km_react"] = float(km_median(t_, s_))
        out["rmst_cond"] = _rmst(t_, s_)
    d_all = np.where(onset, dur, 0.0)
    obs_all = (st == "recovered") | ~onset
    t_, s_ = kaplan_meier(d_all, obs_all)
    out["km_react_overall"] = float(km_median(t_, s_))
    out["rmst"] = _rmst(t_, s_)
    return out


def summarize(records):
    """Per (lead, information level, distortion, model) group: the REACT profile.

    Relapse rates are over recovered events, within RELAPSE_WINDOW frames after recovery;
    mean_I is over the onset-window frames (tested for every event)."""
    groups = {}
    for r in records:
        groups.setdefault((r["lead"], r["info"], r.get("distortion", 0.0), r["model"]),
                          []).append(r)
    out = {}
    for key, rs in groups.items():
        onset = [r for r in rs if r["status"] in ("recovered", "censored")]
        rec = [r for r in rs if r["status"] == "recovered"]
        n_frames = sum(r["n"] for r in rs)
        n_pre = sum(r["pre_n"] for r in rs)
        n_I = sum(r["I_n"] for r in rs)
        rs_h = [r for r in rs if r["status_H"] is not None]
        under_h = {f"{k}_H": v for k, v in km_summary(rs_h, "_H").items()}
        rec_h = [r for r in rs_h if r["status_H"] == "recovered"]
        on0 = [r["onset"] for r in onset if r["onset"] is not None]

        def share(xs, k):
            return float(np.mean([r[k] for r in xs])) if xs else float("nan")

        def med_after(xs, k):
            v = [r[f"{k}_after"] for r in xs if r[k]]
            return float(np.median(v)) if v else float("nan")

        out[key] = {
            "events": len(rs),
            **km_summary(rs),
            **under_h,
            "median_onset": float(np.median(on0)) if on0 else float("nan"),
            "relapse": share(rec, "relapse"),
            "relapse_after_median": med_after(rec, "relapse"),
            "relapse3": share(rec, "relapse3"),
            "relapse3_after_median": med_after(rec, "relapse3"),
            "relapse_H": share(rec_h, "relapse_H"),
            "relapse3_H": share(rec_h, "relapse3_H"),
            "margin_mean": sum(r["margin_mean"] * r["n"] for r in rs) / n_frames,
            "margin_mean_H": sum(r["margin_mean_H"] * r["n"] for r in rs) / n_frames,
            "mean_s_mixture": sum(r["s_mix_sum"] for r in rs) / n_frames,
            "mean_s_gaussian": sum(r["s_sum"] for r in rs) / n_frames,
            "mean_pre_s": sum(r["pre_s_sum"] for r in rs) / n_pre if n_pre else float("nan"),
            "mean_I": sum(r["I_sum"] for r in rs) / n_I if n_I else float("nan"),
        }
    return out


@dataclass
class ReactResult:
    """Per-event REACT records and their summaries.

    records: one dict per (lead, information level, distortion, model, event). summary():
    one REACT profile per group: anticipation, censoring, recovery (Kaplan-Meier medians and
    restricted means, over all events and given onset), relapse."""

    records: List[Dict]
    scorer: Optional[ReactScorer] = None
    timing: Dict = field(default_factory=dict)

    def summary(self) -> Dict[Tuple, Dict]:
        return summarize(self.records)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.records)

    def table(self, lead=None, info=None, distortion=0.0):
        """The summary as a DataFrame, optionally filtered."""
        import pandas as pd

        rows = [
            {"lead": k[0], "info": k[1], "distortion": k[2], "model": k[3], **v}
            for k, v in self.summary().items()
        ]
        df = pd.DataFrame(rows)
        if lead is not None:
            df = df[df["lead"] == lead]
        if info is not None:
            df = df[np.isclose(df["info"], info)]
        if distortion is not None:
            df = df[np.isclose(df["distortion"], distortion)]
        return df.reset_index(drop=True)
