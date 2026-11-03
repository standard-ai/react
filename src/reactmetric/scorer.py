"""REACT as published (TGF 2026, "REACT: Measuring Regime Adaptation in Trajectories").

Two steps, both shared by every model evaluated on a dataset::

    scorer = ReactScorer.fit(calib, info=(3.0, 4.2, 6.0))          # calibration tracks
    result = scorer.evaluate({"MyModel": my_forecaster}, test)       # test tracks
    result.table()                                                   # one row per model

`fit` fits the climatology (section 3.5), the measurement-noise floor (3.7), the correlation
structure of the truth model at each lead and information level (3.2), and the overlap tables
of the exact tests. `evaluate` detects events (3.4), and for every event, lead, information
level and target distance decides onset, recovery and relapse (3.3) for each model and for
Oracle B (3.6). Every record carries the lead, the information level and the distortion, so a
REACT number always travels with its assumptions: REACT(h, d; I_traj).

Leads and windows are counts of observations (frames). The defaults are the paper's, chosen
for 10 Hz data; at another frame rate, set them for the time spans you want (see
ReactScorer.fit).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import tempfile
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import information as ri
from .core.predictive import MixturePredictive
from .reference import fit_climatology
from .results import (  # noqa: F401  (km_summary, summarize: re-exported for the experiments)
    CLIM,
    ReactResult,
    km_summary,
    select_info_level,
    summarize,
)

__all__ = ["ReactScorer", "ReactResult", "select_info_level"]

LEADS = (1, 5, 10, 15)
# the paper's windows, in observations (10 Hz): an event window ends WINDOW_CAP frames after the
# event or at the next event; events with fewer than MIN_WINDOW frames are skipped; an onset
# counts within lead + ONSET_MARGIN frames; calm stretches exclude [t0 - MASK_PRE, t0 + MASK_POST];
# the pre-event score uses the PRE_WINDOW frames before an event
WINDOW_CAP, MIN_WINDOW, ONSET_MARGIN = 150, 25, 5
MASK_PRE, MASK_POST, PRE_WINDOW = 20, 60, 30
RELAPSE_WINDOW, RELAPSE_RUN = ri.RELAPSE_WINDOW, ri.RELAPSE_RUN
# the shortest track stretch the per-track frame builders accept; the shortest calm run used
# for correlations
MIN_FRAMES, MIN_RUN = 15, 12
# Gauss-Hermite order for the mixture entropy
GH_NODES = 5
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
# a model whose realized score averages this far below the climatology (nats per frame) is
# probably degenerate (collapsed or misaligned covariances)
DEGENERATE_SCORE = -10.0

ForecastSource = Union[Mapping[str, object], Callable[[], Mapping[str, object]]]


# ---------------------------------------------------------------------------
# Per-track frames
# ---------------------------------------------------------------------------


def _lead_frames(pred, T, lead, ref):
    """Target frames, means and covariances at one lead with the mask of usable frames:
    inside the track, with enough history for the climatology, and finite (non-finite forecast
    frames count as missing)."""
    try:
        target, mu, S = pred.at_lead(lead)
    except ValueError:
        return None
    vw = int(getattr(ref, "velocity_window", 5))
    ok = (target < T) & (target - lead - vw >= 0)
    ok &= np.isfinite(mu).all(1) & np.isfinite(S).all((1, 2))
    if isinstance(pred, MixturePredictive):
        _, w, mus, covs = pred.mixture_at_lead(lead)
        ok &= np.isfinite(w).all(1) & np.isfinite(mus).all((1, 2)) & np.isfinite(covs).all(
            (1, 2, 3))
    return target, mu, S, ok


def model_frames(pred, T, lead, ref):
    """A forecast's target frames, means and covariances at one lead, or None."""
    if pred is None or len(pred) == 0:
        return None
    fr = _lead_frames(pred, T, lead, ref)
    if fr is None:
        return None
    target, mu, S, ok = fr
    if ok.sum() < MIN_FRAMES:
        return None
    return target[ok].astype(int), mu[ok], S[ok] + 1e-9 * np.eye(2)


def mixture_frames(pred, T, lead, ref):
    """The full mixture of a MixturePredictive over the frames model_frames keeps (same mask
    and order), or None for a Gaussian forecast."""
    if not isinstance(pred, MixturePredictive):
        return None
    fr = _lead_frames(pred, T, lead, ref)
    if fr is None:
        return None
    ok = fr[3]
    if ok.sum() < MIN_FRAMES:
        return None
    _, w, mu, cov = pred.mixture_at_lead(lead)
    return ri.MixtureForecast(w[ok], mu[ok], cov[ok])


def whiten(y, mu, S):
    return np.linalg.solve(np.linalg.cholesky(S), (y - mu)[..., None])[..., 0]


def calm_mask(target, events, exclusion=(MASK_PRE, MASK_POST)):
    pre, post = exclusion
    mask = np.ones(len(target), bool)
    for ev in events:
        mask &= ~((target >= ev.t0 - pre) & (target <= ev.t0 + post))
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


# ---------------------------------------------------------------------------
# Overlap tables, named by what they were built from
# ---------------------------------------------------------------------------


def _table_axes(ref, lead):
    return ref.table_axes(lead, TABLE_CORE, TABLE_STEP, TABLE_GROWTH, TABLE_REACH)


def table_key(ref, lead, c, sig2):
    """Fingerprint of everything an overlap table depends on: the climatology on the table's
    speed grid, the grid, the information level (c), the noise floor and the quadrature."""
    v, core, h, growth, extent = _table_axes(ref, lead)
    w, m, C = ref.body_mixture(lead)(v)
    hsh = hashlib.sha1()
    for a in (v, w, m, C, [core, h, growth, extent, c, sig2, OVERLAP["overlap_gh"]]):
        hsh.update(np.ascontiguousarray(a, float).tobytes())
    return hsh.hexdigest()[:16]


def table_path(table_dir, lead, info, key):
    return os.path.join(table_dir, f"overlap_lead{lead}_info{info:g}_{key}")


def _build_table(job):
    """Build one table unless a table with the same fingerprint is already there. Files are
    written under temporary names and renamed, so a table is either complete or absent."""
    ref, lead, info, c, path, sig2 = job
    if os.path.exists(path + "_meta.npz") and os.path.exists(path + ".npy"):
        return lead, info, 0.0, "reused"
    t = time.time()
    v, core, h, growth, extent = _table_axes(ref, lead)
    tab = ri.OverlapTable.build(
        ref.body_mixture(lead), c, v, core, h, growth, extent, sig2=sig2, **OVERLAP
    )
    tmp = f"{path}.tmp{uuid.uuid4().hex[:8]}"
    tab.save(tmp)
    os.replace(tmp + ".npy", path + ".npy")
    os.replace(tmp + "_meta.npz", path + "_meta.npz")
    return lead, info, time.time() - t, tab.G.shape


# ---------------------------------------------------------------------------
# Forecasts: a mapping name -> Forecaster (``predict(traj)``) or {track id: Predictive}, or a
# zero-argument callable returning such a mapping (built once per worker process and call, for
# models that are expensive or awkward to pickle)
# ---------------------------------------------------------------------------

_WORKER: Dict[object, object] = {}  # per-process cache: models and overlap tables


def _cached(key, make):
    if key not in _WORKER:
        _WORKER[key] = make()
    return _WORKER[key]


def _resolve(forecasts: ForecastSource, token=None):
    """The model mapping. A callable is built once per process for each fit/evaluate call
    (token), never reused across calls."""
    if callable(forecasts) and not isinstance(forecasts, Mapping):
        return _cached(("forecasts", token), forecasts)
    return forecasts


def _predict_all(models, traj, errors):
    out = {}
    for name, f in models.items():
        if isinstance(f, Mapping):
            out[name] = f.get(traj.id)
            continue
        try:
            out[name] = f.predict(traj)
        except Exception as e:  # noqa: BLE001  (reported per model by the caller)
            out[name] = None
            n, first = errors.get(name, (0, None))
            errors[name] = (n + 1, first or f"{type(e).__name__}: {e}")
    return out


def _warn_errors(errors, n_tracks):
    for name, (n, first) in sorted(errors.items()):
        warnings.warn(
            f"{name}.predict raised on {n} of {n_tracks} tracks, which are not scored for it "
            f"(first error: {first})", RuntimeWarning, stacklevel=3,
        )


def _merge_errors(total, part):
    for name, (n, first) in part.items():
        m, f = total.get(name, (0, None))
        total[name] = (m + n, f or first)


def _subset(forecasts, ids):
    """Precomputed forecasts restricted to the tracks of one chunk."""
    if not isinstance(forecasts, Mapping):
        return forecasts
    return {name: ({t: f[t] for t in ids if t in f} if isinstance(f, Mapping) else f)
            for name, f in forecasts.items()}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _calib_chunk(tracks, ref, forecasts, detector, model_leads, leads, token=None,
                 exclusion=(MASK_PRE, MASK_POST)):
    models = _resolve(forecasts, token) if forecasts is not None else {}
    names = list(models)
    nodes, gw = ri.gh_nodes_2d(GH_NODES)
    errors: Dict[str, Tuple[int, str]] = {}
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
        preds = _predict_all(models, traj, errors)
        T = len(traj.positions)
        for lead in leads:
            fr = ref.gaussian_frames(traj, lead, MIN_FRAMES)
            if fr is None:
                continue
            target, y, mu_r, S_r = fr
            calm = calm_mask(target, events, exclusion)
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
    out["errors"] = errors
    return out


def _split_chunks(tracks, n):
    ids = list(tracks.ids) if hasattr(tracks, "ids") else list(tracks)
    return [{t: tracks[t] for t in ids[i::n]} for i in range(n) if ids[i::n]]


def _small_chunks(tracks, size):
    """Chunks of about `size` tracks (strided, so long and short tracks mix)."""
    n_ids = len(tracks.ids) if hasattr(tracks, "ids") else len(tracks)
    return _split_chunks(tracks, max(1, -(-n_ids // size)))


def _pool_error(e):
    """BrokenProcessPool usually means a script started worker processes without a
    `if __name__ == "__main__":` guard (macOS and Windows start workers by re-importing it)."""
    from concurrent.futures.process import BrokenProcessPool

    if isinstance(e, BrokenProcessPool):
        raise RuntimeError(
            "a worker process died. If this runs from a script, put the calls behind "
            "`if __name__ == \"__main__\":` (worker processes re-import the script), or use "
            "n_jobs=1."
        ) from e
    raise e


@dataclass
class Level:
    """One assumed information level at one lead: c = e^-I and the fitted correlations."""

    info: float
    c: float
    rho_a: np.ndarray
    rho_b: np.ndarray


def _check_levels(info, lead_info):
    for I in list(info) + [x for v in (lead_info or {}).values() for x in v]:
        if not float(I) > 0:
            raise ValueError(f"information levels must be positive (nats); got {I}")


def _check_distortions(distortions):
    """Target distances besides d = infinity, which is always included (inf or None in the
    list are ignored)."""
    out = []
    for d in distortions:
        if d is None or (isinstance(d, (int, float)) and np.isinf(d)):
            continue
        if not float(d) > 0:
            raise ValueError(f"target distances must be positive metres; got {d} (d = infinity "
                             "is always included)")
        out.append(float(d))
    return tuple(out)


def _check_isotropic(ref, leads):
    for lead in leads:
        S = np.asarray(ref.stop_cov[lead], float)
        if not np.allclose(S, 0.5 * np.trace(S) * np.eye(2), rtol=1e-9, atol=1e-15):
            raise ValueError(
                f"the climatology's stopped regime at lead {lead} is not isotropic; REACT's "
                "overlap tables assume it (fit it with fit_climatology(isotropic_stop=True))"
            )


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


@dataclass
class ReactScorer:
    """Everything REACT shares across the models evaluated on one dataset.

    Build it with ReactScorer.fit on calibration tracks, then call evaluate(). Save it with
    save() and reload it with ReactScorer.load() to score more models against the same fit.
    """

    climatology: object
    levels: Dict[int, List[Level]]
    sigma: float
    events: object
    table_dir: str
    leads: Tuple[int, ...] = LEADS
    distortions: Tuple[float, ...] = (0.1, 0.3)
    p: float = 0.95
    eps: float = 0.0
    entropy_margin_info: Optional[Tuple[float, ...]] = None
    diagnostics: Dict = field(default_factory=dict)
    noise_floor: Dict = field(default_factory=dict)
    window: int = WINDOW_CAP
    min_window: int = MIN_WINDOW
    onset_margin: int = ONSET_MARGIN
    relapse_window: int = RELAPSE_WINDOW
    relapse_run: int = RELAPSE_RUN
    calm_exclusion: Tuple[int, int] = (MASK_PRE, MASK_POST)
    pre_window: int = PRE_WINDOW
    tables: Dict = field(default_factory=dict)  # (lead, info) -> table path
    table_seconds: Dict = field(default_factory=dict)

    @property
    def log_k(self) -> float:
        return float(np.log(self.p / (1.0 - self.p)))

    def __repr__(self) -> str:
        lv = {lead: [x.info for x in v] for lead, v in self.levels.items()}
        return (f"ReactScorer(leads={list(self.leads)}, info={lv}, distortions="
                f"{['inf', *self.distortions]}, sigma={100 * self.sigma:.2f} cm, "
                f"window={self.window})")

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
        forecasts: Optional[ForecastSource] = None,
        model_leads: Optional[Mapping[str, Sequence[int]]] = None,
        window: int = WINDOW_CAP,
        min_window: int = MIN_WINDOW,
        onset_margin: int = ONSET_MARGIN,
        relapse_window: int = RELAPSE_WINDOW,
        relapse_run: int = RELAPSE_RUN,
        calm_exclusion: Tuple[int, int] = (MASK_PRE, MASK_POST),
        pre_window: int = PRE_WINDOW,
        table_dir: Optional[str] = None,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> ReactScorer:
        """Fit REACT's shared quantities on calibration tracks (a TrajectorySet).

        info: assumed information levels I_traj (nats, > 0); the paper reports 3, 4.2 and 6.
            lead_info: per-lead overrides (the paper ran lead 1 at 4.2 only).
        leads: forecast leads h, in observations.
        sigma: the measurement-noise floor: "estimate" (from stationary calibration frames),
            0 for none, or a value in metres.
        distortions: target distances d (m, > 0) besides d = infinity, which is always
            included.
        events: the event source; default BayesianChangepoints.tuned_sg() (paper section 3.4).
        p: decision probability; a verdict needs posterior probability >= p.
        eps: extra margin (nats) a forecast must beat the climatology by to count as recovered,
            on top of the distortion margin. The paper uses 0.
        climatology: a fitted climatology with an isotropic stopped regime; default
            fit_climatology(calib, leads).
        forecasts: optional models, scored on calm calibration stretches; their mean realized
            score (diagnostics[lead]["calm_mean_score"]) is the demonstrated information, a
            lower bound on I_traj.
        entropy_margin_info: levels at which outcomes are also recorded under the
            mixture-entropy distortion margin (the _H fields); default every level.

        Windows, all in observations (the defaults are the paper's at 10 Hz: 15 s, 2.5 s,
        0.5 s, 1 s, 0.3 s, 2 s / 6 s and 3 s):
            window: an event window ends this many frames after the event, or at the next event.
            min_window: events with fewer frames than this before the next event are skipped.
            onset_margin: an onset counts within lead + onset_margin frames of the event.
            relapse_window, relapse_run: a relapse is relapse_run consecutive "behind" frames
                starting within relapse_window frames of recovery.
            calm_exclusion: (before, after) frames around each event left out of the calm
                stretches the truth model's correlations are fitted on.
            pre_window: frames before an event in the pre-event score.
        The event detector's and noise floor's own windows are set on those objects.

        table_dir: where the overlap tables are written (and reused when one with the same
        fingerprint exists); default a temporary directory.
        """
        from .events import BayesianChangepoints

        leads = tuple(int(x) for x in leads)
        _check_levels(info, lead_info)
        distortions = _check_distortions(distortions)
        detector = events if events is not None else BayesianChangepoints.tuned_sg()
        model_leads = {k: tuple(v) for k, v in (model_leads or {}).items()}
        ref = climatology if climatology is not None else fit_climatology(
            calib, leads=leads, dt=calib.dt, seed=seed
        )
        _check_isotropic(ref, leads)
        corr_len = int(window) + 1

        if isinstance(sigma, str):
            if sigma.lower() not in ("estimate", "a"):
                raise ValueError(f"sigma must be 'estimate', 0 or metres, not {sigma!r}")
            sig, n_res, n_tr = ri.estimate_noise_floor(calib)
            noise_fit = {"floor": "A", "sigma_cm": 100 * sig, "residuals_per_axis": n_res,
                         "tracks": n_tr}
            if n_res == 0:
                warnings.warn(
                    "no stationary calibration frames to estimate the noise floor from; using "
                    "no floor (sigma = 0). Pass sigma= (metres) if you know your sensor's noise.",
                    RuntimeWarning, stacklevel=2,
                )
                sig, noise_fit = 0.0, {"floor": "none", "residuals_per_axis": 0, "tracks": 0}
        elif float(sigma) == 0.0:
            sig, noise_fit = 0.0, {"floor": "none"}
        else:
            sig, noise_fit = float(sigma), {"floor": "given"}
        noise_fit["sigma_cm"] = 100 * sig
        sig2 = sig**2

        token = uuid.uuid4().hex
        chunks = _split_chunks(calib, max(1, n_jobs))
        if forecasts is not None and n_jobs <= 1:
            forecasts = _resolve(forecasts, token)
        args = (ref, forecasts, detector, model_leads, leads, token, tuple(calm_exclusion))
        if n_jobs > 1:
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    parts = list(pool.map(_calib_chunk, chunks,
                                          *[[a] * len(chunks) for a in args]))
            except Exception as e:  # noqa: BLE001
                _pool_error(e)
        else:
            parts = [_calib_chunk(ch, *args) for ch in chunks]
        errors: Dict[str, Tuple[int, str]] = {}
        for pt in parts:
            _merge_errors(errors, pt.pop("errors"))
        _warn_errors(errors, len(calib.ids))
        names = list(parts[0][leads[0]]["s_sum"]) if parts else []

        diagnostics, levels = {}, {}
        for lead in leads:
            runs = [r for pt in parts for r in pt[lead]["runs"]]
            rho_w, ms = ri.pooled_autocorr(runs, N_LAGS)
            s_mean = {
                m: sum(pt[lead]["s_sum"][m] for pt in parts)
                / max(sum(pt[lead]["s_n"][m] for pt in parts), 1)
                for m in names
            }
            rho_b = ri.overlap_corr(lead, corr_len)
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
                    Level(float(I), c, ri.knowable_corr(corr_len, scale, tau, omega), rho_b)
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
            diagnostics[lead] = {
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
        paths = {
            (lead, lv.info): table_path(table_dir, lead, lv.info,
                                        table_key(ref, lead, lv.c, sig2))
            for lead in leads for lv in levels[lead]
        }
        jobs = [(ref, lead, lv.info, lv.c, paths[(lead, lv.info)], sig2)
                for lead in leads for lv in levels[lead]]
        if n_jobs > 1:
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    built = list(pool.map(_build_table, jobs))
            except Exception as e:  # noqa: BLE001
                _pool_error(e)
        else:
            built = [_build_table(j) for j in jobs]
        timing = {f"L{b[0]}/{b[1]:g}": b[2] for b in built}

        ent = None if entropy_margin_info is None else tuple(
            round(float(x), 4) for x in entropy_margin_info
        )
        return cls(
            climatology=ref, levels=levels, sigma=sig, events=detector, table_dir=table_dir,
            leads=leads, distortions=distortions, p=float(p), eps=float(eps),
            entropy_margin_info=ent, diagnostics=diagnostics, noise_floor=noise_fit,
            window=int(window), min_window=int(min_window), onset_margin=int(onset_margin),
            relapse_window=int(relapse_window), relapse_run=int(relapse_run),
            calm_exclusion=tuple(int(x) for x in calm_exclusion), pre_window=int(pre_window),
            tables=paths, table_seconds=timing,
        )

    # -- saving --------------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted scorer and its overlap tables to the directory `path` (created),
        to score more models later against exactly the same fit. Uses pickle: load only
        scorers you saved yourself."""
        os.makedirs(path, exist_ok=True)
        tables = {}
        for key, src in self.tables.items():
            dst = os.path.join(path, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                for ext in (".npy", "_meta.npz"):
                    shutil.copyfile(src + ext, dst + ext)
            tables[key] = os.path.basename(src)
        state = {k: v for k, v in self.__dict__.items() if k not in ("table_dir", "tables")}
        with open(os.path.join(path, "scorer.pkl"), "wb") as f:
            pickle.dump({"version": 1, "state": state, "tables": tables}, f)

    @classmethod
    def load(cls, path: str) -> ReactScorer:
        """A scorer saved with save(); its tables are read from `path`."""
        with open(os.path.join(path, "scorer.pkl"), "rb") as f:
            blob = pickle.load(f)
        tables = {k: os.path.join(path, v) for k, v in blob["tables"].items()}
        return cls(**blob["state"], table_dir=path, tables=tables)

    # -- scoring -------------------------------------------------------------------------------

    def _tables(self):
        return {
            key: _cached(("table", p_), lambda p_=p_: ri.OverlapTable.load(p_))
            for key, p_ in self.tables.items()
        }

    def score_track(self, traj, predictives: Mapping[str, object], events=None,
                    model_leads: Optional[Mapping[str, Sequence[int]]] = None, clock=None):
        """REACT records for every event on one track: one row per (lead, level, distortion,
        model, event), Oracle B included. predictives: name -> Predictive (or None).

        events: the track's events (default: this scorer's event source)."""
        ref = self.climatology
        model_leads = model_leads or {}
        events = sorted(self.events.detect(traj) if events is None else events,
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
                sel = (target > ev.t0) & (target < nxt) & (target <= ev.t0 + self.window)
                if sel.sum() < self.min_window:
                    continue
                fw, yw, mrw, Srw, ww = target[sel], y[sel], mu_r[sel], S_r[sel], w[sel]
                pre = (target >= ev.t0 - self.pre_window) & (target < ev.t0)
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
                        if len(ia) >= self.min_window:
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
            evaluate, F, ev.t0, log_k, len(mg), lead + self.onset_margin, oracle=oracle,
            relapse_window=self.relapse_window, relapse_run=self.relapse_run,
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

    def evaluate(self, forecasts: ForecastSource, data, *, model_leads=None, n_jobs: int = 1,
                 event_set: str = "shared", chunk_tracks: int = 5,
                 progress: Optional[Callable[[int, int], None]] = None) -> ReactResult:
        """Score every model on every event of the test tracks (a TrajectorySet).

        forecasts: name -> a forecaster (anything with predict(traj) returning a Predictive
            or MixturePredictive, or None) or {track id: Predictive}; or a zero-argument
            callable returning that mapping, built once per worker process (for models that
            are expensive or awkward to pickle).
        model_leads: name -> leads at which that model is scored (default: all).
        event_set: "shared" (default) keeps, at each lead, only the events every model has a
            forecast window for, so models are compared on the same events, and warns when
            that drops any; "all" scores each model on every event it covers.
        n_jobs: worker processes. With n_jobs > 1 a script must run under
            `if __name__ == "__main__":`.
        progress: called as progress(chunks done, chunks) while scoring.

        Returns a ReactResult. A model whose predict raises is skipped on that track, with a
        warning."""
        if event_set not in ("shared", "all"):
            raise ValueError(f"event_set must be 'shared' or 'all', not {event_set!r}")
        model_leads = {k: tuple(v) for k, v in (model_leads or {}).items()}
        if isinstance(forecasts, Mapping):
            _check_ids(forecasts, data)
        token = uuid.uuid4().hex
        if n_jobs <= 1:
            forecasts = _resolve(forecasts, token)
        names = list(forecasts) if isinstance(forecasts, Mapping) else None
        chunks = _small_chunks(data, chunk_tracks)
        results, clock, errors = {}, {}, {}
        t0 = time.time()

        def collect(i, out):
            recs, clk, err = out
            results[i] = recs
            for k_, v_ in clk.items():
                clock[k_] = clock.get(k_, 0) + v_
            _merge_errors(errors, err)

        if n_jobs > 1:
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    futs = {pool.submit(_score_chunk, self, ch, _subset(forecasts, ch),
                                        model_leads, token): i
                            for i, ch in enumerate(chunks)}
                    for n_done, fut in enumerate(as_completed(futs), 1):
                        collect(futs[fut], fut.result())
                        if progress is not None:
                            progress(n_done, len(chunks))
            except Exception as e:  # noqa: BLE001
                _pool_error(e)
        else:
            for i, ch in enumerate(chunks):
                collect(i, _score_chunk(self, ch, forecasts, model_leads, token))
                if progress is not None:
                    progress(i + 1, len(chunks))
        clock["wall"] = time.time() - t0
        _warn_errors(errors, len(data.ids))
        records = [r for i in sorted(results) for r in results[i]]
        if names is None:
            names = list(dict.fromkeys(r["model"] for r in records if r["model"] != "oracle_B"))
        records, coverage = _event_coverage(records, names, model_leads, self.leads,
                                            shared=event_set == "shared")
        _warn_degenerate(records)
        return ReactResult(records=records, scorer=self, timing=clock, coverage=coverage,
                           meta={"event_set": event_set})


def _event_coverage(records, names, model_leads, leads, shared=True):
    """Events covered per lead and model; with shared=True keep, at each lead, only the events
    every model scored there covers (warning when that drops any)."""
    by = {}
    for r in records:
        by.setdefault(r["lead"], {}).setdefault(r["model"], set()).add((r["track"], r["t0"]))
    coverage, keep = {}, {}
    for lead in leads:
        got = by.get(lead, {})
        expected = [m for m in names if model_leads.get(m) is None or lead in model_leads[m]]
        missing = [m for m in expected if m not in got]
        for m in missing:
            warnings.warn(f"{m} has no scorable forecast window at lead {lead}", RuntimeWarning,
                          stacklevel=3)
        present = [m for m in expected if m in got]
        events = got.get("oracle_B", set())
        common = set(events)
        for m in present:
            common &= got[m]
        coverage[lead] = {"events": len(events), "shared": len(common),
                          **{m: len(got[m]) for m in present}}
        keep[lead] = common
        dropped = len(events) - len(common)
        if shared and dropped and present:
            short = {m: len(events) - len(got[m]) for m in present if len(got[m]) < len(events)}
            warnings.warn(
                f"lead {lead}: {dropped} of {len(events)} events lack a forecast window from "
                f"every model and are left out so all models are compared on the same "
                f"{len(common)} events (events missing per model: {short}); pass "
                f"event_set='all' to keep them", RuntimeWarning, stacklevel=3,
            )
    if shared:
        records = [r for r in records if (r["track"], r["t0"]) in keep.get(r["lead"], ())]
    return records, coverage


def _warn_degenerate(records):
    tot = {}
    for r in records:
        if r["model"] != "oracle_B" and r["distortion"] == 0.0:
            k = (r["model"], r["lead"])
            s, n = tot.get(k, (0.0, 0))
            tot[k] = (s + r["s_sum"], n + r["n"])
    for (m, lead), (s, n) in sorted(tot.items()):
        if n and s / n < DEGENERATE_SCORE:
            warnings.warn(
                f"{m} at lead {lead} scores {s / n:.1f} nats per frame below the climatology on "
                "average; its forecasts look degenerate (collapsed or misaligned covariances?)",
                RuntimeWarning, stacklevel=3,
            )


def _check_ids(forecasts, data):
    """Precomputed forecasts keyed by ids that match no test track score nothing; say so."""
    ids = set(data.ids)
    for name, f in forecasts.items():
        if isinstance(f, Mapping) and f and not ids & set(f):
            raise ValueError(
                f"forecasts for {name!r} are keyed by ids that match no test track "
                f"(e.g. {next(iter(f))!r}; tracks look like {next(iter(ids))!r})"
            )


def _score_chunk(scorer, tracks, forecasts, model_leads, token=None):
    models = _resolve(forecasts, token)
    clock = {"predict": 0.0}
    errors: Dict[str, Tuple[int, str]] = {}
    t_chunk = time.time()
    records = []
    for traj in tracks.values():
        events = sorted(scorer.events.detect(traj), key=lambda e: e.t0)
        if not events:
            continue
        t_ = time.time()
        preds = _predict_all(models, traj, errors)
        clock["predict"] += time.time() - t_
        records.extend(scorer.score_track(traj, preds, events, model_leads, clock))
    clock["chunk total"] = time.time() - t_chunk
    return records, clock, errors
