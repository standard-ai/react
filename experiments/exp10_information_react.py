"""Exp 10: REACT in the information formulation (../docs/react-spec.md), on Eindhoven.

Uses the same track draw, split and 'low'-preset events as exp09 seed 0.

Reference: the kinematic climatology conditioned on the velocity over the last
--velocity-window frames (default 1: the last observed step), with its stopped/moving share
fitted by EM rather than a hand-set ramp, on calibration frames whose still runs are cut at
WINDOW_CAP frames (stillness REACT can never score does not train the reference). The
5-frame version is a calibrated forecaster in its own right, so it is kept only as an option.

Models: the Kalman constant-velocity filter and the IMM have their noise parameters fitted by
maximum likelihood on the calibration tracks' one-step innovations (ConstantVelocity.fit,
IMM.fit). CTRV keeps its hand-set parameters. GRU checkpoints can be supplied with
--gru-dir (a directory holding gru_position_best.pt and gru_orientation_best.pt); otherwise
the shipped retail-trained checkpoints are used.
Trajectron++ (--tpp-dir, a clone of the repository; the Eindhoven run-1 checkpoint is in
../artifacts/trajectron/eindhoven_run1) enters twice from one network pass: "Trajectron"
(moment-matched Gaussian) and "Trajectron_mix" (all 25 components, no pruning or merging, in the
realized score and the tests; reactmetric.information.MixtureForecast / RegimeTestsMixture).
The base case, recorded as distortion 0.0 (label "clim"), has no target: the climatology's own
spread is the bar; distortions > 0 are the REACT@d targets.

  1. Calibration tracks. Each model's mean realized score on calm stretches (reported only).
     Climatology's whitened residual autocorrelation rho_w(l), and for each assumed
     information level I (--levels, nats) the knowable correlation rho_a fitted net of the
     structural overlap.
  2. Test tracks. For every event window: the truth posterior (shared by all models); per model
     the per-frame onset and recovery Bayes factors -> onset, REACT, status, relapse; the
     perfect-forecaster floor; mean realized score vs mean posterior advantage (consistency).

Needs the export first (../scripts/export_eindhoven_csv.py --tracks 20000). Run from here:

    ../.venv/bin/python exp10_information_react.py --tracks 1200 --seed 0 --jobs 8 \
        --gru-dir ../artifacts/gru_checkpoints/eindhoven_10hz
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from functools import partial  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from trajectron_forecaster import add_cli_args as add_tpp_args  # noqa: E402

from reactmetric import baselines  # noqa: E402
from reactmetric.core.predictive import MixturePredictive, Predictive  # noqa: E402
from reactmetric.core.trajectory import TrajectorySet  # noqa: E402
from reactmetric.events import bayes as bcp  # noqa: E402
from reactmetric.reference import Climatology, fit_climatology  # noqa: E402
from reactmetric.scorer import (  # noqa: E402, F401  (re-exported for exp11, realized_behind)
    CLIM,
    CORR_LEN,
    GH_NODES,
    MIN_FRAMES,
    MIN_WINDOW,
    ONSET_MARGIN,
    OVERLAP,
    PRE_WINDOW,
    WINDOW_CAP,
    ReactScorer,
    model_frames,
    summarize,
    table_key,
    table_path,
    whiten,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CSV = os.path.join(ROOT, ".data", "eindhoven_20000tracks.csv")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "exp10")
LEADS = (1, 5, 10, 15)
# IMM fit bounds: the original hand-set values +- a physical margin, so the modes stay a stop
# mode, a straight mode and two turns (unbounded, the turn modes become catch-all noise modes).
IMM_BOUNDS = {
    "measurement_noise": (0.001, 0.038),
    "turn_rate": (0.4, 0.8),
    "sigma_a_cv": (0.2, 0.6),
    "sigma_a_ct": (0.5, 0.9),
    "sigma_a_stop": (0.0, 0.25),
    "stop_velocity_decay": (0.05, 0.25),
    "p_stay": (0.94, 0.9999),
    "measurement_noise_stop": (0.001, 0.038),
    "p_stay_stop": (0.94, 0.9999),
}
DISTORTIONS = (0.1, 0.2, 0.3)


# ---------------------------------------------------------------------------
# Per-track frames
# ---------------------------------------------------------------------------


class Trajectron:
    """One network pass per track, shared by the moment-matched ("Trajectron") and full-mixture
    ("Trajectron_mix") entries of the specs."""

    def __init__(self, **kw):
        from trajectron_forecaster import TrajectronForecaster

        self.net, self._last = TrajectronForecaster(threads=1, **kw), (None, None)

    def predict(self, traj):
        if self._last[0] is not traj:
            mix = self.net.predict_mixture(traj)
            pred = None if mix is None else MixturePredictive(
                mix["issue_frames"], mix["weights"], mix["means"], mix["covs"], self.net.horizon
            )
            self._last = (traj, pred)
        return self._last[1]


class _MomentMatched:
    """The moment-matched Gaussian of a mixture forecaster (the "Trajectron" entry)."""

    def __init__(self, net):
        self.net = net

    def predict(self, traj):
        pred = self.net.predict(traj)
        if pred is None:
            return None
        return Predictive(pred.issue_frames, pred.means, pred.covs, pred.horizon)


def build_models(specs):
    out, tpp = {}, {}
    for m, (name, kw, _) in specs.items():
        if name == "Trajectron":  # "Trajectron" and "Trajectron_mix" share one network
            key = json.dumps(kw, sort_keys=True)
            tpp.setdefault(key, Trajectron(**kw))
            out[m] = tpp[key] if m.endswith("_mix") else _MomentMatched(tpp[key])
        else:
            out[m] = baselines.build(name, **kw)
    return out


def make_detector(spec):
    """Event source: ("bayes-sg", hazard) tuned Bayes with Savitzky-Golay smoothing (the
    paper's), or ("bayes", hazard) the earlier Bayes settings."""
    kind, arg = spec
    if kind == "bayes-sg":
        return bcp.BayesianChangepoints.tuned_sg(hazard=arg)
    return bcp.BayesianChangepoints(hazard=arg)


def cap_still_runs(tset, cap=WINDOW_CAP, stop_speed=0.2, window=5):
    """Calibration tracks with every still run cut at ``cap`` frames.

    REACT never scores more than WINDOW_CAP frames after an event, so stillness beyond
    that is evidence the metric cannot use; letting it train the reference would make the
    reference tighter than anything the evaluated frames substantiate. A frame is still
    when its ``window``-frame speed is under ``stop_speed``; each still run keeps its first
    ``cap`` frames and the track is split around the removed part.
    """
    out, dropped, total = {}, 0, 0
    for tid in tset.ids:
        obs = np.asarray(tset[tid].positions, float)
        total += len(obs)
        speed = np.full(len(obs), np.inf)
        speed[window:] = np.hypot(*(obs[window:] - obs[:-window]).T) / (window * tset.dt)
        edges = np.flatnonzero(np.diff(np.r_[0, (speed < stop_speed).astype(int), 0]))
        keep = np.ones(len(obs), bool)
        for a, b in zip(edges[::2], edges[1::2]):
            keep[a + cap : b] = False
        dropped += int((~keep).sum())
        cut = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
        for k, (a, b) in enumerate(zip(cut[::2], cut[1::2])):
            if b - a > 20:
                out[f"{tid}_{k}"] = obs[a:b]
    return out, dropped / max(total, 1)


def fit_reference(calib, velocity_window, seed, cap_still=True, stop_isotropic=True):
    """Kinematic climatology for the headline reference.

    velocity_window 1: the stopped/moving share is fitted by EM (stop_model="fitted") on
    calibration frames with still runs capped at WINDOW_CAP. velocity_window 5: the old
    hand-set ramp on all calibration frames, kept for comparison.

    stop_isotropic: replace the stopped regime's fitted spread by its isotropic average. The
    fitted spread is fixed in world axes and 5-8% wider along one of them; the isotropic one
    scores slightly better on held-out frames (+0.0001 to +0.0006 nats per frame on Eindhoven)
    and makes the climatology depend on the speed alone in the mover's body frame, which the
    overlap tables of the exact tests rely on.
    """
    if velocity_window == 5:
        return Climatology.fit(calib, leads=LEADS, mode="kinematic"), None
    if cap_still:
        capped, share = cap_still_runs(calib)
    else:
        capped, share = {t: np.asarray(calib[t].positions, float) for t in calib.ids}, 0.0
    ref = fit_climatology(
        capped, leads=LEADS, dt=calib.dt, seed=seed, velocity_window=velocity_window,
        isotropic_stop=stop_isotropic,
    )
    info = {
        "stop_model": "fitted",
        "stop_isotropic": stop_isotropic,
        "still_run_cap_frames": WINDOW_CAP if cap_still else None,
        "calibration_frames_dropped": share,
        "stop_sd_cm": {
            lead: float(100 * np.sqrt(np.trace(ref.stop_cov[lead]) / 2)) for lead in LEADS
        },
        "p_moving_at_zero_speed": {lead: float(ref.move_weight_grid[lead][0]) for lead in LEADS},
    }
    return ref, info


def fit_models(calib, gru_dir, gru_at5_dir=None, imm_params=None, tpp=None, ctrv_params=None,
               tpp_gaussian=True):
    """Model specs (baseline name, kwargs, leads or None) with noise fitted on calibration.

    imm_params: optional JSON from an earlier IMM.fit with IMM_BOUNDS (the fit takes ~15 min).
    ctrv_params: optional JSON with sigma_a, sigma_yaw, measurement_noise (default: hand-set CTRV).
    tpp_gaussian: False drops the moment-matched "Trajectron" entry, keeping "Trajectron_mix".
    """
    cv = baselines.ConstantVelocity.fit(calib)
    if imm_params:
        imm_fit = json.load(open(imm_params))
        imm_kw = {k: float(imm_fit[k]) for k in baselines.IMM._FIT_PARAMS if k in imm_fit}
    else:
        imm = baselines.IMM.fit(calib, bounds=IMM_BOUNDS)
        imm_kw = {k: float(getattr(imm, k)) for k in baselines.IMM._FIT_PARAMS}
        imm_fit = {**imm_kw, **imm.fit_result_}
    specs = {
        "ConstantVelocity": (
            "ConstantVelocity",
            {"measurement_noise": cv.measurement_noise, "accel_noise": cv.accel_noise},
            None,
        ),
        "CTRV": (
            "CTRV",
            {k: float(json.load(open(ctrv_params))[k]) for k in ("sigma_a", "sigma_yaw", "measurement_noise")}
            if ctrv_params
            else {},
            None,
        ),
        "IMM": ("IMM", imm_kw, None),
        "GRUPosition": ("GRUPosition", {}, None),
        "GRUOrientation": ("GRUOrientation", {}, None),
    }
    if gru_dir:
        specs["GRUPosition"][1]["checkpoint"] = os.path.join(gru_dir, "gru_position_best.pt")
        specs["GRUOrientation"][1]["checkpoint"] = os.path.join(gru_dir, "gru_orientation_best.pt")
    if gru_at5_dir:
        for m, f in (
            ("GRUPosition", "gru_position_best.pt"),
            ("GRUOrientation", "gru_orientation_best.pt"),
        ):
            specs[f"{m}@5"] = (m, {"checkpoint": os.path.join(gru_at5_dir, f)}, (5,))
    if tpp:  # tpp = dict(tpp_dir=clone, model_dir=..., checkpoint=epoch or None)
        if tpp_gaussian:
            specs["Trajectron"] = ("Trajectron", dict(tpp), None)
        specs["Trajectron_mix"] = ("Trajectron", dict(tpp), None)
    fitted = {
        "ConstantVelocity": {
            **specs["ConstantVelocity"][1],
            **cv.fit_result_,
            "effective_memory_frames": cv.effective_memory(calib.dt),
        },
        "IMM": imm_fit,
    }
    return specs, fitted


def fmt(x, w=6, d=0):
    return f"{'—':>{w}}" if x is None or not np.isfinite(x) else f"{x:>{w}.{d}f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument(
        "--dataset",
        choices=["eindhoven", "day-in-the-life"],
        default="eindhoven",
        help="day-in-the-life: Standard AI public release (waist positions), see day_in_the_life.py",
    )
    ap.add_argument("--tracks", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--p", type=float, default=0.95, help="posterior-probability threshold")
    ap.add_argument(
        "--distortions",
        type=float,
        nargs="*",
        default=list(DISTORTIONS),
        help="target distortions d (m) for REACT@d; the base case with no target (the "
        "climatology's own spread, label 'clim', distortion 0.0) is always included",
    )
    ap.add_argument(
        "--events",
        choices=("bayes-sg", "bayes"),
        default="bayes-sg",
        help="event source: tuned Bayesian changepoints on Savitzky-Golay smoothed positions "
        "(default; speed, turn, vx, vy channels, fixed tuned hazard), the earlier Bayesian "
        "changepoints (speed and turn, empirical-Bayes hazard)",
    )
    ap.add_argument(
        "--hazard",
        type=float,
        default=None,
        help="changepoint hazard (default: 9e-4 for bayes-sg, empirical Bayes for bayes)",
    )
    ap.add_argument(
        "--noise-floor",
        default="A",
        help="measurement-noise floor of the truth spread, Omega = c S + sigma^2 I: 'A' (default) "
        "estimates sigma from the calibration tracks (residual about a 9-frame local quadratic "
        "fit on stationary frames), 'none' sets sigma = 0, a number sets sigma in cm",
    )
    ap.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 3.0],
        help="assumed information levels, in nats",
    )
    ap.add_argument(
        "--lead1-levels",
        type=float,
        nargs="+",
        default=None,
        help="information levels (nats) at which lead 1 is evaluated (default: all --levels); "
        "the other leads always use all of --levels",
    )
    ap.add_argument(
        "--entropy-margin-levels",
        type=float,
        nargs="+",
        default=None,
        help="levels at which the mixture-entropy margin outcomes (_H) are computed "
        "(default: all --levels)",
    )
    ap.add_argument(
        "--no-tpp-gaussian",
        action="store_true",
        help="with --tpp-dir, drop the moment-matched 'Trajectron' entry and keep 'Trajectron_mix'",
    )
    ap.add_argument(
        "--chunk-tracks",
        type=int,
        default=5,
        help="test tracks per task; tasks are dealt to the workers as they free up",
    )
    ap.add_argument(
        "--velocity-window",
        type=int,
        default=1,
        help="frames the reference's velocity is taken over (5 = the old default)",
    )
    ap.add_argument(
        "--gru-at5-dir",
        default=None,
        help="directory with lead-5-only GRU checkpoints (scored at lead 5 only)",
    )
    ap.add_argument("--imm-params", default=None, help="JSON of an earlier bounded IMM fit")
    ap.add_argument(
        "--ctrv-params",
        default=None,
        help="JSON with fitted CTRV sigma_a, sigma_yaw, measurement_noise (default: hand-set)",
    )
    add_tpp_args(ap)
    ap.add_argument(
        "--no-still-cap",
        action="store_true",
        help="fit the reference on all calibration frames (no cut of still runs at WINDOW_CAP)",
    )
    ap.add_argument(
        "--gru-dir",
        default=None,
        help="directory with gru_position_best.pt / gru_orientation_best.pt",
    )
    ap.add_argument(
        "--stop-anisotropic",
        action="store_true",
        help="keep the stopped regime's fitted world-axis spread (not usable with --tests exact)",
    )
    ap.add_argument(
        "--tests",
        choices=("exact",),
        default="exact",
        help="exact: overlap term per truth location (tables)",
    )
    ap.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="score only this many test tracks (seeded sample); calibration and fits unchanged",
    )
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.dataset == "eindhoven" and not os.path.exists(args.csv):
        raise SystemExit(
            f"{args.csv} not found; run ../scripts/export_eindhoven_csv.py --tracks 20000"
        )
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    if args.dataset == "day-in-the-life":
        from day_in_the_life import load_day_in_the_life

        full, _ = load_day_in_the_life()
    else:
        from reactmetric.datasets.eindhoven import load_eindhoven

        full = load_eindhoven(path=args.csv)
    ids = list(full.ids)
    rng = np.random.default_rng(args.seed)
    pick = [ids[i] for i in rng.choice(len(ids), size=min(args.tracks, len(ids)), replace=False)]
    data = TrajectorySet({t: full[t] for t in pick})
    del full
    calib, test = data.split(fraction=0.3, seed=args.seed)
    if args.test_subset:
        keep = np.random.default_rng(args.seed + 1).permutation(sorted(test.ids))[: args.test_subset]
        test = TrajectorySet({t: test[t] for t in keep})
    if args.stop_anisotropic and args.tests == "exact":
        raise SystemExit("--tests exact needs the isotropic stopped spread")
    ref, ref_fit = fit_reference(
        calib,
        args.velocity_window,
        args.seed,
        cap_still=not args.no_still_cap,
        stop_isotropic=not args.stop_anisotropic,
    )
    chosen = {k: v for k, v in (ref_fit or {}).items() if k != "grid"}
    print(
        f"{len(calib)} calibration / {len(test)} test tracks; climatology fitted "
        f"(velocity window {args.velocity_window}, {chosen})",
        flush=True,
    )
    tpp = (
        dict(tpp_dir=args.tpp_dir, model_dir=args.tpp_model, checkpoint=args.tpp_ckpt)
        if args.tpp_dir
        else None
    )
    specs, fitted = fit_models(
        calib, args.gru_dir, args.gru_at5_dir, args.imm_params, tpp, args.ctrv_params,
        tpp_gaussian=not args.no_tpp_gaussian,
    )
    MODEL_NAMES = list(specs) + ["oracle_B"]
    print(
        f"models fitted ({(time.time() - t_start) / 60:.1f} min): "
        + json.dumps(
            {
                m: {k: round(v, 4) if isinstance(v, float) else v for k, v in f.items()}
                for m, f in fitted.items()
            }
        ),
        flush=True,
    )

    event_fit = {"detector": args.events}
    if args.events == "bayes-sg":
        hazard = args.hazard if args.hazard is not None else bcp.TUNED_SG["hazard"]
        det_spec = ("bayes-sg", float(hazard))
        event_fit["hazard"] = float(hazard)
        event_fit["settings"] = dict(bcp.TUNED_SG, hazard=float(hazard))
    elif args.events == "bayes":
        if args.hazard is None:
            hazard, curve = bcp.fit_hazard([calib[t] for t in calib.ids], seed=args.seed, jobs=args.jobs)
            event_fit["hazard_curve"] = curve
        else:
            hazard = args.hazard
        det_spec = ("bayes", float(hazard))
        event_fit["hazard"] = float(hazard)
    print(f"event source: {event_fit} ({(time.time() - t_start) / 60:.1f} min)", flush=True)
    nf = str(args.noise_floor).strip()
    if nf.lower() == "none":
        sigma = 0.0
    elif nf.upper() == "A":
        sigma = "estimate"
    else:
        sigma = float(nf) / 100.0
    lead_info = None
    if args.lead1_levels is not None:
        keep = {round(float(x), 4) for x in args.lead1_levels}
        lead_info = {1: [x for x in args.levels if round(float(x), 4) in keep]}
    forecasts = partial(build_models, specs)
    model_leads = {m: sp[2] for m, sp in specs.items() if sp[2] is not None}
    scorer = ReactScorer.fit(
        calib, info=args.levels, leads=LEADS, sigma=sigma, distortions=args.distortions,
        events=make_detector(det_spec), p=args.p, eps=args.eps, lead_info=lead_info,
        entropy_margin_info=args.entropy_margin_levels, climatology=ref, forecasts=forecasts,
        model_leads=model_leads, table_dir=os.path.join(args.out, "tables"), n_jobs=args.jobs,
        seed=args.seed,
    )
    noise_fit, calib_summary = scorer.noise_floor, scorer.diagnostics
    print(f"measurement-noise floor {noise_fit}", flush=True)
    print(
        f"calibration and overlap tables done ({(time.time() - t_start) / 60:.1f} min; per table "
        + ", ".join(f"{k}: {v:.0f} s" for k, v in scorer.table_seconds.items()) + ")",
        flush=True,
    )

    t_test = time.time()

    def progress(n_done, n):
        if n_done % max(1, n // 10) == 0 or n_done == n:
            print(f"  {n_done}/{n} tasks ({time.time() - t_test:.0f} s)", flush=True)

    # the paper scored each model on every event it covers
    result = scorer.evaluate(forecasts, test, model_leads=model_leads, n_jobs=args.jobs,
                             chunk_tracks=args.chunk_tracks, progress=progress, event_set="all")
    records, clock = result.records, result.timing
    print(
        "test phase cpu seconds (all workers): "
        + ", ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in clock.items()),
        flush=True,
    )
    summary = summarize(records)
    print(
        f"measurement done ({(time.time() - t_start) / 60:.1f} min, "
        f"{len(records)} event-model-level rows)"
    )
    levels = {lead: [lv.info for lv in scorer.levels[lead]] for lead in LEADS}
    rule = f"P >= {args.p:g}"

    tag = f"seed{args.seed}_n{args.tracks}_vw{args.velocity_window}"
    if args.no_still_cap:
        tag += "_uncapped"
    with open(os.path.join(args.out, f"records_{tag}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "reference_fit": ref_fit,
                "model_fits": fitted,
                "event_source": event_fit,
                "noise_floor": noise_fit,
                "calibration": {str(k): v for k, v in calib_summary.items()},
                "results": [
                    {
                        "lead": k[0],
                        "info": k[1],
                        "distortion": k[2],
                        "distortion_label": f"{k[2]:g}m" if k[2] else CLIM,
                        "model": k[3],
                        **v,
                    }
                    for k, v in summary.items()
                ],
            },
            f,
            indent=1,
        )

    print("\nCALIBRATION (calibration tracks, calm stretches)")
    for lead in LEADS:
        cs = calib_summary[lead]
        print(
            f"  lead {lead:>2}: calm-stretch mean score ("
            + ", ".join(f"{m} {v:+.2f}" for m, v in cs["calm_mean_score"].items())
            + ")"
            f";  climatology residual mean square {cs['reference_mean_square']:.2f}"
            " (1 = calibrated)"
        )
        print("           rho_w lags 1-10: " + " ".join(f"{v:+.2f}" for v in cs["rho_w"]))
        for d, row in cs["margin_median_nats"].items():
            print(
                f"           median margin at d = {d} m (nats; covariance / mixture entropy): "
                + "; ".join(
                    f"{k} (n {r['n']}) {r['entropy_cov']:.2f} / {r['entropy_mix']:.2f}"
                    for k, r in row.items()
                )
            )
        for key, ft in cs["fits"].items():
            print(
                f"           I = {key} nats (c = {ft['c']:.3f}): knowable correlation "
                f"scale {ft['scale']:.2f}, tau {ft['tau']:.1f} frames, "
                f"omega {ft['omega']:.2f} rad/frame"
            )

    print(
        f"\nRESULTS (test tracks; eps = {args.eps}, {rule}; "
        f"target: {CLIM} = the climatology's own spread, no distortion target)"
    )
    hdr = (
        f"{'lead':>4} {'I (nats)':>8} {'model':>16} | {'events':>6} {'KM REACT':>8} {'onset':>5} | "
        f"{'recov':>5} {'cens':>5} {'better':>6} {'indist':>6} {'relapse':>7} {'rel3':>5} | "
        f"{'anticip':>7} {'RMST':>6} | "
        f"{'mean s':>7} {'mean I':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for lead in LEADS:
        for info in levels[lead]:
            for m in MODEL_NAMES:
                v = summary.get((lead, round(float(info), 4), 0.0, m))
                if v is None:
                    continue
                print(
                    f"{lead:>4} {info:>8.2f} {m:>16} | {v['events']:>6} {fmt(v['km_react'], 8)} "
                    f"{fmt(v['median_onset'], 5)} | {v['recovered']:>5.2f} {v['censored']:>5.2f} "
                    f"{v['better']:>6.2f} {v['indistinguishable']:>6.2f} "
                    f"{fmt(v['relapse'], 7, 2)} {fmt(v['relapse3'], 5, 2)} | "
                    f"{v['anticipation']:>7.2f} {fmt(v['rmst'], 6, 1)} | "
                    f"{v['mean_s_mixture']:>+7.2f} {v['mean_I']:>+7.2f}"
                )
        print()

    if args.distortions:
        print(
            "\nREACT@d, KM-median REACT (frames) under the covariance margin (default) | the "
            "mixture-entropy margin (_H); mean margin (nats) the same; recovered share"
        )
        for lead in LEADS:
            for info in levels[lead]:
                for m in MODEL_NAMES:
                    cells = []
                    for d in args.distortions:
                        v = summary.get((lead, round(float(info), 4), float(d), m))
                        if v is not None:
                            cells.append(
                                f"d={d:g}: {fmt(v['km_react'], 4)} | {fmt(v['km_react_H'], 4)}"
                                f" (margin {v['margin_mean']:.2f} | {v['margin_mean_H']:.2f},"
                                f" rec {v['recovered']:.2f} | {v['recovered_H']:.2f})"
                            )
                    if cells:
                        print(f"  lead {lead:>2} I {info:>4.2f} {m:>16}: " + "   ".join(cells))

    prev = os.path.join(ROOT, "outputs", "exp09", f"seed{args.seed}_n{args.tracks}.json")
    if os.path.exists(prev):
        p9 = json.load(open(prev))
        print(
            "SAME EVENTS UNDER THE EARLIER RULES (exp09): "
            "KM-median REACT, 3/3 rule | posterior rule"
        )
        for m in MODEL_NAMES:
            cells = [p9["cells"].get(f"{m}|{lead}") for lead in LEADS]
            print(
                f"  {m:>16}: "
                + "   ".join(
                    f"lead {lead}: {fmt(c['cur_km'], 4)} | {fmt(c['post_km'], 4)}"
                    for lead, c in zip(LEADS, cells)
                    if c
                )
            )
    print(f"\ntotal {(time.time() - t_start) / 60:.1f} min; outputs in {args.out}")


if __name__ == "__main__":
    main()
