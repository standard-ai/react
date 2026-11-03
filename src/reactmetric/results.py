"""REACT results: per-event records, their summaries, confidence intervals and files.

A ReactResult holds one record per (lead, information level, distortion, model, event).
`table()` summarizes them, one row per model, under one information level and one target
distance; `ci()` and `compare()` add track-level bootstrap intervals (paper section 4.6).

Target distances: d = infinity (no target, the plain comparison with the climatology) is
written `distortion=None`; 0, inf and "clim" are accepted for it. Records store it as 0.0 with
the label "clim".
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metric.survival import kaplan_meier, km_median

# label of the base case (distortion 0.0 in the records): no target, d = infinity
CLIM = "clim"
WINDOW = 150  # the paper's event window (frames): restricted means are taken up to it

# the brief table: summary statistic -> column name
BRIEF = {
    "events": "events",
    "anticipation": "anticipated",
    "recovered": "recovered",
    "censored": "censored",
    "km_react": "median_react",
    "rmst": "rmst",
    "relapse3": "relapsed",
}

GLOSSARY = {
    "events": "events scored for this model (shared by every model unless event_set='all')",
    "anticipated": "share of events with no onset: the forecast never fell confidently behind "
                   "the climatology within lead + onset_margin frames of the event",
    "recovered": "share of events with an onset and then a confident recovery within the window",
    "censored": "share of events with an onset and no recovery before the window ended",
    "median_react": "median frames from the event to recovery, given onset (Kaplan-Meier; NaN "
                    "when fewer than half of the onsets recover within the window)",
    "rmst": "restricted mean frames to recovery over all events (anticipated ones count as 0), "
            "up to the window; lower is better",
    "relapsed": "share of recovered events that fall behind again: relapse_run consecutive "
                "frames starting within relapse_window frames of recovery",
    "rmst_cond": "restricted mean frames to recovery, given onset",
    "km_react_overall": "median frames to recovery over all events (anticipated count as 0)",
    "better": "share of events with no onset and confidently ahead within the onset window "
              "(part of anticipated)",
    "indistinguishable": "share of events with no onset and not confidently ahead either (part "
                         "of anticipated)",
    "median_onset": "median frames from the event to onset, over events with an onset",
    "relapse": "share of recovered events with any single 'behind' frame within relapse_window",
    "relapse_after_median": "median frames from recovery to that relapse",
    "relapse3_after_median": "median frames from recovery to the start of the 3-frame relapse",
    "margin_mean": "mean distortion margin (nats) the forecast had to beat the climatology by",
    "mean_s_mixture": "mean realized score (nats/frame) against the climatology mixture",
    "mean_s_gaussian": "mean realized score (nats/frame) against the climatology's Gaussian",
    "mean_pre_s": "mean realized score over the frames before the event",
    "mean_I": "posterior mean information advantage (nats/frame) over the onset window",
    "_H": "suffix: the same outcome under the mixture-entropy distortion margin",
}

# record fields by type, for reading records back from CSV
_INT = {"lead", "t0", "n", "I_n", "pre_n", "onset", "react", "duration", "relapse_after",
        "relapse3_after", "react_H", "duration_H", "relapse_after_H", "relapse3_after_H",
        "first_ahead", "first_ahead_H"}
_BOOL = {"relapse", "relapse3", "relapse_H", "relapse3_H", "onset_prior_driven",
         "recovery_prior_driven"}
_STR = {"distortion_label", "model", "track", "event_type", "status", "status_H"}


def _dist(d):
    """A distortion filter value: None, 0, inf and "clim" all mean d = infinity (0.0 in the
    records); "all" means no filter."""
    if isinstance(d, str):
        if d.lower() in ("clim", "inf", "infinity"):
            return 0.0
        if d.lower() == "all":
            return "all"
        if d.endswith("m"):
            d = d[:-1]
        d = float(d)
    if d is None or d == 0 or np.isinf(d):
        return 0.0
    if d < 0:
        raise ValueError(f"distortion must be positive metres or None (d = infinity); got {d}")
    return float(d)


def _dist_label(d):
    return "inf" if d == 0 else f"{d:g} m"


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _rmst(t_, s_, window=WINDOW):
    t_ = np.append(np.minimum(t_, window), window)
    return float(np.sum(np.diff(t_) * s_))


def km_summary(rs, sfx="", window=WINDOW):
    """Anticipation and REACT for one group under one outcome rule (paper section 3.3).

    Given onset: Kaplan-Meier over events with an onset (time to recovery given the forecast
    fell behind); median REACT is its median. Over all events: anticipated ones (no onset)
    recover at frame 0. Besides medians, the restricted mean (area under the Kaplan-Meier curve
    up to `window` frames; Royston & Parmar 2013), which stays informative when over half the
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
        out["rmst_cond"] = _rmst(t_, s_, window)
    d_all = np.where(onset, dur, 0.0)
    obs_all = (st == "recovered") | ~onset
    t_, s_ = kaplan_meier(d_all, obs_all)
    out["km_react_overall"] = float(km_median(t_, s_))
    out["rmst"] = _rmst(t_, s_, window)
    return out


def _by_key(by):
    if by is None:
        return ()
    return (by,) if isinstance(by, str) else tuple(by)


def summarize(records, by=None, window=WINDOW):
    """Per (lead, information level, distortion, model[, by fields]) group: the REACT profile.

    by: a record field or fields to break the groups down by, e.g. "event_type". Relapse rates
    are over recovered events; mean_I is over the onset-window frames (tested for every
    event)."""
    extra = _by_key(by)
    groups = {}
    for r in records:
        key = (r["lead"], r["info"], r.get("distortion", 0.0), r["model"],
               *(r.get(f) for f in extra))
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rs in groups.items():
        onset = [r for r in rs if r["status"] in ("recovered", "censored")]
        rec = [r for r in rs if r["status"] == "recovered"]
        n_frames = sum(r["n"] for r in rs)
        n_pre = sum(r["pre_n"] for r in rs)
        n_I = sum(r["I_n"] for r in rs)
        rs_h = [r for r in rs if r["status_H"] is not None]
        under_h = {f"{k}_H": v for k, v in km_summary(rs_h, "_H", window).items()}
        rec_h = [r for r in rs_h if r["status_H"] == "recovered"]
        on0 = [r["onset"] for r in onset if r["onset"] is not None]

        def share(xs, k):
            return float(np.mean([r[k] for r in xs])) if xs else float("nan")

        def med_after(xs, k):
            v = [r[f"{k}_after"] for r in xs if r[k]]
            return float(np.median(v)) if v else float("nan")

        out[key] = {
            "events": len(rs),
            **km_summary(rs, window=window),
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


def select_info_level(summary, distortion=None, threshold: float = 0.95,
                      model: str = "oracle_B"):
    """The paper's rule for the headline information level (section 3.7): the lowest assumed
    level at which Oracle B recovers on at least `threshold` of events at every lead evaluated
    at that level.

    summary: ReactResult.summary() (keys (lead, info, distortion, model)). distortion: the
    target the rule is judged at (None for d = infinity). The paper does not state it; on
    Eindhoven the rule gives the published 4.2 nats at d = 0.1 m (3 nats at d = infinity).

    Returns (level or None, {level: {lead: recovered share}})."""
    distortion = _dist(distortion)
    shares: Dict[float, Dict[int, float]] = {}
    for key, row in summary.items():
        lead, info, dist, m = key[:4]
        if m == model and np.isclose(dist, distortion) and len(key) == 4:
            shares.setdefault(float(info), {})[int(lead)] = float(row["recovered"])
    for info in sorted(shares):
        if all(v >= threshold for v in shares[info].values()):
            return info, shares
    return None, shares


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _wkm(d, obs, w):
    """Weighted Kaplan-Meier (the same steps as kaplan_meier with integer weights)."""
    if w.sum() <= 0:
        return np.array([0.0]), np.array([1.0])
    ut, inv = np.unique(d, return_inverse=True)
    deaths = np.bincount(inv, weights=w * obs, minlength=len(ut))
    total = np.bincount(inv, weights=w, minlength=len(ut))
    at_risk = total[::-1].cumsum()[::-1]
    hit = deaths > 0
    s = np.cumprod(1.0 - deaths[hit] / at_risk[hit])
    return np.r_[0.0, ut[hit]], np.r_[1.0, s]


def _group_arrays(rs):
    st = np.array([r["status"] for r in rs])
    dur = np.array([np.nan if r["duration"] in (None, "") else float(r["duration"]) for r in rs])
    onset = np.isin(st, ("recovered", "censored"))
    return {
        "onset": onset,
        "rec": st == "recovered",
        "cens": st == "censored",
        "dur": np.where(onset, dur, 0.0),
        "relapse3": np.array([bool(r["relapse3"]) for r in rs]),
    }


STATS = ("anticipated", "recovered", "censored", "median_react", "rmst", "rmst_cond", "relapsed")
_ALIASES = {"anticipation": "anticipated", "km_react": "median_react", "relapse3": "relapsed",
            "react": "median_react"}


def _stat(name):
    name = _ALIASES.get(name, name)
    if name not in STATS:
        raise ValueError(f"statistic must be one of {STATS}; got {name!r}")
    return name


def _weighted_stat(g, w, name, window):
    tot = w.sum()
    if name == "anticipated":
        return float((w * ~g["onset"]).sum() / tot)
    if name == "recovered":
        return float((w * g["rec"]).sum() / tot)
    if name == "censored":
        return float((w * g["cens"]).sum() / tot)
    if name == "relapsed":
        wr = (w * g["rec"]).sum()
        return float((w * g["rec"] * g["relapse3"]).sum() / wr) if wr else float("nan")
    if name in ("median_react", "rmst_cond"):
        o = g["onset"]
        if not (w * o).sum():
            return float("nan")
        t_, s_ = _wkm(g["dur"][o], g["rec"][o], w[o])
        if name == "rmst_cond":
            return _rmst(t_, s_, window)
        # undefined (fewer than half recover in the window): beyond the window
        m = km_median(t_, s_)
        return m if np.isfinite(m) else math.inf
    t_, s_ = _wkm(g["dur"], g["rec"] | ~g["onset"], w)  # rmst over all events
    return _rmst(t_, s_, window)


def _interval(draws, level):
    a = (1.0 - level) / 2.0
    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return float("nan"), float("nan")
    # median_react draws can be +inf (open-ended); the quantiles then are too
    with np.errstate(invalid="ignore"):
        lo, hi = np.quantile(d, [a, 1.0 - a], method="inverted_cdf")
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class ReactResult:
    """Per-event REACT records and their summaries.

    Made by ReactScorer.evaluate, or reloaded with ReactResult.load. records: one dict per
    (lead, information level, distortion, model, event)."""

    records: List[Dict]
    scorer: Optional[object] = None
    timing: Dict = field(default_factory=dict)
    coverage: Dict = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    # -- what is in it -------------------------------------------------------------------------

    @property
    def models(self) -> List[str]:
        """Model names in the order they were scored (Oracle B last)."""
        names = list(dict.fromkeys(r["model"] for r in self.records))
        return [m for m in names if m != "oracle_B"] + (["oracle_B"] if "oracle_B" in names
                                                        else [])

    @property
    def leads(self) -> List[int]:
        return sorted({r["lead"] for r in self.records})

    @property
    def info_levels(self) -> List[float]:
        return sorted({r["info"] for r in self.records})

    @property
    def distortions(self) -> List[Optional[float]]:
        """Target distances in the result; None is d = infinity."""
        ds = sorted({r["distortion"] for r in self.records})
        return [None if d == 0 else d for d in ds]

    @property
    def window(self) -> int:
        return int(self.settings().get("window_frames", WINDOW))

    def __repr__(self) -> str:
        n = len({(r["track"], r["t0"]) for r in self.records})
        ds = ", ".join(_dist_label(0 if d is None else d) for d in self.distortions)
        return (f"ReactResult({n} events, models={self.models}, leads={self.leads}, "
                f"info={self.info_levels}, distortions=[{ds}])")

    def settings(self) -> Dict:
        """The assumptions every number in this result carries."""
        sc = self.scorer
        if sc is None:
            return dict(self.meta.get("settings", {}))
        return {
            "leads": list(sc.leads),
            "info_levels": {str(k): [lv.info for lv in v] for k, v in sc.levels.items()},
            "distortions_m": [None, *sc.distortions],
            "decision_probability": sc.p,
            "eps": sc.eps,
            "noise_floor": sc.noise_floor,
            "onset_window": f"lead + {sc.onset_margin} frames",
            "onset_margin_frames": sc.onset_margin,
            "window_frames": sc.window,
            "min_window_frames": sc.min_window,
            "relapse": f"{sc.relapse_run} consecutive frames within {sc.relapse_window} frames",
            "relapse_run_frames": sc.relapse_run,
            "relapse_window_frames": sc.relapse_window,
            "event_set": self.meta.get("event_set", "all"),
            "event_source": type(sc.events).__name__,
        }

    def subset(self, where=None, **fields) -> ReactResult:
        """The records matching every field (a value or a list of values), e.g.
        subset(event_type="stop") or subset(model=["MyModel", "oracle_B"]), and where(record)
        if given. Windows stay as detected."""
        def ok(r):
            for k, v in fields.items():
                if k not in r:
                    raise KeyError(f"records have no field {k!r}")
                if r[k] not in ((v,) if isinstance(v, (str, int, float)) else tuple(v)):
                    return False
            return where is None or bool(where(r))

        recs = [r for r in self.records if ok(r)]
        if not recs:
            raise ValueError(f"no records match {fields}")
        return ReactResult(records=recs, scorer=self.scorer, timing=self.timing,
                           coverage=self.coverage, meta=self.meta)

    # -- summaries -----------------------------------------------------------------------------

    def summary(self, by=None) -> Dict[Tuple, Dict]:
        """One REACT profile per (lead, info, distortion, model[, by]) group. by: a record
        field (or fields) to break groups down by, e.g. "event_type"."""
        return summarize(self.records, by=by, window=self.window)

    def info_level(self, distortion="rule", threshold: float = 0.95, model: str = "oracle_B"):
        """select_info_level on this result: the lowest level at which `model` recovers on at
        least `threshold` of events at every lead, judged at `distortion` ("rule": the smallest
        finite target in the result, d = 0.1 m by default, which gives the paper's 4.2 nats on
        Eindhoven; None: d = infinity). Returns (level or None, shares)."""
        if distortion == "rule":
            finite = [d for d in self.distortions if d is not None]
            distortion = min(finite) if finite else None
        return select_info_level(self.summary(), distortion, threshold, model)

    def headline_info(self) -> Tuple[float, str]:
        """The information level table() uses by default, and why."""
        levels = self.info_levels
        if len(levels) == 1:
            return levels[0], f"the only level evaluated ({levels[0]:g} nats)"
        finite = [d for d in self.distortions if d is not None]
        d = min(finite) if finite else None
        level, shares = self.info_level(d)
        where = "d = inf" if d is None else f"d = {d:g} m"
        if level is None:
            top = max(levels)
            return top, (f"no level meets the paper's rule (Oracle B recovers on >= 95% of "
                         f"events at every lead, at {where}); using the highest, {top:g} nats")
        return level, (f"the paper's rule: the lowest level at which Oracle B recovers on "
                       f">= 95% of events at every lead, at {where}")

    def _groups(self, lead, info, distortion, model):
        """Records matching the filters, raising (with what is available) when none do."""
        if info == "auto":
            info = self.headline_info()[0]
        dist = _dist(distortion)
        rs = self.records
        for name, val, ok in (
            ("lead", lead, lambda r, v: r["lead"] == v),
            ("info", info, lambda r, v: np.isclose(r["info"], float(v))),
            ("distortion", dist, lambda r, v: np.isclose(r["distortion"], v)),
            ("model", model, lambda r, v: r["model"] in ((v,) if isinstance(v, str) else v)),
        ):
            if val is None or (name == "distortion" and val == "all") or (
                    name == "info" and val == "all"):
                continue
            kept = [r for r in rs if ok(r, val)]
            if not kept:
                have = sorted({r[name] for r in rs}, key=str)
                if name == "distortion":
                    have = [_dist_label(x) for x in sorted({r["distortion"] for r in rs})]
                raise ValueError(f"no records with {name}={val!r}; the result has {have}")
            rs = kept
        return rs, info

    def table(self, lead=None, info="auto", distortion=None, model=None, by=None,
              columns="brief", ci: bool = False, n_boot: int = 1000, level: float = 0.95,
              seed: int = 0):
        """The summary as a DataFrame (needs pandas), one row per (lead, info, distortion,
        model[, by]) group.

        info: "auto" (default) the headline level from headline_info() (the paper's rule), a
            level in nats, or "all".
        distortion: None (default) for d = infinity, a target in metres, or "all".
        lead, model: filters (None: all). by: a record field to break rows down by, e.g.
            "event_type".
        columns: "brief" (the REACT profile; see GLOSSARY) or "all" (every summary field).
        ci: add <stat>_low / <stat>_high track-level bootstrap intervals to the brief columns.
        A filter that matches nothing raises. df.attrs["info_reason"] says how info was
        chosen."""
        import pandas as pd

        if columns not in ("brief", "all"):
            raise ValueError("columns must be 'brief' or 'all'")
        if not self.records:
            cols = ["lead", "info", "distortion", "model", *_by_key(by)]
            return pd.DataFrame(columns=cols + (list(BRIEF.values()) if columns == "brief"
                                                else []))
        reason = None
        if info == "auto":
            info, reason = self.headline_info()
        rs, info = self._groups(lead, info, distortion, model)
        summ = summarize(rs, by=by, window=self.window)
        extra = list(_by_key(by))
        rows = []
        for k, v in summ.items():
            row = {"lead": k[0], "info": k[1],
                   "distortion": None if k[2] == 0 else k[2], "model": k[3],
                   **dict(zip(extra, k[4:]))}
            if columns == "brief":
                row.update({BRIEF[s]: v[s] for s in BRIEF})
            else:
                row.update(v)
            rows.append(row)
        order = {m: i for i, m in enumerate(self.models)}
        rows.sort(key=lambda r: (r["lead"], r["info"], r["distortion"] or 0.0,
                                 order.get(r["model"], 99), *[str(r[e]) for e in extra]))
        df = pd.DataFrame(rows)
        if ci and columns == "brief" and len(df):
            for s in ("anticipated", "median_react", "rmst"):
                lo, hi = [], []
                for r in rows:
                    sel = [x for x in rs if x["lead"] == r["lead"] and x["info"] == r["info"]
                           and x["distortion"] == (r["distortion"] or 0.0)
                           and x["model"] == r["model"]
                           and all(x.get(e) == r[e] for e in extra)]
                    out = self._boot(sel, s, n_boot, level, seed)
                    lo.append(out["low"])
                    hi.append(out["high"])
                pos = df.columns.get_loc(s) + 1
                df.insert(pos, f"{s}_high", hi)
                df.insert(pos, f"{s}_low", lo)
        df.attrs["info_reason"] = reason or f"info={info!r} requested"
        return df

    # -- intervals -----------------------------------------------------------------------------

    def _boot(self, rs, stat, n_boot, level, seed):
        stat = _stat(stat)
        tracks = sorted({r["track"] for r in rs})
        code = {t: i for i, t in enumerate(tracks)}
        g = _group_arrays(rs)
        tix = np.array([code[r["track"]] for r in rs])
        est = _weighted_stat(g, np.ones(len(rs)), stat, self.window)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            cnt = np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks))
            draws.append(_weighted_stat(g, cnt[tix].astype(float), stat, self.window))
        lo, hi = _interval(draws, level)
        return {"estimate": est, "low": lo, "high": hi, "events": len(rs),
                "tracks": len(tracks)}

    def ci(self, stat: str = "rmst", *, model: str, lead: int, info="auto", distortion=None,
           n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> Dict:
        """A track-level bootstrap interval for one statistic of one model (paper section 4.6):
        test tracks are resampled with replacement and every event on a drawn track comes
        with it, since events on one track are not independent. Conditional on the fitted
        climatology and truth model; sensitivity to the information level is reported
        separately (compare info levels).

        stat: one of anticipated, recovered, censored, median_react, rmst, rmst_cond, relapsed.
        median_react is +inf in a resample where fewer than half the onsets recover (an
        open-ended interval). Returns {"estimate", "low", "high", "events", "tracks"}."""
        rs, _ = self._groups(lead, info, distortion, model)
        return self._boot(rs, stat, n_boot, level, seed)

    def compare(self, model_a: str, model_b: str, stat: str = "rmst", *, lead: int,
                info="auto", distortion=None, n_boot: int = 2000, level: float = 0.95,
                seed: int = 0) -> Dict:
        """Paired bootstrap of stat(model_a) - stat(model_b) on the events both were scored
        on: each resample draws test tracks with replacement and evaluates both models on the
        same draw. For rmst and median_react, a negative difference means model_a recovers
        sooner.

        Returns {"a", "b", "difference", "low", "high", "share_a_lower", "events", "tracks"};
        share_a_lower is the share of resamples in which model_a's statistic is lower."""
        stat = _stat(stat)
        ra, _ = self._groups(lead, info, distortion, model_a)
        rb, _ = self._groups(lead, info, distortion, model_b)
        both = {(r["track"], r["t0"]) for r in ra} & {(r["track"], r["t0"]) for r in rb}
        ra = sorted([r for r in ra if (r["track"], r["t0"]) in both],
                    key=lambda r: (r["track"], r["t0"]))
        rb = sorted([r for r in rb if (r["track"], r["t0"]) in both],
                    key=lambda r: (r["track"], r["t0"]))
        if not ra:
            raise ValueError(f"{model_a} and {model_b} share no events at lead {lead}")
        tracks = sorted({r["track"] for r in ra})
        code = {t: i for i, t in enumerate(tracks)}
        tix = np.array([code[r["track"]] for r in ra])
        ga, gb = _group_arrays(ra), _group_arrays(rb)
        one = np.ones(len(ra))
        a = _weighted_stat(ga, one, stat, self.window)
        b = _weighted_stat(gb, one, stat, self.window)
        rng = np.random.default_rng(seed)
        diffs, lower = [], 0
        for _ in range(n_boot):
            cnt = np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks))
            w = cnt[tix].astype(float)
            x, y = _weighted_stat(ga, w, stat, self.window), _weighted_stat(gb, w, stat,
                                                                            self.window)
            diffs.append(x - y if not (math.isinf(x) and math.isinf(y)) else 0.0)
            lower += x < y
        lo, hi = _interval(diffs, level)
        return {"a": a, "b": b, "difference": a - b if not (math.isinf(a) and math.isinf(b))
                else 0.0, "low": lo, "high": hi, "share_a_lower": lower / n_boot,
                "events": len(ra), "tracks": len(tracks)}

    # -- files ---------------------------------------------------------------------------------

    def to_dataframe(self):
        """The per-event records as a DataFrame (needs pandas)."""
        import pandas as pd

        return pd.DataFrame(self.records)

    def to_json(self, path: str, records: bool = True) -> None:
        """Settings, summary, diagnostics and (by default) the per-event records as JSON (NaN
        written as null). ReactResult.load(path) reads it back."""

        def clean(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            if isinstance(v, (np.floating, np.integer, np.bool_)):
                return clean(v.item())
            if isinstance(v, dict):
                return {str(k): clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            return v

        rows = [{"lead": k[0], "info": k[1], "distortion": k[2], "model": k[3], **v}
                for k, v in self.summary().items()]
        payload = {"schema": "reactmetric/0.2", "settings": self.settings(), "summary": rows,
                   "coverage": self.coverage}
        if self.scorer is not None:
            payload["diagnostics"] = {str(k): v for k, v in self.scorer.diagnostics.items()}
        elif "diagnostics" in self.meta:
            payload["diagnostics"] = self.meta["diagnostics"]
        if records:
            payload["records"] = self.records
        with open(path, "w") as f:
            json.dump(clean(payload), f, indent=1)

    def to_csv(self, path: str) -> None:
        """Per-event records as CSV. ReactResult.load(path) reads them back (pass the JSON
        from to_json as settings= to keep the settings)."""
        if not self.records:
            open(path, "w").close()
            return
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
            wr.writeheader()
            wr.writerows(self.records)

    @classmethod
    def load(cls, path: str, settings: Optional[str] = None) -> ReactResult:
        """A result saved with to_json (records included) or to_csv. For a CSV, settings: the
        JSON from to_json, to restore the settings (window, levels, noise floor)."""
        if str(path).endswith(".json"):
            with open(path) as f:
                payload = json.load(f)
            if "records" not in payload:
                raise ValueError(f"{path} has no records; save with to_json(records=True)")
            recs = payload["records"]
            for r in recs:  # JSON wrote NaN as null
                for k, v in r.items():
                    if v is None and k not in _INT | _BOOL | _STR and k not in (
                            "speed_before", "speed_after", "heading_change"):
                        r[k] = float("nan")
            meta = {k: payload.get(k) for k in ("settings", "diagnostics") if k in payload}
            meta["event_set"] = payload.get("settings", {}).get("event_set", "all")
            return cls(records=recs, coverage=payload.get("coverage", {}), meta=meta)
        recs = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                recs.append({k: _parse(k, v) for k, v in row.items()})
        meta = {}
        if settings is not None:
            with open(settings) as f:
                payload = json.load(f)
            meta = {k: payload.get(k) for k in ("settings", "diagnostics") if k in payload}
            meta["event_set"] = payload.get("settings", {}).get("event_set", "all")
        return cls(records=recs, meta=meta)


def _parse(k, v):
    if v == "":
        return None
    if k in _STR:
        return v
    if k in _BOOL:
        return v == "True"
    if k in _INT:
        return int(float(v))
    return float(v)



__all__ = ["ReactResult", "select_info_level", "GLOSSARY", "CLIM"]
