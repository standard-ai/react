"""Event outcomes from per-frame decision statistics: onset, recovery, relapse, anticipation.

Paper section 3.3. Decision statistics are posterior log-odds compared with log(p / (1 - p)).
"""

from __future__ import annotations

import numpy as np

RELAPSE_RUN = 3


def _first_run(hit, n):
    """Index of the first frame of the first run of n consecutive True values, or None."""
    if len(hit) < n:
        return None
    ok = np.convolve(hit.astype(int), np.ones(n, int), mode="valid") == n
    return int(np.argmax(ok)) if ok.any() else None


def classify(lbf_on, lbf_rec, frames, t0, log_k, onset_window=None, first_verdict=True):
    """Outcome of one event window.

    first_verdict: scanning forward from the event, whichever test reaches k first
    settles it; "ahead" first means the event was anticipated and anything later belongs
    to a new regime. Otherwise the first onset (subject to the window) counts even after
    an earlier "ahead" verdict. onset_window (frames after t0): an onset counts only
    within the frames this event can explain, e.g. lead + a few frames; later onsets are
    ignored. Without an onset the event is anticipated: confirmed ("better") if the
    "ahead" test ever reaches k, otherwise unconfirmed ("indistinguishable").

    `relapse` flags a recovered event whose onset test fires again later in the window (a single
    frame suffices); `relapse_after` is the number of frames from recovery to that first renewed
    onset. `relapse3` and `relapse3_after` apply the stricter rule: the onset test must fire on
    RELAPSE_RUN (3) consecutive frames after recovery, and the time is counted to the first frame
    of the first such run.
    """
    hit_on = lbf_on >= log_k
    if onset_window is not None:
        hit_on = hit_on & (frames - t0 <= onset_window)
    on = np.flatnonzero(hit_on)
    ahead = np.flatnonzero(lbf_rec >= log_k)
    if len(on) and (not first_verdict or len(ahead) == 0 or on[0] < ahead[0]):
        o = int(on[0])
        rec = ahead[ahead > o]
        if len(rec):
            r = int(rec[0])
            again = np.flatnonzero(lbf_on[r + 1 :] >= log_k)
            run = _first_run(lbf_on[r + 1 :] >= log_k, RELAPSE_RUN)
            return {
                "status": "recovered",
                "onset": int(frames[o] - t0),
                "react": int(frames[r] - t0),
                "duration": int(frames[r] - t0),
                "relapse": bool(len(again)),
                "relapse_after": int(frames[r + 1 + again[0]] - frames[r]) if len(again) else None,
                "relapse3": run is not None,
                "relapse3_after": int(frames[r + 1 + run] - frames[r]) if run is not None else None,
            }
        return {
            "status": "censored",
            "onset": int(frames[o] - t0),
            "react": None,
            "duration": int(frames[-1] - t0),
            "relapse": False,
            "relapse_after": None,
            "relapse3": False,
            "relapse3_after": None,
        }
    status = "better" if len(ahead) else "indistinguishable"
    return {
        "status": status,
        "onset": None,
        "react": None,
        "duration": None,
        "relapse": False,
        "relapse_after": None,
        "relapse3": False,
        "relapse3_after": None,
    }


# Lazy evaluation: test only the frames that can decide an event's outcome.
RECOVERY_BLOCK = 16  # frames tested per step while searching for recovery
RELAPSE_WINDOW = 10  # frames after a recovery in which a renewed onset counts as a relapse


def classify_lazy(evaluate, frames, t0, log_k, n_ch, onset_window, oracle=False,
                  block=RECOVERY_BLOCK, relapse_window=RELAPSE_WINDOW):
    """Outcome of one event window for n_ch recovery channels (base, each distortion margin),
    testing only the frames that matter. Same headline rule as classify(first_verdict=False,
    onset_window=...), with these differences in what is tested:

    * Onset tests run on the first frames only (frame - t0 <= onset_window). The first onset
      there is shared by every channel.
    * No onset: the event is anticipated, and "better" means a channel's recovery test is
      confidently ahead somewhere in those same onset-window frames (else "indistinguishable").
    * Onset: recovery is searched after it, in blocks of `block` frames, for the channels still
      unrecovered; the search stops when all have recovered or the window ends (censored).
    * Relapse: after a recovery at frame r, a confident onset test in frames r+1 .. r+relapse_window
      (`relapse`), or a run of RELAPSE_RUN consecutive ones starting there (it may extend
      RELAPSE_RUN - 1 frames past the range; `relapse3`), both counted per channel. Nothing later
      counts, and no later frame is tested.
    * oracle: no onset rule; per channel, `first_ahead` is the first confidently-ahead frame,
      searched from the window start (for Oracle B).

    evaluate(idx, chs, need_on) tests window rows idx (array) and returns a dict with "on",
    "lbf_on" (when need_on or channel 0 is requested; arrays over idx), "rec", "lbf_rec"
    (arrays (len(chs), len(idx)); channel 0 is the base) and optionally "mean_I".

    Returns the per-channel outcome dicts (classify's fields), per-channel first_ahead (index
    into frames, oracle only), and the statistics computed (NaN where untested).
    """
    frames = np.asarray(frames)
    n = len(frames)
    don, lon = np.full(n, np.nan), np.full(n, np.nan)
    drec, lrec = np.full((n_ch, n), np.nan), np.full((n_ch, n), np.nan)
    mean_I = np.full(n, np.nan)

    def run(idx, chs, need_on):
        v = evaluate(idx, chs, need_on)
        if "on" in v:
            don[idx], lon[idx] = v["on"], v["lbf_on"]
        for a, ch in enumerate(chs):
            drec[ch, idx], lrec[ch, idx] = v["rec"][a], v["lbf_rec"][a]
        if "mean_I" in v:
            mean_I[idx] = v["mean_I"]

    def search(chs, lo, found):
        """Blocks from row lo: record in found[ch] the first row with recovery >= log_k."""
        pos = lo
        while pos < n and any(found[ch] is None for ch in chs):
            live = [ch for ch in chs if found[ch] is None]
            blk = np.arange(pos, min(pos + block, n))
            run(blk, live, 0 in live)
            for ch in live:
                hit = np.flatnonzero(drec[ch, blk] >= log_k)
                if len(hit):
                    found[ch] = int(blk[hit[0]])
            pos += block

    chs = list(range(n_ch))
    k = int(np.searchsorted(frames - t0, onset_window, side="right"))
    if k:
        run(np.arange(k), chs, True)
    found = [None] * n_ch
    for ch in chs:
        hit = np.flatnonzero(drec[ch, :k] >= log_k)
        if len(hit):
            found[ch] = int(hit[0])
    none = {"onset": None, "react": None, "relapse": False, "relapse_after": None,
            "relapse3": False, "relapse3_after": None}
    stats = {"on": don, "lbf_on": lon, "rec": drec, "lbf_rec": lrec, "mean_I": mean_I, "k": k}
    if oracle:
        search(chs, k, found)
        return [None] * n_ch, found, stats

    on = np.flatnonzero(don[:k] >= log_k)
    if not len(on):  # anticipated: "better" only if confidently ahead within the onset window
        out = [{"status": "better" if found[ch] is not None else "indistinguishable",
                "duration": None, **none} for ch in chs]
        return out, [None] * n_ch, {**stats, "onset": None, "react": [None] * n_ch}
    o = int(on[0])
    found = [None] * n_ch
    for ch in chs:  # recovery is a frame after the onset (rows o + 1 .. k - 1 are already tested)
        hit = np.flatnonzero(drec[ch, o + 1 : k] >= log_k)
        if len(hit):
            found[ch] = o + 1 + int(hit[0])
    search(chs, max(k, o + 1), found)
    # relapse: onset tests on the rows after each recovery (those not yet tested)
    ends = {}
    need = np.zeros(n, bool)
    for ch in chs:
        if found[ch] is not None:
            r = found[ch]
            ends[ch] = (
                int(np.searchsorted(frames, frames[r] + relapse_window, side="right")),
                int(np.searchsorted(frames, frames[r] + relapse_window + RELAPSE_RUN - 1,
                                    side="right")),
            )
            need[r + 1 : ends[ch][1]] = True
    todo = np.flatnonzero(need & np.isnan(don))
    if len(todo):
        run(todo, [], True)
    out = []
    for ch in chs:
        r = found[ch]
        if r is None:
            out.append({**none, "status": "censored", "onset": int(frames[o] - t0),
                        "duration": int(frames[-1] - t0)})
            continue
        hi, hi3 = ends[ch]
        again = np.flatnonzero(don[r + 1 : hi] >= log_k)
        seq = _first_run(don[r + 1 : hi3] >= log_k, RELAPSE_RUN)
        seq = seq if seq is not None and seq < hi - (r + 1) else None
        out.append({
            "status": "recovered",
            "onset": int(frames[o] - t0),
            "react": int(frames[r] - t0),
            "duration": int(frames[r] - t0),
            "relapse": bool(len(again)),
            "relapse_after": int(frames[r + 1 + again[0]] - frames[r]) if len(again) else None,
            "relapse3": seq is not None,
            "relapse3_after": int(frames[r + 1 + seq] - frames[r]) if seq is not None else None,
        })
    return out, [None] * n_ch, {**stats, "onset": o, "react": found}


def oracle_outcome(first_ahead, last):
    """Oracle B's outcome: never behind the climatology, REACT the first confirmed frame
    (frames after the event; None if never confirmed, then censored at `last`)."""
    return {
        "status": "recovered" if first_ahead is not None else "censored",
        "onset": None,
        "react": first_ahead,
        "duration": first_ahead if first_ahead is not None else last,
        "relapse": False,
        "relapse_after": None,
        "relapse3": False,
        "relapse3_after": None,
    }
