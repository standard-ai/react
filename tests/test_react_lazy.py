"""ri.classify_lazy (experiments/react_information.py): the lazily tested outcome equals the
full-window rule with the onset window, recovery search and relapse window, and tests fewer frames."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import react_information as ri  # noqa: E402

LOG_K = 1.0


def _brute(d_on, d_rec, frames, t0, onset_window, win, run_len=3):
    """Reference implementation on full-window statistics (one channel)."""
    hit = (d_on >= LOG_K) & (frames - t0 <= onset_window)
    ahead = d_rec >= LOG_K
    on = np.flatnonzero(hit)
    if not len(on):
        early = ahead & (frames - t0 <= onset_window)
        return {"status": "better" if early.any() else "indistinguishable"}
    o = on[0]
    rec = np.flatnonzero(ahead & (np.arange(len(frames)) > o))
    if not len(rec):
        return {"status": "censored", "onset": frames[o] - t0}
    r = rec[0]
    after = np.flatnonzero((frames > frames[r]) & (frames <= frames[r] + win))
    hits = d_on[after] >= LOG_K
    # runs may extend run_len - 1 frames beyond the window
    ext = np.flatnonzero((frames > frames[r]) & (frames <= frames[r] + win + run_len - 1))
    h2 = d_on[ext] >= LOG_K
    st = [i for i in range(len(after)) if i + run_len <= len(ext) and h2[i : i + run_len].all()]
    return {
        "status": "recovered",
        "onset": frames[o] - t0,
        "react": frames[r] - t0,
        "relapse": bool(hits.any()),
        "relapse3": bool(len(st)),
        "relapse3_after": int(frames[ext[st[0]]] - frames[r]) if st else None,
    }


@pytest.mark.parametrize("seed", range(40))
def test_classify_lazy_matches_full_window(seed):
    rng = np.random.default_rng(seed)
    n, n_ch, t0, ow = int(rng.integers(25, 120)), 3, 100, int(rng.integers(6, 20))
    frames = t0 + 1 + np.arange(n)
    p = rng.uniform(0.02, 0.4)
    d_on = np.where(rng.random(n) < p, 2.0, 0.0)
    d_rec = np.where(rng.random((n_ch, n)) < p / 2, 2.0, 0.0)
    tested = []

    def evaluate(idx, chs, need_on):
        tested.append(np.asarray(idx))
        out = {"rec": d_rec[chs][:, idx], "lbf_rec": d_rec[chs][:, idx] - 1.0}
        if need_on or 0 in chs:
            out.update(on=d_on[idx], lbf_on=d_on[idx] - 1.0, mean_I=np.zeros(len(idx)))
        return out

    outs, _, st = ri.classify_lazy(evaluate, frames, t0, LOG_K, n_ch, ow)
    for ch in range(n_ch):
        ref = _brute(d_on, d_rec[ch], frames, t0, ow, ri.RELAPSE_WINDOW)
        for k, v in ref.items():
            assert outs[ch][k] == v, (seed, ch, k)
    assert max(int(t.max()) for t in tested if len(t)) < n  # nothing past the window end


def test_classify_lazy_oracle_first_ahead_and_frames_tested():
    n, t0 = 200, 50
    frames = t0 + 1 + np.arange(n)
    d_rec = np.full((2, n), -5.0)
    d_rec[0, 7], d_rec[1, 41] = 3.0, 3.0
    calls = []

    def evaluate(idx, chs, need_on):
        calls.append((len(idx), tuple(chs)))
        return {"rec": d_rec[chs][:, idx], "lbf_rec": d_rec[chs][:, idx], "mean_I": np.zeros(len(idx)),
                "on": np.zeros(len(idx)), "lbf_on": np.zeros(len(idx))}

    _, fa, st = ri.classify_lazy(evaluate, frames, t0, LOG_K, 2, 10, oracle=True)
    assert fa == [7, 41]
    assert sum(c[0] for c in calls) < 70  # stops once both channels are found (not all 200 rows)
