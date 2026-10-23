"""The measurement-noise floor sigma of the truth spread (paper section 3.7, floor "A")."""

from __future__ import annotations

import numpy as np


def estimate_noise_floor(tracks, window=51, quad=9, still=0.10):
    """Measurement-noise floor sigma (m per axis): floor "A" of docs/react-spec.md.

    The root-mean-square residual about a `quad`-frame local quadratic fit (centre-frame
    residual, rescaled by 1 / sqrt(1 - leverage) to be unbiased for white noise), taken at
    frames whose `window`-frame neighbourhood is observed stationary (every position within
    `still` m of the window mean). Uses the tracks' observations only. Returns (sigma, n
    residuals per axis, n tracks used).

    Sensor error only: body sway is real motion a forecaster could anticipate. For tracker
    output that has been smoothed this is a lower bound; report sensitivity to sigma."""
    from numpy.lib.stride_tricks import sliding_window_view as swv

    h = quad // 2
    t = np.arange(quad) - h
    X = np.stack([np.ones(quad), t, t**2], 1)
    P = X @ np.linalg.pinv(X)
    kern = (np.eye(quad)[h] - P[h])[::-1]
    scale = 1.0 / np.sqrt(1.0 - P[h, h])
    hb = window // 2
    res, used = [], 0
    for tid in tracks.ids:
        o = np.asarray(tracks[tid].positions, float)
        if len(o) < window:
            continue
        win = swv(o, window, axis=0)  # (T - window + 1, 2, window)
        mx = win.mean(-1)
        stat = np.abs(win - mx[..., None]).max((1, 2)) < still
        if not stat.any():
            continue
        used += 1
        for a in range(2):
            r = np.convolve(o[:, a], kern, mode="valid") * scale  # centre frame i + h
            res.append(r[hb - h : len(o) - hb - h][stat])
    r = np.concatenate(res) if res else np.zeros(1)
    return float(np.sqrt(np.mean(r**2))), int(len(r) // 2), used
