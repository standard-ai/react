"""Distortion targets: the margin a forecast must beat climatology by (paper section 3.3, Eq. 8)."""

from __future__ import annotations

import numpy as np


def distortion_margin(entropy, d):
    """Rate-distortion margin (nats): the information beyond a climatology of differential
    entropy `entropy` needed to localize a 2D position at RMS error d (target N(0, d^2/2 I)),
    clamped at 0: max(0, H_r - (1 + log 2 pi + log(d^2 / 2))).

    The paper's margin passes the entropy of the Gaussian with the climatology mixture's total
    mean and covariance (densities.gaussian_entropy of mixture_moments), which includes the
    between-regime spread. The mixture's own differential entropy understates the requirement
    for a bimodal climatology: a standing person with a 22% chance of walking away looks
    localized within a few cm, but the position to be reported spans the walking range."""
    return np.maximum(entropy - (1.0 + np.log(2.0 * np.pi) + np.log(d * d / 2.0)), 0.0)
