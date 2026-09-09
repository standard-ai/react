"""Scoring rules used by REACT.

`nll` is the headline log-score (a strictly proper scoring rule whose pointwise
difference vs a reference is the information score REACT keys off). `crps` adds
the calibration-robust CRPS / energy-score variants.
"""

from .nll import (
    LOG_2PI,
    GaussianMixtureND,
    eps_star_from_cov,
    fuse_with_gaussian,
    gaussian_nll,
    supported_distortion_from_cov,
)

__all__ = [
    "LOG_2PI",
    "GaussianMixtureND",
    "gaussian_nll",
    "eps_star_from_cov",
    "supported_distortion_from_cov",
    "fuse_with_gaussian",
]

# CRPS / energy score are imported lazily by name to avoid a hard import cycle
# during package init; see reactmetric.scoring.crps.
