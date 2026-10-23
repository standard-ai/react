"""Scoring rules used by REACT.

`nll` is the log score, a strictly proper scoring rule whose pointwise difference
from the climatology is the realized information score REACT keys off.
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
