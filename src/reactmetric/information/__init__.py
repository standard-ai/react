"""The information formulation of REACT (Breseman and Woollard, TGF 2026, section 3).

Usefulness is the log-score advantage of a forecast over a climatological reference, in nats.
Decisions concern the model's true advantage at each frame, judged against a latent truth
whose spread is set by an assumed information level and a measurement-noise floor:

* ``truth``: correlation structure of the climatology's residuals and the truth posterior,
  shared by every model;
* ``regime_tests``: posterior probabilities that a Gaussian forecast is behind or ahead of the
  two-regime climatology, by exact quadrature;
* ``overlap``: the climatology's overlap term, computed directly or tabulated per lead and level;
* ``mixture``: the same tests for Gaussian-mixture forecasts, scored natively;
* ``outcomes``: onset, recovery, relapse and anticipation from the per-frame decisions.
"""

from .densities import (
    gaussian_entropy,
    gaussian_logpdf,
    gh_nodes_2d,
    mixture_entropy,
    mixture_logpdf,
    mixture_moments,
    mixture_prep,
)
from .margins import distortion_margin
from .mixture import MixtureForecast, RegimeTestsMixture
from .noise import estimate_noise_floor
from .outcomes import (
    RECOVERY_BLOCK,
    RELAPSE_RUN,
    RELAPSE_WINDOW,
    classify,
    classify_lazy,
    oracle_outcome,
)
from .overlap import OverlapTable, overlap_on_grid
from .regime_tests import RegimePosterior, RegimeTests, advantage_quadratic, regime_predictive
from .truth import (
    fit_knowable_corr,
    info_to_c,
    knowable_corr,
    knowable_share,
    overlap_corr,
    pooled_autocorr,
    to_world,
    truth_posterior,
)

__all__ = [
    "MixtureForecast",
    "OverlapTable",
    "RECOVERY_BLOCK",
    "RELAPSE_RUN",
    "RELAPSE_WINDOW",
    "RegimePosterior",
    "RegimeTests",
    "RegimeTestsMixture",
    "advantage_quadratic",
    "classify",
    "classify_lazy",
    "distortion_margin",
    "estimate_noise_floor",
    "fit_knowable_corr",
    "gaussian_entropy",
    "gaussian_logpdf",
    "gh_nodes_2d",
    "info_to_c",
    "knowable_corr",
    "knowable_share",
    "mixture_entropy",
    "mixture_logpdf",
    "mixture_moments",
    "mixture_prep",
    "oracle_outcome",
    "overlap_corr",
    "overlap_on_grid",
    "pooled_autocorr",
    "regime_predictive",
    "to_world",
    "truth_posterior",
]
