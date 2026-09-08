"""
Join-then-fit stacking estimators for log-space profiles.

Central definition of how N noisy log10 profiles are combined into one stacked data
vector, and of the corresponding per-bin uncertainty of that stack. Two estimators:

- "median" (the original choice): per-bin median of the log10 profiles. Unbiased for
  the log of the noiseless profile under log-normal noise, at a sqrt(pi/2) efficiency
  cost relative to a mean.
- "corrected_mean" (IR2 response): per-bin mean of the LINEAR profiles with the known
  log-normal bias factor exp(s^2/2), s = sigma_dex * ln 10, divided out. This is the
  realizable stacked-lensing estimator (stacking in data is a weighted mean over
  source-lens pairs, not a median over per-cluster profiles), made unbiased for the
  noiseless profile by the deterministic correction. Its log-space uncertainty follows
  from the log-normal moments: CV(mean of N) = sqrt(exp(s^2) - 1)/sqrt(N), so
  sigma_stack,log10 = sqrt(exp(s^2) - 1) / (sqrt(N) ln 10).

Both estimators take and return log10-space quantities (profiles are stored in log
space throughout the pipeline) and per-bin noise sigmas in dex.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

STACK_ESTIMATORS = ("median", "corrected_mean")
LN10 = np.log(10.0)


def stack_suffix(estimator: str) -> str:
    """Output-directory suffix for a stack estimator ('' for the default median)."""
    return "" if estimator == "median" else f".{estimator}"


def stack_log_profiles(
    log_profiles: NDArray[np.floating],
    sigmas_dex: NDArray[np.floating],
    estimator: str = "median",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Stack N log10 profiles into one log10 data vector with per-bin uncertainties.

    Parameters
    ----------
    log_profiles : (N, nbins) log10 profiles (noise is log-normal: Gaussian in dex).
    sigmas_dex : (nbins,) per-bin noise sigma of a SINGLE profile, in dex.
    estimator : "median" or "corrected_mean".

    Returns
    -------
    (stacked_log_profile, stacked_sigmas_dex) : ((nbins,), (nbins,))
    """
    log_profiles = np.asarray(log_profiles, dtype=float)
    sigmas_dex = np.asarray(sigmas_dex, dtype=float)
    n = log_profiles.shape[0]

    if estimator == "median":
        stacked = np.median(log_profiles, axis=0)
        stacked_sigmas = np.sqrt(np.pi / 2.0) * sigmas_dex / np.sqrt(n)
        return stacked, stacked_sigmas

    if estimator == "corrected_mean":
        s2 = (sigmas_dex * LN10) ** 2
        # mean of linear profiles, bias factor exp(s^2/2) divided out, back to log10
        linear_mean = np.mean(10.0 ** log_profiles, axis=0)
        stacked = np.log10(linear_mean) - s2 / (2.0 * LN10)
        # log-normal moments: CV of the N-mean = sqrt(exp(s^2)-1)/sqrt(N)
        stacked_sigmas = np.sqrt(np.expm1(s2)) / (np.sqrt(n) * LN10)
        return stacked, stacked_sigmas

    raise ValueError(f"estimator must be one of {STACK_ESTIMATORS}, got {estimator!r}")
