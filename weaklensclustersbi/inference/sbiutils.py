"""
Utilities for SBI-based inference.

This module provides helper functions for creating observations
and computing summary statistics for simulation-based inference.
"""

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

PERCENTILE_LEVELS: list[int] = [5, 16, 25, 50, 75, 84, 95]


def create_observation_nfw(
    mc_pair: NDArray[np.floating] | list[float],
    nfw_profile: NDArray[np.floating],
) -> tuple[Tensor, Tensor]:
    """
    Create an observation of an NFW profile for SBI.

    Parameters
    ----------
    mc_pair : array-like
        Mass-concentration pair or summary statistics.
    nfw_profile : NDArray[np.floating]
        NFW profile values at radial bins.

    Returns
    -------
    tuple[Tensor, Tensor]
        (theta, x) - parameters and observations as PyTorch tensors.
    """
    theta_truth_np = np.array(mc_pair)
    x_truth_np = np.array(nfw_profile)

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o


def _compute_mc_summary(mc_pairs: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute percentile summary statistics and correlation for mass-concentration pairs.

    Parameters
    ----------
    mc_pairs : NDArray[np.floating]
        Array of (log10mass, concentration) pairs with shape (N, 2).

    Returns
    -------
    NDArray[np.floating]
        Concatenated array of mass percentiles, concentration percentiles, and correlation.
        Shape is (2 * len(PERCENTILE_LEVELS) + 1,).
    """
    percentiles = np.percentile(mc_pairs, PERCENTILE_LEVELS, axis=0)
    corr = float(
        np.nan_to_num(np.corrcoef(mc_pairs.T)[0, 1], nan=0.0, posinf=0.0, neginf=0.0)
    )
    corr = float(np.clip(corr, -0.99, 0.99))
    return np.concatenate((percentiles[:, 0], percentiles[:, 1], [corr]))


def create_join_fit_observation_nfw(
    mc_pairs: NDArray[np.floating],
    nfw_profiles: NDArray[np.floating],
    stack_estimator: str = "median",
    sigmas: NDArray[np.floating] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Create a join-then-fit observation for SBI.

    Takes many model inputs (mc_pairs) and outputs (nfw_profiles), computes
    percentile summary statistics for parameters and the median profile.

    Parameters
    ----------
    mc_pairs : NDArray[np.floating]
        Array of (log10mass, concentration) pairs with shape (N, 2).
    nfw_profiles : NDArray[np.floating]
        Array of NFW profiles with shape (N, n_radial_bins).

    Returns
    -------
    tuple[Tensor, Tensor]
        (theta, x) - percentile summaries and median profile as PyTorch tensors.
    """
    theta = _compute_mc_summary(mc_pairs)
    if stack_estimator == "median" and sigmas is None:
        stacked_profile = np.median(nfw_profiles, axis=0)
    else:
        from .stackutils import stack_log_profiles

        if sigmas is None:
            raise ValueError("stack_estimator != 'median' requires per-bin sigmas")
        stacked_profile, _ = stack_log_profiles(nfw_profiles, sigmas, stack_estimator)
    return create_observation_nfw(theta, stacked_profile)


def create_fit_join_observation_nfw(
    mc_pairs: NDArray[np.floating],
    nfw_profiles: NDArray[np.floating],
) -> tuple[Tensor, Tensor]:
    """
    Create a fit-then-join observation for SBI.

    Takes many model inputs (mc_pairs) and outputs (nfw_profiles), computes
    percentile summary statistics for parameters with all profiles concatenated.

    Parameters
    ----------
    mc_pairs : NDArray[np.floating]
        Array of (log10mass, concentration) pairs with shape (N, 2).
    nfw_profiles : NDArray[np.floating]
        Array of NFW profiles with shape (N, n_radial_bins).

    Returns
    -------
    tuple[Tensor, Tensor]
        (theta, x) - percentile summaries and flattened profiles as PyTorch tensors.
    """
    theta = _compute_mc_summary(mc_pairs)
    return create_observation_nfw(theta, nfw_profiles.flatten())
