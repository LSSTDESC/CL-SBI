"""
SBI (Simulation-Based Inference) module for weak lensing cluster analysis.

This module provides neural posterior estimation using SNPE (Sequential Neural
Posterior Estimation) for inferring mass and concentration parameters from
weak lensing observations.

The workflow is:
1. Create an inferrer with priors using gen_inferrer()
2. Train the posterior using train_inferrer() with simulated data
3. Apply observations using apply_observations() to sample from the posterior
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
from sbi.utils import BoxUniform
from sbi.inference import SNPE
import torch
from torch import Tensor

from ..types import PriorConfig


def gen_inferrer(priors: PriorConfig, param_dim: int = 2) -> SNPE:
    """
    Create an SNPE inferrer with the priors from the inference config.

    Parameters
    ----------
    priors : PriorConfig
        Prior configuration dictionary containing min/max values for
        log10mass and concentration.
    param_dim : int, optional
        Parameter dimensionality. Default is 2 (mass, concentration).
        For percentile-based inference, use 2*n_levels + 1 (with correlation).

    Returns
    -------
    SNPE
        An SNPE inferrer ready for training.

    Raises
    ------
    ValueError
        If param_dim is less than 2 or has unexpected parity.
    """
    if param_dim < 2 or param_dim % 2 not in (0, 1):
        raise ValueError(
            "Unexpected parameter dimensionality: expected percentiles plus optional correlation."
        )

    has_correlation = param_dim % 2 == 1
    n_levels = (param_dim - int(has_correlation)) // 2
    lower = [priors["min_log10mass"]] * n_levels + [
        priors["min_concentration"]
    ] * n_levels
    upper = [priors["max_log10mass"]] * n_levels + [
        priors["max_concentration"]
    ] * n_levels

    if has_correlation:
        lower.append(-1.0)
        upper.append(1.0)

    prior = BoxUniform(
        torch.as_tensor(lower, dtype=torch.float32),
        torch.as_tensor(upper, dtype=torch.float32),
    )
    return SNPE(prior, density_estimator="mdn", device="cpu")


def train_inferrer(
    inferrer: SNPE,
    sample_mc_pairs: NDArray[np.floating],
    simulated_nfw_profiles: NDArray[np.floating],
) -> Any:
    """
    Train the inferrer with simulations and build a posterior.

    Parameters
    ----------
    inferrer : SNPE
        The SNPE inferrer created by gen_inferrer().
    sample_mc_pairs : NDArray[np.floating]
        Array of (log10mass, concentration) pairs used for training.
    simulated_nfw_profiles : NDArray[np.floating]
        Array of simulated NFW profiles corresponding to the mc_pairs.

    Returns
    -------
    DirectPosterior
        A trained posterior ready for sampling given observations.
    """
    theta_np = np.array(sample_mc_pairs.T).T
    x_np = simulated_nfw_profiles.reshape(simulated_nfw_profiles.shape[0], -1)

    theta = torch.as_tensor(theta_np, dtype=torch.float32)
    x = torch.as_tensor(x_np, dtype=torch.float32)

    inferrer = inferrer.append_simulations(theta, x)

    density_estimator = inferrer.train(
        num_atoms=4,
        training_batch_size=50,
        learning_rate=0.0005,
        validation_fraction=0.1,
        stop_after_epochs=20,
        max_num_epochs=500,
        clip_max_norm=5.0,
        calibration_kernel=None,
        resume_training=False,
        discard_prior_samples=False,
        use_combined_loss=False,
        show_train_summary=False,
        dataloader_kwargs=None,
    )

    posterior = inferrer.build_posterior(density_estimator)
    return posterior


def apply_observations(
    posterior: Any,
    posterior_jtf: Any,
    drawn_mc_pairs: NDArray[np.floating],
    drawn_nfw_profiles: NDArray[np.floating],
    err_dex: float = 0.0,
    stack_estimator: str = "median",
    sigmas: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    Tensor,
    Tensor,
]:
    """
    Apply observations to trained posteriors and sample.

    Parameters
    ----------
    posterior : DirectPosterior
        Trained posterior for fit-then-join inference.
    posterior_jtf : DirectPosterior
        Trained posterior for join-then-fit inference.
    drawn_mc_pairs : NDArray[np.floating]
        Observed mass-concentration pairs.
    drawn_nfw_profiles : NDArray[np.floating]
        Observed NFW profiles.
    err_dex : float, optional
        Error in dex (not currently used). Default is 0.0.

    Returns
    -------
    tuple
        (samples_jtf, samples_ftj, map_mc_jtf, map_mc_ftj, logp_jtf, logp_ftj)
        Samples and MAP estimates for both inference strategies.
    """
    from .sbiutils import (
        create_join_fit_observation_nfw,
        create_fit_join_observation_nfw,
        PERCENTILE_LEVELS,
    )

    # Join-then-fit: stack the observations (median or bias-corrected mean) then fit
    theta_o_jf, x_o_jf = create_join_fit_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles,
        stack_estimator=stack_estimator, sigmas=sigmas,
    )

    samples_jf = posterior_jtf.sample((10000,), x=x_o_jf)
    logp_jf = posterior_jtf.log_prob(samples_jf, x=x_o_jf)
    idx_jf = torch.argmax(logp_jf)
    map_mc_jf = samples_jf[idx_jf]

    # Fit-then-join: fit each observation then combine
    theta_o_fj, x_o_fj = create_fit_join_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles
    )

    samples_fj = posterior.sample((10000,), x=x_o_fj)
    logp_fj = posterior.log_prob(samples_fj, x=x_o_fj)
    idx_fj = torch.argmax(logp_fj)
    map_mc_fj = samples_fj[idx_fj]

    def collapse_map(vec: Tensor) -> Tensor:
        dim = vec.ndim if hasattr(vec, "ndim") else vec.dim()
        if dim == 1 and vec.shape[0] >= 2 * len(PERCENTILE_LEVELS):
            med_idx = PERCENTILE_LEVELS.index(50)
            levels = len(PERCENTILE_LEVELS)
            return torch.stack((vec[med_idx], vec[levels + med_idx]))
        return vec

    map_mc_jf = collapse_map(map_mc_jf)
    map_mc_fj = collapse_map(map_mc_fj)

    return (
        samples_jf.numpy(),
        samples_fj.numpy(),
        map_mc_jf.numpy(),
        map_mc_fj.numpy(),
        logp_jf,
        logp_fj,
    )
