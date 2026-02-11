"""
MCMC inference module for weak lensing cluster analysis.

This module provides MCMC-based inference using emcee for fitting
NFW profiles to weak lensing observations.
"""

from typing import Any, Callable
from multiprocessing.pool import Pool

import numpy as np
from numpy.typing import NDArray
import emcee

from ..types import MCMCConfig, PriorConfig

np.random.seed(2807)


def default_config() -> MCMCConfig:
    """
    Return the default MCMC configuration.

    Returns
    -------
    MCMCConfig
        Dictionary containing nwalkers, npar, starts, nsteps_burn, nsteps_per_chain.
    """
    return {
        "nwalkers": 100,
        "npar": 3,
        "starts": np.array([14, 4, 0]),
        "nsteps_burn": 100,
        "nsteps_per_chain": 500,
    }


def _run_mcmc_base(
    log_prob_fn: Callable[..., float],
    priors: PriorConfig,
    data: NDArray[np.floating] | list[NDArray[np.floating]],
    pool: Pool | None = None,
) -> emcee.EnsembleSampler:
    """
    Base MCMC runner that handles the common logic for all MCMC runs.

    Parameters
    ----------
    log_prob_fn : callable
        Log probability function to use (logprob or joint_logprob).
    priors : PriorConfig
        Prior configuration dictionary.
    data : array-like or list
        Data to pass to the log probability function.
    pool : multiprocessing.Pool, optional
        Pool for parallel execution.

    Returns
    -------
    emcee.EnsembleSampler
        The sampler after running burn-in and production chains.
    """
    config = default_config()
    sampler = emcee.EnsembleSampler(
        config["nwalkers"],
        config["npar"],
        log_prob_fn,
        args=[priors, data],
        pool=pool,
    )

    # Add some noise to starting positions for walkers
    starts = config["starts"] + 5 * np.random.uniform(
        size=(config["nwalkers"], config["npar"])
    )

    # burn-in
    print("## burning in ... ")
    pos, prob, stat = sampler.run_mcmc(starts, config["nsteps_burn"])

    # reset the sampler
    sampler.reset()

    # run the full chain
    print("## running the full chain ... ")
    sampler.run_mcmc(pos, config["nsteps_per_chain"])

    return sampler


def run_mcmc(
    truth_val: NDArray[np.floating],
    priors: PriorConfig,
    pool: Pool | None = None,
) -> emcee.EnsembleSampler:
    """
    Run MCMC on a single observation.

    Parameters
    ----------
    truth_val : NDArray[np.floating]
        Observed profile data concatenated with uncertainties.
    priors : PriorConfig
        Prior configuration dictionary.
    pool : multiprocessing.Pool, optional
        Pool for parallel execution.

    Returns
    -------
    emcee.EnsembleSampler
        The sampler after running.
    """
    from .mcmcutils import logprob

    return _run_mcmc_base(logprob, priors, truth_val, pool=pool)


def run_joint_mcmc(
    profiles: list[NDArray[np.floating]],
    priors: PriorConfig,
    pool: Pool | None = None,
) -> emcee.EnsembleSampler:
    """
    Run MCMC jointly on multiple profiles.

    Parameters
    ----------
    profiles : list[NDArray[np.floating]]
        List of observed profile data arrays.
    priors : PriorConfig
        Prior configuration dictionary.
    pool : multiprocessing.Pool, optional
        Pool for parallel execution.

    Returns
    -------
    emcee.EnsembleSampler
        The sampler after running.
    """
    from .mcmcutils import joint_logprob

    return _run_mcmc_base(joint_logprob, priors, profiles, pool=pool)


def fit_then_join(
    profiles: NDArray[np.floating],
    sigmas: NDArray[np.floating],
    priors: PriorConfig,
    pool: Pool | None = None,
) -> tuple[NDArray[np.floating], emcee.EnsembleSampler]:
    """
    Fit each profile with MCMC then join the chains.

    For a given set of profiles, we run MCMC jointly on all of them (fit).
    The chains are combined into a single flat chain (join).

    Parameters
    ----------
    profiles : NDArray[np.floating]
        Array of observed profiles with shape (N, n_radial_bins).
    sigmas : NDArray[np.floating]
        Uncertainties for each radial bin.
    priors : PriorConfig
        Prior configuration dictionary.
    pool : multiprocessing.Pool, optional
        Pool for parallel execution.

    Returns
    -------
    tuple[NDArray[np.floating], emcee.EnsembleSampler]
        Flat chain and the sampler.
    """
    joint_payload = [np.concatenate((prof, sigmas)) for prof in profiles]
    sampler = run_joint_mcmc(joint_payload, priors, pool=pool)
    flat_chain = sampler.flatchain
    return flat_chain, sampler


def join_then_fit(
    profiles: NDArray[np.floating],
    sigmas: NDArray[np.floating],
    priors: PriorConfig,
    pool: Pool | None = None,
) -> tuple[NDArray[np.floating], emcee.EnsembleSampler]:
    """
    Join profiles by taking median then fit with MCMC.

    For a given set of profiles, we first find the median profile (join)
    to reduce noise and then run MCMC on that (fit).

    Parameters
    ----------
    profiles : NDArray[np.floating]
        Array of observed profiles with shape (N, n_radial_bins).
    sigmas : NDArray[np.floating]
        Uncertainties for each radial bin.
    priors : PriorConfig
        Prior configuration dictionary.
    pool : multiprocessing.Pool, optional
        Pool for parallel execution.

    Returns
    -------
    tuple[NDArray[np.floating], emcee.EnsembleSampler]
        Flat chain and the sampler.
    """
    avg_profile = np.median(profiles, axis=0)
    avg_profile = np.concatenate((avg_profile, sigmas))
    sampler = run_mcmc(avg_profile, priors, pool=pool)
    flat_chain = sampler.flatchain
    return flat_chain, sampler
