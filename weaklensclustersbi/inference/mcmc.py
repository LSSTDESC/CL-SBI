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


def stacked_config() -> MCMCConfig:
    """
    Return a lighter MCMC configuration for stacked posterior runs.

    Uses fewer walkers and steps per individual fit to manage runtime
    when running many independent MCMCs.

    Returns
    -------
    MCMCConfig
        Dictionary containing nwalkers, npar, starts, nsteps_burn, nsteps_per_chain.
    """
    return {
        "nwalkers": 50,
        "npar": 3,
        "starts": np.array([14, 4, 0]),
        "nsteps_burn": 50,
        "nsteps_per_chain": 200,
    }


def _run_mcmc_single_light(
    profile_with_sigma: NDArray[np.floating],
    priors: PriorConfig,
) -> emcee.EnsembleSampler:
    """
    Run a single lightweight MCMC for one observation.

    Parameters
    ----------
    profile_with_sigma : NDArray[np.floating]
        Observed profile concatenated with uncertainties.
    priors : PriorConfig
        Prior configuration dictionary.

    Returns
    -------
    emcee.EnsembleSampler
        The sampler after running burn-in and production chains.
    """
    from .mcmcutils import logprob

    config = stacked_config()
    sampler = emcee.EnsembleSampler(
        config["nwalkers"],
        config["npar"],
        logprob,
        args=[priors, profile_with_sigma],
    )

    starts = config["starts"] + 5 * np.random.uniform(
        size=(config["nwalkers"], config["npar"])
    )

    # burn-in (silent)
    pos, prob, stat = sampler.run_mcmc(starts, config["nsteps_burn"], progress=False)
    sampler.reset()

    # production (silent)
    sampler.run_mcmc(pos, config["nsteps_per_chain"], progress=False)

    return sampler


def fit_then_join_stacked(
    profiles: NDArray[np.floating],
    sigmas: NDArray[np.floating],
    priors: PriorConfig,
    pool: Pool | None = None,
    n_population_samples: int = 50000,
) -> tuple[NDArray[np.floating], list[emcee.EnsembleSampler], dict[str, Any]]:
    """
    Two-stage population inference: individual MCMCs then fit population distribution.

    Stage 1: Run N independent MCMCs (one per observation) to get individual posteriors.
    Stage 2: Extract MAP estimates from each posterior and fit a 2D Gaussian to
    infer the underlying population distribution of (M,c).

    This approach separates measurement uncertainty from population spread,
    giving an estimate of the true underlying population distribution.

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
    n_population_samples : int, optional
        Number of samples to draw from fitted population distribution.

    Returns
    -------
    tuple[NDArray[np.floating], list[emcee.EnsembleSampler], dict]
        - Samples from the fitted population distribution (for plotting)
        - List of individual samplers (for diagnostics)
        - Dictionary with fitted population parameters:
          {'mu': [mu_M, mu_c], 'cov': 2x2 covariance, 'map_estimates': Nx2 array}
    """
    n_obs = len(profiles)
    print(f"## Running two-stage FTJ MCMC for {n_obs} observations...")
    print("## Stage 1: Running individual MCMCs...")

    # Prepare payloads for each observation
    payloads = [(np.concatenate((prof, sigmas)), priors) for prof in profiles]

    samplers = []
    if pool is not None:
        # Parallel execution using starmap
        samplers = pool.starmap(_run_mcmc_single_light, payloads)
    else:
        # Sequential execution
        for i, (payload, prior) in enumerate(payloads):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"   Processing observation {i + 1}/{n_obs}")
            sampler = _run_mcmc_single_light(payload, prior)
            samplers.append(sampler)

    # Stage 2: Extract MAP estimates and fit population distribution
    print("## Stage 2: Fitting population distribution to MAP estimates...")
    map_estimates = []
    for sampler in samplers:
        log_probs = sampler.get_log_prob(flat=True)
        map_idx = np.argmax(log_probs)
        # Only take M and c (indices 0 and 1), not log_f
        map_mc = sampler.flatchain[map_idx, :2]
        map_estimates.append(map_mc)

    map_estimates = np.array(map_estimates)  # Shape: (n_obs, 2)

    # Fit 2D Gaussian to MAP estimates
    mu = np.mean(map_estimates, axis=0)
    cov = np.cov(map_estimates.T)

    print(f"## Fitted population: μ_M={mu[0]:.3f}, μ_c={mu[1]:.3f}")
    print(f"## Fitted population: σ_M={np.sqrt(cov[0,0]):.3f}, σ_c={np.sqrt(cov[1,1]):.3f}")
    print(f"## Fitted population: ρ={cov[0,1]/(np.sqrt(cov[0,0]*cov[1,1])):.3f}")

    # Sample from fitted population distribution
    population_samples = np.random.multivariate_normal(mu, cov, size=n_population_samples)

    # Store fitted parameters
    population_params = {
        'mu': mu,
        'cov': cov,
        'map_estimates': map_estimates,
        'sigma_M': np.sqrt(cov[0, 0]),
        'sigma_c': np.sqrt(cov[1, 1]),
        'rho': cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1])),
    }

    print(f"## Population samples shape: {population_samples.shape}")
    return population_samples, samplers, population_params


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
