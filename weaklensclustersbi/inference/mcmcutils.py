import numpy as np
import scipy.stats
from colossus.cosmology import cosmology
from colossus.lss import mass_function

from weaklensclustersbi.simulations import wlprofile, populationutils


def logprior(params, priors):
    """
    Log prior for MCMC that matches SBI's implicit prior.

    Includes:
    - Mass function prior: P(M) ∝ dn/dlnM
    - Richness selection: P(M | λ_min, λ_max) via r-m relation
    - M-c relation prior on concentration
    """
    log10mass, concentration, log_f = params

    # Check for NaN inputs
    if not np.isfinite(log10mass) or not np.isfinite(concentration) or not np.isfinite(log_f):
        return -np.inf

    # Hard bounds
    if not priors["min_log10mass"] < log10mass < priors["max_log10mass"]:
        return -np.inf
    if not priors["min_concentration"] < concentration < priors["max_concentration"]:
        return -np.inf
    if not (-10 < log_f < 10):
        return -np.inf

    z = (priors["min_z"] + priors["max_z"]) / 2
    logp = 0.0

    # 1. Mass function prior (if enabled)
    if priors.get("use_mass_function", False):
        mf_model = priors.get("mass_function_model", "tinker08")
        mdef = priors.get("mdef", "200m")
        cosmology.setCosmology(priors.get("cosmo", "planck18"))

        # dn/dlnM - convert to log prior
        dndlnM = mass_function.massFunction(
            10**log10mass, z, mdef=mdef, model=mf_model, q_out="dndlnM"
        )
        if dndlnM > 0:
            logp += np.log(dndlnM)
        else:
            return -np.inf

    # 2. Richness selection prior (if richness bin specified)
    if "min_richness" in priors and "max_richness" in priors:
        rm_relation = priors.get("rm_relation", "mcclintock18")
        rm_scatter = priors.get("rm_scatter", 0.1)

        # Expected richness for this mass
        lambda_expected = populationutils.get_richness(log10mass, z=z, model=rm_relation)

        # P(λ ∈ [λ_min, λ_max] | M) assuming log-normal scatter
        # Scatter is in log10(mass) at fixed richness, so we need inverse
        # Approximate: use scatter on log(lambda) ~ rm_scatter (similar magnitude)
        log_lambda_exp = np.log(lambda_expected)
        log_lambda_min = np.log(priors["min_richness"])
        log_lambda_max = np.log(priors["max_richness"])

        # CDF of normal distribution
        p_in_bin = (
            scipy.stats.norm.cdf(log_lambda_max, log_lambda_exp, rm_scatter) -
            scipy.stats.norm.cdf(log_lambda_min, log_lambda_exp, rm_scatter)
        )

        if p_in_bin > 1e-10:
            logp += np.log(p_in_bin)
        else:
            return -np.inf

    # 3. M-c relation prior on concentration
    mc_scatter = priors["mc_scatter"]
    c_from_m = populationutils.get_concentration(
        log10mass, model=priors["mc_relation"], z=z
    )
    if not np.isfinite(c_from_m):
        return -np.inf
    logp += scipy.stats.norm(loc=c_from_m, scale=mc_scatter).logpdf(concentration)

    # Final NaN check
    if not np.isfinite(logp):
        return -np.inf

    return logp


def logprior_flat(params, priors):
    """
    Original flat prior for comparison/backwards compatibility.
    Only uses M-c relation, no mass function or richness selection.
    """
    log10mass, concentration, log_f = params

    # Check for NaN inputs
    if not np.isfinite(log10mass) or not np.isfinite(concentration) or not np.isfinite(log_f):
        return -np.inf

    if not priors["min_log10mass"] < log10mass < priors["max_log10mass"]:
        return -np.inf
    if not priors["min_concentration"] < concentration < priors["max_concentration"]:
        return -np.inf
    if not (-10 < log_f < 10):
        return -np.inf

    z = (priors["min_z"] + priors["max_z"]) / 2
    mc_scatter = priors["mc_scatter"]
    c_from_m = populationutils.get_concentration(
        log10mass, model=priors["mc_relation"], z=z
    )
    if not np.isfinite(c_from_m):
        return -np.inf
    logp = scipy.stats.norm(loc=c_from_m, scale=mc_scatter).logpdf(concentration)
    return logp if np.isfinite(logp) else -np.inf


def loglike(params, priors, model):
    """Log likelihood for NFW profile fit."""
    log10mass, concentration, log_f = params

    z = (priors["min_z"] + priors["max_z"]) / 2

    estimate = np.log10(
        wlprofile.simulate_nfw(
            log10mass,
            concentration,
            z=z,
        )
    )

    num_radial_bins = len(estimate)
    yerr = model[num_radial_bins : 2 * num_radial_bins]
    sigma2 = yerr**2 + (np.exp(log_f) * estimate) ** 2
    return -0.5 * np.sum(
        (estimate - model[:num_radial_bins]) ** 2 / sigma2 + np.log(sigma2)
    )


def logprob(params, priors, model):
    """Log posterior using informed prior (mass function + richness selection)."""
    lp = logprior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, priors, model)


def logprob_flat(params, priors, model):
    """Log posterior using flat prior (for comparison)."""
    lp = logprior_flat(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, priors, model)


def joint_logprob(params, priors, models):
    """Joint log posterior for multiple observations (informed prior)."""
    lp = logprior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = 0.0
    for model in models:
        ll += loglike(params, priors, model) / len(models)
    return lp + ll


def joint_logprob_flat(params, priors, models):
    """Joint log posterior for multiple observations (flat prior)."""
    lp = logprior_flat(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = 0.0
    for model in models:
        ll += loglike(params, priors, model) / len(models)
    return lp + ll
