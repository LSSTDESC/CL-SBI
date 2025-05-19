import emcee
import numpy as np
from context import wlprofile, populationutils, population
import scipy.stats


def logprior(params, priors):
    log10mass, concentration = params
    if not priors["min_log10mass"] < log10mass < priors["max_log10mass"]:
        return -np.inf
    if not priors["min_concentration"] < concentration < priors["max_concentration"]:
        return -np.inf

    # Calculate the probability of the MCMC concentration based on m-c relation
    mc_scatter = priors["mc_scatter"]
    z = (priors["min_z"] + priors["max_z"]) / 2
    c_from_m = populationutils.get_concentration(
        log10mass, model=priors["mc_relation"], z=z
    )
    c_norm = scipy.stats.norm(c_from_m, mc_scatter).pdf(concentration)
    if (c_norm / 20) <= 0:
        return -np.inf

    # TODO: how do we want to update this based on the priors passed in
    return np.log(c_norm / 20)


def loglike(params, priors, model):
    log10mass, concentration = params

    z = (priors["min_z"] + priors["max_z"]) / 2

    estimate = wlprofile.simulate_nfw(log10mass, concentration, z=z)

    # TODO: clean this up
    if len(model) == 1:
        model = model[0]

    num_radial_bins = len(estimate)
    # Compute yerr for all radial bins at once
    yerr = model[num_radial_bins : 2 * num_radial_bins]
    # Compute the squared differences
    squared_diffs = (estimate - model[:num_radial_bins]) ** 2
    # Compute the log-likelihood terms
    ll_terms = -0.5 * (squared_diffs / yerr**2) + np.log(2 * np.pi * yerr**2)
    # Sum all terms to get the final log-likelihood
    return np.sum(ll_terms)


def logprob(params, priors, model):
    lp = logprior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, priors, model)
