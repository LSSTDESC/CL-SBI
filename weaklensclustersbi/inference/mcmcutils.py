import emcee
import numpy as np
from context import wlprofile, populationutils, population
import scipy.stats


def logprior(params, priors):
    log10mass, concentration, log_f = params
    if not priors["min_log10mass"] < log10mass < priors["max_log10mass"]:
        return -np.inf
    if not priors["min_concentration"] < concentration < priors["max_concentration"]:
        return -np.inf
    # Arbitrary prior on log_f, but we can set a range if we want
    if not (-10 < log_f < 10):
        return -np.inf

    # Calculate the probability of the MCMC concentration based on m-c relation
    mc_scatter = priors["mc_scatter"]
    z = (priors["min_z"] + priors["max_z"]) / 2
    c_from_m = populationutils.get_concentration(
        log10mass, model=priors["mc_relation"], z=z
    )
    return scipy.stats.norm(loc=c_from_m, scale=mc_scatter).logpdf(concentration)


def loglike(params, priors, model):
    log10mass, concentration, log_f = params

    z = (priors["min_z"] + priors["max_z"]) / 2

    estimate = wlprofile.simulate_nfw(
        log10mass,
        concentration,
        z=z,
    )

    num_radial_bins = len(estimate)
    yerr = model[num_radial_bins : 2 * num_radial_bins]
    sigma2 = yerr**2 + (np.exp(log_f) * estimate) ** 2
    return -0.5 * np.sum(
        (estimate - model[:num_radial_bins]) ** 2 / sigma2 + np.log(sigma2)
    )


def logprob(params, priors, model):
    lp = logprior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, priors, model)
