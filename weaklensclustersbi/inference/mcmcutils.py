import emcee
import numpy as np


def logprior(params, priors):
    log10mass, concentration = params
    if not priors['min_log10mass'] < log10mass < priors['max_log10mass']:
        return -np.inf
    if not priors['min_concentration'] < concentration < priors[
            'max_concentration']:
        return -np.inf
    #TODO: how do we want to update this based on the priors passed in
    return np.log(1 / 20)


def loglike(params, model):
    log10mass, concentration = params

    from ..simulations.wlprofile import simulate_nfw
    estimate = simulate_nfw(log10mass=log10mass, concentration=concentration)
    # TODO: sanity check my error calculation
    error = np.std(log10mass)**2 + np.std(concentration)**2
    return -0.5 * np.sum((estimate - model)**2 / np.exp(2 * error) + 2 * error)


def logprob(params, priors, model):
    lp = logprior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, model)
