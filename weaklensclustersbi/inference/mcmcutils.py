import emcee
import numpy as np
from context import wlprofile


def logprior(params, priors):
    log10mass, concentration, yerr = params
    if not priors['min_log10mass'] < log10mass < priors['max_log10mass']:
        return -np.inf
    if not priors['min_concentration'] < concentration < priors[
            'max_concentration']:
        return -np.inf
    #TODO: how do we want to update this based on the priors passed in
    return np.log(1 / 20)


def loglike(params, model):
    log10mass, concentration, yerr = params

    # TODO: add redshift to below function call?
    estimate = wlprofile.simulate_nfw(log10mass, concentration)
    return -0.5 * np.sum((estimate - model)**2 /
                         (yerr**2) + np.log(2 * np.pi * yerr**2))


def logprob(params, priors, model):
    lp = logprior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, model)
