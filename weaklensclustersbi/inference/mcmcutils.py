import emcee
import numpy as np


def logprior(params):
    log10mass, concentration = params
    if not 0 < log10mass < 20:
        return -np.inf
    if not 0 < concentration < 20:
        return -np.inf
    return np.log(1 / 20)


def loglike(params, model):
    log10mass, concentration = params

    from ..simulations.wlprofile import simulate_nfw
    estimate = simulate_nfw(log10mass=log10mass, concentration=concentration)
    # TODO: sanity check my error calculation
    error = np.std(log10mass**2) + np.std(concentration)**2
    return -0.5 * np.sum((estimate - model)**2 / np.exp(2 * error) + 2 * error)


def logprob(params, model):
    lp = logprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, model)
