import numpy as np
import emcee

# starting points for the chains
np.random.seed(2807)


def default_config():
    return {
        'nwalkers': 100,
        'npar': 2,
        'starts': np.array([10, 10]),
        'nsteps_burn': 500,
        'nsteps_per_chain': 2000,
    }


def run_mcmc(truth_val, priors):
    from .mcmcutils import logprob
    # set up the sampler
    config = default_config()
    sampler = emcee.EnsembleSampler(config['nwalkers'],
                                    config['npar'],
                                    logprob,
                                    args=[priors, truth_val])

    # Add some noise to starting positions for walkers
    # TODO: randomly sample within the priors
    starts = config['starts'] + 5 * np.random.randn(config['nwalkers'],
                                                    config['npar'])

    # burn-in
    print('## burning in ... ')
    pos, prob, stat = sampler.run_mcmc(starts,
                                       config['nsteps_burn'],
                                       progress=True)

    # reset the sampler
    sampler.reset()

    # run the full chain
    print('## running the full chain ... ')
    sampler.run_mcmc(pos, config['nsteps_per_chain'], progress=True)

    return sampler


def fit_then_join(profiles, priors):
    '''
    For a given set of profiles, we run MCMC on each of them (fit). To reduce noise, at the end, we stack
    all of the chains into a single one (join).
    '''
    chains = []
    samplers = []
    for profile in profiles:
        sampler = run_mcmc(profile, priors)
        samplers.append(sampler)
        chains.append(sampler.flatchain)
    return np.vstack(chains), samplers


def join_then_fit(profiles, priors):
    '''
    For a given set of profiles, we first find the average profile (join) to reduce noise and then 
    run MCMC on that (fit).
    '''
    mean_profile = np.mean(profiles, keepdims=True, axis=0)
    sampler = run_mcmc(mean_profile, priors)
    return sampler.flatchain, sampler
