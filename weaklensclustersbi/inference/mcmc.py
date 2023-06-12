import numpy as np
import emcee

# starting points for the chains
np.random.seed(2807)


# TODO: use this if MCMC params not passed in?
def default_config():
    return {
        'nwalkers': 100,
        'npar': 2,
        'starts': np.array([10, 10]),
        'nsteps_burn': 500,
        'nsteps_per_chain': 2000,
    }


def run_mcmc(truth_val, config):
    from .mcmcutils import logprob
    # set up the sampler
    sampler = emcee.EnsembleSampler(config['nwalkers'],
                                    config['npar'],
                                    logprob,
                                    args=[truth_val])

    # Add some noise to starting positions for walkers
    starts = config['starts'] + 0.1 * np.random.randn(config['nwalkers'],
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


def fit_then_join(profiles, config):
    from .mcmcutils import logprob
    chains = []
    for profile in profiles:
        sampler = run_mcmc(profile, config)
        chains.append(sampler.flatchain)
    return np.vstack(chains)


def join_then_fit(profiles, config):
    from .mcmcutils import logprob
    mean_profile = np.mean(profiles, keepdims=True, axis=0)
    sampler = run_mcmc(mean_profile, config)
    return sampler.flatchain