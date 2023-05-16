import numpy as np
import emcee

# starting points for the chains
np.random.seed(2807)


def mcmc_config():
    return {
        'nwalkers': 100,
        'npar': 2,
        'starts': np.array([10, 10]),
        'nsteps_burn': 500,
        'nsteps_per_chain': 2000,
    }


def run_mcmc(truth_val):
    from .mcmcutils import logprob
    config = mcmc_config()
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


def fit_then_join(profiles):
    from .mcmcutils import logprob
    config = mcmc_config()
    chains = []
    for profile in profiles:
        sampler = run_mcmc(profile)
        chains.append(sampler.flatchain)
    return np.vstack(chains)


def join_then_fit(profiles):
    from .mcmcutils import logprob
    mean_profile = np.mean(profiles, keepdims=True, axis=0)
    config = mcmc_config()
    sampler = run_mcmc(mean_profile)
    return sampler.flatchain