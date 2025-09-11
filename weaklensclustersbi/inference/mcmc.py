import numpy as np
import emcee

# import multiprocessing

# starting points for the chains
np.random.seed(2807)


def default_config():
    return {
        "nwalkers": 100,
        # "npar": 2,
        # "starts": np.array([15, 5]),
        "npar": 3,
        "starts": np.array([14, 4, 0]),
        "nsteps_burn": 100,
        "nsteps_per_chain": 500,
    }


def run_mcmc(truth_val, priors, pool=None):
    from .mcmcutils import logprob

    # set up the sampler
    config = default_config()
    sampler = emcee.EnsembleSampler(
        config["nwalkers"], config["npar"], logprob, args=[priors, truth_val], pool=pool
    )

    # Add some noise to starting positions for walkers
    # TODO: randomly sample within the priors
    starts = config["starts"] + 5 * np.random.uniform(
        size=(config["nwalkers"], config["npar"])
    )

    # burn-in
    print("## burning in ... ")
    pos, prob, stat = sampler.run_mcmc(
        starts, config["nsteps_burn"]
    )  # , progress=True)

    # reset the sampler
    sampler.reset()

    # run the full chain
    print("## running the full chain ... ")
    sampler.run_mcmc(pos, config["nsteps_per_chain"])  # , progress=True)

    return sampler


def run_joint_mcmc(profiles, priors, pool=None):
    from .mcmcutils import joint_logprob

    # set up the sampler
    config = default_config()
    sampler = emcee.EnsembleSampler(
        config["nwalkers"],
        config["npar"],
        joint_logprob,
        args=[priors, profiles],
        pool=pool,
    )

    # Add some noise to starting positions for walkers
    # TODO: randomly sample within the priors
    starts = config["starts"] + 5 * np.random.uniform(
        size=(config["nwalkers"], config["npar"])
    )

    # burn-in
    print("## burning in ... ")
    pos, prob, stat = sampler.run_mcmc(
        starts, config["nsteps_burn"]
    )  # , progress=True)

    # reset the sampler
    sampler.reset()

    # run the full chain
    print("## running the full chain ... ")
    sampler.run_mcmc(pos, config["nsteps_per_chain"])  # , progress=True)

    return sampler


def fit_then_join(profiles, sigmas, priors, pool=None):
    """
    For a given set of profiles, we run MCMC on each of them (fit). To reduce noise, at the end, we stack
    all of the chains into a single one (join).
    """
    # chains = []
    # samplers = []
    # for profile in profiles:
    #     profile = np.concatenate((profile, sigmas))
    #     sampler = run_mcmc(profile, priors, pool=pool)
    #     samplers.append(sampler)
    #     # tau = sampler.get_autocorr_time()
    #     # print(f"Autocorrelation time for fit-then-join chain: {tau}")
    #     # flat_chain = sampler.get_chain(thin=int(tau[0] / 2), flat=True)
    #     # flat_chain = sampler.get_chain(thin=25, flat=True)
    #     # chains.append(flat_chain)
    #     chains.append(sampler.flatchain)
    # return np.vstack(chains), samplers

    joint_payload = [np.concatenate((prof, sigmas)) for prof in profiles]
    sampler = run_joint_mcmc(joint_payload, priors, pool=pool)
    flat_chain = sampler.flatchain
    return flat_chain, sampler


def join_then_fit(profiles, sigmas, priors, pool=None):
    """
    For a given set of profiles, we first find the median profile (join) to reduce noise and then
    run MCMC on that (fit).
    """
    # normalize sigmas based on number of profiles
    # sigmas = sigmas / np.sqrt(len(profiles))

    avg_profile = np.median(profiles, axis=0)
    avg_profile = np.concatenate((avg_profile, sigmas))
    sampler = run_mcmc(avg_profile, priors, pool=pool)
    # tau = sampler.get_autocorr_time()
    # print(f"Autocorrelation time for join-then-fit chain: {tau}")
    # flat_chain = sampler.get_chain(thin=int(tau[0] / 2), flat=True)
    # flat_chain = sampler.get_chain(thin=25, flat=True)
    flat_chain = sampler.flatchain
    return flat_chain, sampler
