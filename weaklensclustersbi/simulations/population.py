"""
Core structures and tools for dealing with a simulated population of galaxy clusters that can be observed

Notes::

    [sample format] weaklensingclustersbi currently supports samples defined with a given mean richness mass relation, or mean concentration-mass relation

Copyright 2022-2023, LSST-DESC
"""
import numpy as np


def random_mass_conc(
    min_log10mass,
    max_log10mass,
    num_sims,
    mc_scatter=0,
    mc_relation='murata17',
    z=0,
):
    '''
    In the provided log10mass range, randomly sample num_sims log10masses and find their
    corresponding concentrations, with some added random normal noise added.

    Args:
        min_log10mass: lower bound of log10masses in our sample before we add noise
        max_log10mass: upper bound of log10masses in our sample before we add noise
        num_sims : number of mc_pairs we want in our result
        mc_scatter : amount of random normal noise to add to our mc_pairs

    Returns:
        mc_pairs : a numpy array of (log10mass, concentration) tuples of size num_sims
    '''
    from .populationutils import get_concentration

    log10mass_sample = np.random.uniform(min_log10mass,
                                         max_log10mass,
                                         size=num_sims)
    non_noisy_concentration_sample = get_concentration(
        log10mass_sample,
        model=mc_relation,
        z=z,
    )
    concentration_sample = np.random.normal(non_noisy_concentration_sample,
                                            mc_scatter, num_sims)
    return list(zip(log10mass_sample, concentration_sample))


def generate_concentration_for_sample(log10masses,
                                      z=0.0,
                                      mc_relation='child18',
                                      mc_scatter=0):
    '''
    From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift, 
    define the corresponding concentrations of this population assuming a mean concentration-mass relation 
    based on theoretical prediction) with some scatter in concentration at fixed mass. 

    Args:
        log10masses: a numpy array of masses, e.g. np.random.uniform(13,15,size=10000)
        z : redshift, default: 0.0
        mc_relation : string that is a key to the model name of the M-c relation from colossus, default: child18

    Returns:
        concentrations : a numpy array of concentration values

    '''

    from .populationutils import get_concentration

    non_noisy_concentrations = get_concentration(log10masses,
                                                 z=z,
                                                 model=mc_relation)
    # Note: May want to later generalize the distribution of concentration about the mean relation that is not random normal
    concentrations = np.random.normal(non_noisy_concentrations, mc_scatter,
                                      np.shape(log10masses))

    return concentrations


def generate_richness_for_sample(log10masses,
                                 z=0.0,
                                 rm_relation='murata17',
                                 rm_scatter=0):
    '''
    From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift, 
    define the corresponding richness of this population assuming a mean richness-mass relation 
    based on theoretical prediction) with some scatter in richness at fixed mass. 

    Args:
        log10masses: a numpy array of masses, e.g. np.random.uniform(13,15,size=10000)
        z : redshift, default: 0.0
        rm_relation : string that is a key to the model name of the richness-mass relation from colossus, default: murata17

    Returns:
        richnesses : a numpy array of richness values

    '''
    from .populationutils import get_richness

    non_noisy_richnesses = np.array([
        get_richness(log10mass, z=z, model=rm_relation)
        for log10mass in log10masses
    ])
    richnesses = np.random.normal(non_noisy_richnesses, rm_scatter,
                                  np.shape(log10masses))

    return richnesses


def draw_masses_in_richness_bin(
    lambda_min,
    lambda_max,
    rm_relation='murata17',
    num_obs=10,
    rm_scatter=0,
):
    '''
    From a given richness bin (defined by the minimum and maximum lambda values), randomly draw a given number of masses and then
    apply some scatter to the rm relation.

    Args:
        lambda_min: minimum richness (should be at least 20 if using murata17)
        lambda_max: maximum richness (should be at most 100 if using murata 17)
        rm_relation : string that is a key to the model name of the richness-mass relation from colossus, default: murata17
        num_obs: number of masses to be drawn. this will determine the size of the output
        rm_scatter: how much scatter to be applied to the rm relation

    Returns:
        log10masses: a numpy array of size num_obs of log10mass values
    '''

    from .populationutils import get_log10mass_from_richness

    lambdas = np.random.rand(num_obs) * (lambda_max - lambda_min) + lambda_min
    non_noisy_log10masses = np.array([
        get_log10mass_from_richness(lambda_, model=rm_relation)
        for lambda_ in lambdas
    ])
    log10masses = np.random.normal(non_noisy_log10masses, rm_scatter,
                                   np.shape(lambdas))
    return log10masses


def gen_mc_pairs_in_richness_bin(
    lambda_min,
    lambda_max,
    rm_relation='murata17',
    mc_relation='child18',
    num_obs=10,
    mc_scatter=0,
    rm_scatter=0,
    z=0,
):
    '''
    For a given richness bin, generate {num_obs} mass-concentration samples with some user-specified noise

    Args:
        lambda_min: minimum richness (should be at least 20 if using murata17)
        lambda_max: maximum richness (should be at most 100 if using murata 17)
        rm_relation : string that is a key to the model name of the richness-mass relation from colossus, default: murata17
        num_obs: number of masses to be drawn. this will determine the size of the output
        mc_scatter: how much scatter to be applied to the mc relation
        rm_scatter: how much scatter to be applied to the rm relation
        z: redshift
    Returns:
        mc_pairs: a numpy array of size num_obs of tupes of (log10mass, concentration)
    '''

    log10mass_sample = draw_masses_in_richness_bin(
        lambda_min,
        lambda_max,
        rm_relation=rm_relation,
        num_obs=num_obs,
        rm_scatter=rm_scatter,
    )
    concentration_sample = generate_concentration_for_sample(
        log10mass_sample,
        mc_scatter=mc_scatter,
        mc_relation=mc_relation,
        z=z,
    )
    mc_pairs = list(zip(log10mass_sample, concentration_sample))
    return mc_pairs


def filter_mc_pairs(mc_pairs, criteria='all'):
    '''
    Adding subselection criteria that we may want to use to filter the simulated mass concentration pairs
    that are used in SBI.

    Args:
            mc_pairs: unfiltered mc_pairs from simulations
            criteria: string specifying subselection criteria

        Returns:
            mc_pairs: filtered mc_pairs
    '''

    # TODO: what other criteria will we want to filter by?
    if criteria == 'all':
        return mc_pairs


def calculate_noise(sample, dex=0.0):
    random_noise = np.random.normal(1, dex, np.shape(sample))
    return sample * 10**(random_noise)
