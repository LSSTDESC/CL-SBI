"""
Core structures and tools for dealing with a simulated population of galaxy clusters that can be observed

Notes::

    [sample format] weaklensingclustersbi currently supports samples defined with a given mean richness mass relation, or mean concentration-mass relation

Copyright 2022-2023, LSST-DESC
"""
import numpy as np


def random_mass_conc(min_log10mass,
                     max_log10mass,
                     num_parameter_samples,
                     noise=0.2):
    '''
    In the provided log10mass range, randomly sample num_parameter_samples log10masses and find their
    corresponding concentrations, with some added random normal noise added.

    Args:
        min_log10mass: lower bound of log10masses in our sample before we add noise
        max_log10mass: upper bound of log10masses in our sample before we add noise
        num_parameter_samples : number of mc_pairs we want in our result
        noise : amount of random normal noise to add to our mc_pairs

    Returns:
        mc_pairs : a numpy array of (log10mass, concentration) tuples of size num_parameter_samples
    '''
    from .populationutils import get_concentration

    log10mass_sample = np.random.uniform(min_log10mass,
                                         max_log10mass,
                                         size=num_parameter_samples)
    non_noisy_concentration_sample = get_concentration(log10mass_sample)
    concentration_sample = np.random.normal(non_noisy_concentration_sample,
                                            noise, num_parameter_samples)
    return list(zip(log10mass_sample, concentration_sample))


def generate_concentration_for_sample(log10masses,
                                      z=0.0,
                                      mcrelation='child18',
                                      scatter_concentration_scale=0.2):
    '''
    From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift, 
    define the corresponding concentrations of this population assuming a mean concentration-mass relation 
    based on theoretical prediction) with some scatter in concentration at fixed mass. 

    Args:
        log10masses: a numpy array of masses, e.g. np.random.uniform(13,15,size=10000)
        z : redshift, default: 0.0
        mcrelation : string that is a key to the model name of the M-c relation from colossus, default: child18

    Returns:
        concentrations : a numpy array of concentration values

    '''

    from .populationutils import get_concentration

    non_noisy_concentrations = get_concentration(log10masses,
                                                 z=z,
                                                 model=mcrelation)
    # Note: May want to later generalize the distribution of concentration about the mean relation that is not random normal
    concentrations = np.random.normal(non_noisy_concentrations,
                                      scatter_concentration_scale,
                                      np.shape(log10masses))

    return concentrations


def generate_richness_for_sample(log10masses,
                                 z=0.0,
                                 rmrelation='murata17',
                                 scatter_richness_scale=1.0):
    '''
    From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift, 
    define the corresponding richness of this population assuming a mean richness-mass relation 
    based on theoretical prediction) with some scatter in richness at fixed mass. 

    Args:
        log10masses: a numpy array of masses, e.g. np.random.uniform(13,15,size=10000)
        z : redshift, default: 0.0
        rmrelation : string that is a key to the model name of the richness-mass relation from colossus, default: murata17

    Returns:
        richnesses : a numpy array of richness values

    '''
    from .populationutils import get_richness

    non_noisy_richnesses = np.array([
        get_richness(log10mass, z=z, model=rmrelation)
        for log10mass in log10masses
    ])
    richnesses = np.random.normal(non_noisy_richnesses, scatter_richness_scale,
                                  np.shape(log10masses))

    return richnesses


def draw_masses_in_richness_bin(
    lambda_min,
    lambda_max,
    rmrelation='murata17',
    num_sims=10,
    noise_dex=0,
):
    '''
    From a given richness bin (defined by the minimum and maximum lambda values), randomly draw a given number of masses and then
    apply some noise dex.

    Args:
        lambda_min: minimum richness (should be at least 20 if using murata17)
        lambda_max: maximum richness (should be at most 100 if using murata 17)
        rmrelation : string that is a key to the model name of the richness-mass relation from colossus, default: murata17
        num_sims: number of masses to be drawn. this will determine the size of the output
        noise_dex: how much noise to be applied

    Returns:
        log10masses: a numpy array of size num_sims of log10mass values
    '''

    from .populationutils import get_log10mass_from_richness, calculate_noise

    lambdas = np.random.rand(num_sims) * (lambda_max - lambda_min) + lambda_min
    log10masses = np.array(
        [get_log10mass_from_richness(lambda_) for lambda_ in lambdas])
    return calculate_noise(log10masses, noise_dex)


def gen_mc_pairs_in_richness_bin(
    lambda_min,
    lambda_max,
    rmrelation='murata17',
    num_sims=10,
    noise_dex=0,
):
    '''
    For a given richness bin, generate {num_sims} mass-concentration samples with some user-specified noise

    Args:
        lambda_min: minimum richness (should be at least 20 if using murata17)
        lambda_max: maximum richness (should be at most 100 if using murata 17)
        rmrelation : string that is a key to the model name of the richness-mass relation from colossus, default: murata17
        num_sims: number of masses to be drawn. this will determine the size of the output
        noise_dex: how much noise to be applied

    Returns:
        mc_pairs: a numpy array of size num_sims of tupes of (log10mass, concentration)
    '''

    log10mass_sample = draw_masses_in_richness_bin(
        lambda_min,
        lambda_max,
        rmrelation,
        num_sims,
        noise_dex,
    )
    concentration_sample = generate_concentration_for_sample(
        log10mass_sample, scatter_concentration_scale=0.2)
    mc_pairs = list(zip(log10mass_sample, concentration_sample))
    return mc_pairs
