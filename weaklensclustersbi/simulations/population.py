"""
Core structures and tools for dealing with a simulated population of galaxy clusters that can be observed

Notes::

    [sample format] weaklensingclustersbi currently supports samples defined with a given mean richness mass relation, or mean concentration-mass relation

Copyright 2022-2023, LSST-DESC
"""

import numpy as np
from scipy.stats import halfnorm
from colossus.cosmology import cosmology
from colossus.lss import mass_function


def random_mass_conc(
    min_log10mass,
    max_log10mass,
    num_sims,
    mc_scatter=0,
    mc_relation="child18",
    min_z=0,
    max_z=0,
):
    """
    In the provided log10mass range, randomly sample num_sims log10masses and find their
    corresponding concentrations, with some added random normal noise added.

    Args:
        min_log10mass: lower bound of log10masses in our sample before we add noise
        max_log10mass: upper bound of log10masses in our sample before we add noise
        num_sims : number of mc_pairs we want in our result
        mc_scatter : amount of random normal noise to add to our mc_pairs

    Returns:
        mc_pairs : a numpy array of (log10mass, concentration) tuples of size num_sims
    """
    from colossus.halo import concentration as conc_module

    log10mass_sample = np.random.uniform(min_log10mass, max_log10mass, size=num_sims)
    z_sample = np.random.uniform(min_z, max_z, size=num_sims)

    # Vectorized with np.vectorize to handle per-sample z values
    conc_vec = np.vectorize(
        lambda m, z: conc_module.concentration(10**m, "vir", z, model=mc_relation)
    )
    non_noisy_concentration_sample = conc_vec(log10mass_sample, z_sample)

    concentration_sample = np.random.normal(
        non_noisy_concentration_sample, mc_scatter, num_sims
    )
    return list(zip(log10mass_sample, concentration_sample))


def _generate_property_for_sample(log10masses, property_fn, scatter):
    """
    Generic function to generate a property for a sample of masses with scatter.

    Parameters
    ----------
    log10masses : array-like
        Array of log10 masses.
    property_fn : callable
        Function that takes log10mass (or array) and returns the property value(s).
    scatter : float
        Standard deviation of Gaussian scatter to apply.

    Returns
    -------
    np.ndarray
        Property values with scatter applied.
    """
    # Vectorized: assumes property_fn supports array input (e.g., get_richness)
    log10masses = np.asarray(log10masses)
    non_noisy_values = property_fn(log10masses)
    return np.random.normal(non_noisy_values, scatter, np.shape(log10masses))


def generate_concentration_for_sample(
    log10masses, zs, mc_relation="child18", mc_scatter=0
):
    """
    From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift,
    define the corresponding concentrations of this population assuming a mean concentration-mass relation
    based on theoretical prediction) with some scatter in concentration at fixed mass.

    Args:
        log10masses: a numpy array of masses, e.g. np.random.uniform(13,15,size=10000)
        zs : array of redshifts corresponding to each mass
        mc_relation : string that is a key to the model name of the M-c relation from colossus, default: child18
        mc_scatter : scatter in concentration at fixed mass

    Returns:
        concentrations : a numpy array of concentration values

    """
    from colossus.halo import concentration as conc_module

    log10masses = np.asarray(log10masses)
    zs = np.asarray(zs)

    # Vectorized with np.vectorize to handle per-sample z values
    conc_vec = np.vectorize(
        lambda m, z: conc_module.concentration(10**m, "vir", z, model=mc_relation)
    )
    non_noisy_concentrations = conc_vec(log10masses, zs)
    return np.random.normal(non_noisy_concentrations, mc_scatter, np.shape(log10masses))


def generate_richness_for_sample(
    log10masses, z=0.0, rm_relation="murata17", rm_scatter=0
):
    """
    From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift,
    define the corresponding richness of this population assuming a mean richness-mass relation
    based on theoretical prediction) with some scatter in richness at fixed mass.

    Args:
        log10masses: a numpy array of masses, e.g. np.random.uniform(13,15,size=10000)
        z : redshift, default: 0.0
        rm_relation : string that is a key to the model name of the richness-mass relation from colossus, default: murata17
        rm_scatter : scatter in richness at fixed mass

    Returns:
        richnesses : a numpy array of richness values

    """
    from .populationutils import get_richness

    return _generate_property_for_sample(
        log10masses, lambda m: get_richness(m, z=z, model=rm_relation), rm_scatter
    )


def draw_masses_in_richness_bin(
    lambda_min,
    lambda_max,
    rm_relation="murata17",
    num_samples=10,
    rm_scatter=0,
    # return_pairs=False,
    # mass function params
    mdef="200m",
    model="tinker08",
    cosmo="planck18",
    z=0.5,
):
    """
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
    """

    from .populationutils import get_log10mass_from_richness

    grid_size = 1000
    lambdas = np.random.rand(grid_size) * (lambda_max - lambda_min) + lambda_min
    # Vectorized: get_log10mass_from_richness uses numpy ops that accept arrays
    log10masses = get_log10mass_from_richness(lambdas, model=rm_relation, z=z)
    log10masses = np.random.normal(log10masses, rm_scatter, np.shape(lambdas))

    cosmo = cosmology.setCosmology(cosmo)

    # mass func weighting: "flat" = uniform, otherwise use specified model (e.g., "tinker08")
    if model == "flat":
        # Uniform weighting - no mass function applied
        pdf = np.ones(grid_size)
    else:
        # mass func in units of dn/dlnM
        dndlnM = mass_function.massFunction(
            10**log10masses, z, mdef=mdef, model=model, q_out="dndlnM"
        )
        # probability density function
        pdf = dndlnM * np.log(10)

    pdf /= pdf.sum()

    # pick num_obs indices with replacement from the PDF
    idx = np.random.choice(grid_size, size=num_samples, replace=True, p=pdf)

    return lambdas[idx], log10masses[idx]


def gen_mc_pairs_in_richness_bin(
    lambda_min,
    lambda_max,
    rm_relation="murata17",
    mc_relation="child18",
    num_samples=10,
    mc_scatter=0,
    rm_scatter=0,
    min_z=0,
    max_z=0,
    # mass function params
    mdef="200m",
    model="tinker08",
    cosmo="planck18",
    # richness contam experiment params
    richness_contam_frac=0,
    lambda_min_contam=None,
):
    """
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
    """
    contam_size = int(num_samples * richness_contam_frac)
    non_contam_size = num_samples - contam_size

    _, log10mass_sample = draw_masses_in_richness_bin(
        lambda_min,
        lambda_max,
        rm_relation=rm_relation,
        num_samples=non_contam_size,
        rm_scatter=rm_scatter,
        z=(min_z + max_z) / 2,
        mdef=mdef,
        model=model,
        cosmo=cosmo,
    )
    z_sample = np.random.uniform(min_z, max_z, size=non_contam_size)
    concentration_sample = generate_concentration_for_sample(
        log10mass_sample,
        mc_scatter=mc_scatter,
        mc_relation=mc_relation,
        zs=z_sample,
    )
    mc_pairs = list(zip(log10mass_sample, concentration_sample))

    # Allow for some observations to "contam" up from a lower richness bin
    if contam_size != 0 and lambda_min_contam is not None:
        _, log10mass_contam_sample = draw_masses_in_richness_bin(
            lambda_min_contam,
            lambda_min,
            rm_relation=rm_relation,
            num_samples=contam_size,
            rm_scatter=rm_scatter,
            z=(min_z + max_z) / 2,
            mdef=mdef,
            model=model,
            cosmo=cosmo,
        )
        z_contam_sample = np.random.uniform(min_z, max_z, size=contam_size)
        concentration_contam_sample = generate_concentration_for_sample(
            log10mass_contam_sample,
            mc_scatter=mc_scatter,
            mc_relation=mc_relation,
            zs=z_contam_sample,
        )
        mc_pairs_contam = list(
            zip(log10mass_contam_sample, concentration_contam_sample)
        )
        mc_pairs.extend(mc_pairs_contam)
    return mc_pairs


def filter_mc_pairs(mc_pairs, criteria="all"):
    """
    Adding subselection criteria that we may want to use to filter the simulated mass concentration pairs
    that are used in SBI.

    Args:
            mc_pairs: unfiltered mc_pairs from simulations
            criteria: string specifying subselection criteria

        Returns:
            mc_pairs: filtered mc_pairs
    """

    # TODO: what other criteria will we want to filter by?
    if criteria == "all":
        return mc_pairs


# Calculate noise with a fixed dex value
def calculate_noise(sample, dex=0.0):
    random_noise = np.random.normal(0, dex, np.shape(sample))
    return sample * 10 ** (random_noise)


# Calculate noise with a range of dex values between 0 and max_dex
def calculate_noise_range(sample, max_dex=0.0):
    shape = np.shape(sample)
    dex = np.random.uniform(0, max_dex, size=shape)
    random_noise = np.random.normal(0, dex, shape)
    return sample * 10 ** (random_noise)


def gen_error_bars(nfw_profile, dex=0.0):
    shape = np.shape(nfw_profile)
    upper_error = halfnorm.rvs(loc=0, scale=dex, size=shape)

    # Add some dex to generate our upper error
    upper_error_obs = nfw_profile * 10 ** (upper_error)

    # Subtract some dex to generate our lower error
    lower_error = halfnorm.rvs(loc=0, scale=dex, size=shape)
    lower_error_obs = nfw_profile * 10 ** (-lower_error)

    # return lower_error_obs, nfw_profile, upper_error_obs

    return np.concatenate([lower_error_obs, nfw_profile, upper_error_obs], axis=1)
