"""
Utilities for defining and working with a sample of simulated clusters

Copyright 2022-2023, LSST-DESC
"""
from colossus.halo import concentration
import numpy as np
import warnings


def get_concentration(log10mass, mdef='vir', z=0.0, model='child18'):
    '''
    Return the concentration for a halo of a given mass assuming a model of child18
    at the present day (redshift 0.0), defined with respect to the virial radius. 
    '''

    return concentration.concentration(10**log10mass, mdef, z, model=model)


def get_richness(log10mass, z=0.0, model='murata17', strict=False):
    '''
    Return the optical richness for a halo of a given mass assuming the model of murata17. Note that this relation only holds if the lambda is between 20 and 100.
    '''
    if model == 'murata17':
        a, b, mass_pivot = _get_murata2017_parameters()
        lambda_ = np.exp(a + b * np.log((10**log10mass) / mass_pivot))

        if strict and (lambda_ < 20 or lambda_ > 100):
            warnings.warn(
                f'calculated lambda value ({lambda_}) from the given log10mass ({log10mass}) is outside the range supported by the murata17 relation (20-100)'
            )

        return lambda_
    return 0


def get_log10mass_from_richness(lambda_,
                                z=0.0,
                                model='murata17',
                                strict=False):
    '''
    Return the mass for a halo of a given optical richness assuming the model of murata17. Note that this relation only holds if the lambda is between 20 and 100.
    '''
    if model == 'murata17':
        if strict and (lambda_ < 20 or lambda_ > 100):
            warnings.warn(
                f'provided lambda value ({lambda_}) is outside the range supported by the murata17 relation (20-100)'
            )

        a, b, mass_pivot = _get_murata2017_parameters()
        return np.log10(np.exp((np.log(lambda_) - a) / b) * mass_pivot)
    return 0


def _get_murata2017_parameters():
    '''
    Fetching parameters that are used in the murata17 richness-mass relation
    '''
    a = 3.207
    b = 0.993
    mass_pivot = 3. * 10**14
    return a, b, mass_pivot
