"""
Utilities for defining and working with a sample of simulated clusters

Copyright 2022-2023, LSST-DESC
"""
from colossus.halo import concentration
import numpy as np


def get_concentration(log10mass, mdef='vir', z=0.0, model='child18'):
    '''
  Return the concentration for a halo of a given mass assuming a model of child18
  at the present day (redshift 0.0), defined with respect to the virial radius. 
  '''

    return concentration.concentration(10**log10mass, mdef, z, model=model)


def get_richness(log10mass, z=0.0, model='murata17'):
    '''
  Return the optical richness for a halo of a given mass assuming the model of murata17. Note that this relation only holds if the lambda is between 20 and 100.
  '''

    a, b, mass_pivot = _get_murata2017_parameters()
    lambda_ = np.exp(a + b * np.log((10**log10mass) / mass_pivot))

    if lambda_ < 20 or lambda_ > 100:
        print(log10mass, lambda_)
        raise Exception(
            'provided lambda value is outside the range supported by the murata17 relation'
        )

    return lambda_


def get_log10mass_from_richness(lambda_, z=0.0, model='murata17'):
    '''
  Return the mass for a halo of a given optical richness assuming the model of murata17. Note that this relation only holds if the lambda is between 20 and 100.
  '''
    if lambda_ < 20 or lambda_ > 100:
        print(lambda_)
        raise Exception(
            'provided lambda value is outside the range supported by the murata17 relation'
        )

    a, b, mass_pivot = _get_murata2017_parameters()
    return np.log10(np.exp((np.log(lambda_) - a) / b) * mass_pivot)


def _get_murata2017_parameters():
    '''
  Fetching parameters that are used in the murata17 richness-mass relation
  '''
    a = 3.207
    b = 0.993
    mass_pivot = 3. * 10**14
    return a, b, mass_pivot


def calculate_noise(sample, dex=0.0):
    random_noise = np.random.normal(1, dex, np.shape(sample))
    return 10**(random_noise * np.log10(sample))
