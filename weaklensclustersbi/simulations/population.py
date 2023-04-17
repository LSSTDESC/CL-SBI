"""
Core structures and tools for dealing with a simulated population of galaxy clusters that can be observed

Notes::

    [sample format] weaklensingclustersbi currently supports samples defined with a given mean richness mass relation, or mean concentration-mass relation

Copyright 2022-2023, LSST-DESC
"""
import numpy as np

def generate_concentration_for_sample(log10masses, z=0.0,
                                      mcrelation='child18',
                                      scatter_concentration_scale=0.2) :
    '''From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift, 
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
    
    non_noisy_concentrations =  get_concentration(log10masses, z=z, model=mcrelation)
    # Note: May want to later generalize the distribution of concentration about the mean relation that is not random normal
    concentrations =  np.random.normal(non_noisy_concentrations,scatter_concentration_scale,np.shape(log10masses))
    
    return concentrations


def generate_richness_for_sample(log10masses, z=0.0,
                                 rmrelation='murata18',
                                 scatter_richness_scale=1.0) :
    '''From the assumed true masses for a sampled population of simulated galaxy clusters at a given redshift, 
    define the corresponding richness of this population assuming a mean richness-mass relation 
    based on theoretical prediction) with some scatter in richness at fixed mass. 

    Args:
        log10masses: a numpy array of masses, e.g. np.random.uniform(13,15,size=10000)
        z : redshift, default: 0.0
        rmrelation : string that is a key to the model name of the M-c relation from colossus, default: child18

    Returns:
        concentrations : a numpy array of concentration values

    '''


    pass

