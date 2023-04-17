"""
Core structures and tools for dealing with a simulated sample of galaxy clusters

Notes::

    [sample format] weaklensingclustersbi currently supports samples defined with a given mean richness mass relation, or mean concentration-mass relation

Copyright 2022-2023, LSST-DESC
"""

def generate_concentration_mass_pairs() :
    '''Define a mass sample with some noise, and concentration sample that scatters about the theoretical prediction '''

log10mass_sample = np.random.uniform(13,15,size=10000)
non_noisy_concentration_sample = get_concentration(log10mass_sample)
noisy_concentration = np.random.normal(non_noisy_concentration_sample,0.2,10000)


