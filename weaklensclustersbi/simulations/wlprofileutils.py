"""
Utilities for working with weak lensing profiles

Copyright 2022-2023, LSST-DESC
"""

from matplotlib import pyplot as plt

def plot_profile(radii, profile, **plotkws) :
    '''
    Quick plot for model profile
    '''

    plt.figure()
    plt.loglog()
    plt.xlabel('radius [kpc/h]', fontsize='xx-large')
    plt.ylabel('Density profile [$\\rho/\\rho_m$]', fontsize='xx-large')
    plt.plot(radius, profile, **plotkws)
    
    
    plt.legend(fontsize='xx-large')
    
