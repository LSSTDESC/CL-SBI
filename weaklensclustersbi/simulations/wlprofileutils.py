"""
Utilities for working with weak lensing profiles

Copyright 2022-2023, LSST-DESC
"""


def plot_profile(radii, profile, **plotkws):
    '''
    Quick plot for model profile
    '''
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog()
    plt.xlabel('radius [kpc/h]', fontsize='xx-large')
    plt.ylabel('Density profile [$\\rho/\\rho_m$]', fontsize='xx-large')
    plt.plot(radii, profile, **plotkws)

    plt.legend(fontsize='xx-large')
