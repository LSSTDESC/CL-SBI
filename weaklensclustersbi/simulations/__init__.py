"""
Simulations module for weak lensing cluster analysis.

This module provides tools for simulating NFW weak lensing profiles
and generating populations of galaxy clusters with mass-concentration
and richness-mass relations.

Submodules
----------
wlprofile
    NFW profile simulation.
population
    Galaxy cluster population generation.
populationutils
    Mass-concentration and richness-mass relations.
wlprofileutils
    Profile plotting utilities.
"""

from colossus.cosmology import cosmology

# Set default cosmology for colossus
cosmo = cosmology.setCosmology('planck18')

from . import wlprofile, population, populationutils, wlprofileutils

__all__ = [
    "wlprofile",
    "population",
    "populationutils",
    "wlprofileutils",
    "cosmo",
]
