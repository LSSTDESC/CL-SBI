"""
weaklensclustersbi - Weak Lensing Galaxy Clusters using Simulation Based Inference

A research package for comparing SBI and MCMC inference methods on weak lensing
observations of galaxy clusters. The package simulates NFW weak lensing profiles
and uses neural density estimation (SNPE) to infer mass and concentration parameters.

Copyright 2022-2023, LSST-DESC

Subpackages
-----------
simulations
    NFW profile generation and mass-concentration/richness-mass relations.
inference
    SBI (SNPE) and MCMC (emcee) implementations.

Examples
--------
>>> from weaklensclustersbi.simulations import wlprofile
>>> profile = wlprofile.simulate_nfw(14.5, 5.0, z=0.3)

>>> from weaklensclustersbi.inference import mcmc
>>> config = mcmc.default_config()
"""

__version__ = "0.1.0"

__all__ = [
    "simulations",
    "inference",
    "types",
    "__version__",
]
