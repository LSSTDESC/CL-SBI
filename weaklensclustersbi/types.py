"""
Type definitions for weaklensclustersbi.

This module provides type aliases used throughout the package for better
type hints and documentation.
"""

from typing import Callable, TypedDict
import numpy as np
from numpy.typing import NDArray

# Array type aliases
FloatArray = NDArray[np.floating]

# Mass-concentration pair: (log10mass, concentration)
MCPair = tuple[float, float]
MCPairs = list[MCPair]

# Callable type for mass-concentration relations
MCRelation = Callable[[float, float], float]  # (log10mass, z) -> concentration

# Callable type for richness-mass relations
RMRelation = Callable[[float, float], float]  # (log10mass, z) -> richness


class PriorConfig(TypedDict, total=False):
    """Configuration dictionary for inference priors."""
    min_log10mass: float
    max_log10mass: float
    min_concentration: float
    max_concentration: float
    mc_scatter: float
    mc_relation: str
    min_z: float
    max_z: float


class MCMCConfig(TypedDict):
    """Configuration dictionary for MCMC sampling."""
    nwalkers: int
    npar: int
    starts: NDArray[np.floating]
    nsteps_burn: int
    nsteps_per_chain: int
