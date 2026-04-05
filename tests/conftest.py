"""
Pytest configuration and fixtures for weaklensclustersbi tests.
"""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_mc_pairs(rng):
    """Provide sample mass-concentration pairs for testing."""
    masses = rng.uniform(13.5, 15.0, size=20)
    concentrations = rng.uniform(3.0, 8.0, size=20)
    return np.column_stack([masses, concentrations])


@pytest.fixture
def sample_nfw_profiles(rng):
    """Provide sample NFW profiles for testing."""
    # 20 observations, 30 radial bins
    return rng.uniform(1e7, 1e9, size=(20, 30))


@pytest.fixture
def sample_priors():
    """Provide sample prior configuration for testing."""
    return {
        "min_log10mass": 12.0,
        "max_log10mass": 16.0,
        "min_concentration": 1.0,
        "max_concentration": 15.0,
        "mc_scatter": 0.5,
        "mc_relation": "child18",
        "min_z": 0.1,
        "max_z": 0.5,
    }


@pytest.fixture
def sample_sigmas():
    """Provide sample sigma values (uncertainties) for testing."""
    return np.ones(30) * 0.1


@pytest.fixture(scope="session")
def cosmology_initialized():
    """Ensure colossus cosmology is initialized for the test session."""
    from colossus.cosmology import cosmology
    cosmology.setCosmology("planck18")
    return True
