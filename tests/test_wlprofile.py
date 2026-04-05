"""
Tests for wlprofile.py
"""

import pytest
import numpy as np

from weaklensclustersbi.simulations import wlprofile


class TestSimulateNfw:
    """Tests for simulate_nfw function."""

    def test_returns_array(self):
        """Test that function returns a numpy array."""
        profile = wlprofile.simulate_nfw(14.0, 5.0)
        assert isinstance(profile, np.ndarray)

    def test_output_shape_matches_rbins(self):
        """Test that output shape matches number of radial bins."""
        rbins = 10 ** np.arange(0, 2, 0.1)
        profile = wlprofile.simulate_nfw(14.0, 5.0, rbins=rbins)
        assert profile.shape == rbins.shape

    def test_profile_positive(self):
        """Test that all profile values are positive."""
        profile = wlprofile.simulate_nfw(14.0, 5.0)
        assert np.all(profile > 0)

    def test_profile_decreases_with_radius(self):
        """Test that surface density decreases with radius."""
        rbins = 10 ** np.arange(0, 3, 0.1)
        profile = wlprofile.simulate_nfw(14.0, 5.0, rbins=rbins)
        # For NFW, surface density should generally decrease with radius
        # Check that outer value is less than inner value
        assert profile[-1] < profile[0]

    def test_higher_mass_higher_profile(self):
        """Test that higher mass gives higher profile amplitude."""
        profile_low = wlprofile.simulate_nfw(13.5, 5.0)
        profile_high = wlprofile.simulate_nfw(14.5, 5.0)
        # Higher mass should have higher surface density
        assert np.mean(profile_high) > np.mean(profile_low)

    def test_density_kind(self):
        """Test that density kind returns valid output."""
        profile = wlprofile.simulate_nfw(14.0, 5.0, kind="density")
        assert isinstance(profile, np.ndarray)
        assert np.all(profile > 0)

    def test_surface_density_kind(self):
        """Test that surface_density kind returns valid output."""
        profile = wlprofile.simulate_nfw(14.0, 5.0, kind="surface_density")
        assert isinstance(profile, np.ndarray)
        assert np.all(profile > 0)

    def test_redshift_affects_profile(self):
        """Test that redshift affects the profile."""
        profile_z0 = wlprofile.simulate_nfw(14.0, 5.0, z=0.0)
        profile_z1 = wlprofile.simulate_nfw(14.0, 5.0, z=1.0)
        # Profiles should be different at different redshifts
        assert not np.allclose(profile_z0, profile_z1)


class TestModelProfiles:
    """Tests for model_profiles registry."""

    def test_nfw_registered(self):
        """Test that NFW model is registered."""
        assert "nfw" in wlprofile.model_profiles

    def test_nfw_callable(self):
        """Test that registered NFW function is callable."""
        nfw_fn = wlprofile.model_profiles["nfw"]
        profile = nfw_fn(14.0, 5.0)
        assert isinstance(profile, np.ndarray)
