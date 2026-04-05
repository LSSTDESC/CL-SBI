"""
Tests for sbiutils.py
"""

import pytest
import numpy as np
import torch

from weaklensclustersbi.inference import sbiutils


class TestPercentileLevels:
    """Tests for PERCENTILE_LEVELS constant."""

    def test_contains_median(self):
        """Test that percentile levels include median (50)."""
        assert 50 in sbiutils.PERCENTILE_LEVELS

    def test_symmetric(self):
        """Test that percentile levels are roughly symmetric around 50."""
        levels = sbiutils.PERCENTILE_LEVELS
        # Check for pairs like (5, 95), (16, 84), etc.
        assert 5 in levels and 95 in levels
        assert 16 in levels and 84 in levels

    def test_sorted(self):
        """Test that percentile levels are sorted."""
        assert sbiutils.PERCENTILE_LEVELS == sorted(sbiutils.PERCENTILE_LEVELS)


class TestCreateObservationNfw:
    """Tests for create_observation_nfw function."""

    def test_returns_tensors(self):
        """Test that function returns PyTorch tensors."""
        mc_pair = [14.0, 5.0]
        profile = np.random.uniform(1e7, 1e9, size=30)
        theta, x = sbiutils.create_observation_nfw(mc_pair, profile)
        assert isinstance(theta, torch.Tensor)
        assert isinstance(x, torch.Tensor)

    def test_correct_dtype(self):
        """Test that tensors have float32 dtype."""
        mc_pair = [14.0, 5.0]
        profile = np.random.uniform(1e7, 1e9, size=30)
        theta, x = sbiutils.create_observation_nfw(mc_pair, profile)
        assert theta.dtype == torch.float32
        assert x.dtype == torch.float32

    def test_theta_shape(self):
        """Test that theta has correct shape."""
        mc_pair = [14.0, 5.0]
        profile = np.random.uniform(1e7, 1e9, size=30)
        theta, x = sbiutils.create_observation_nfw(mc_pair, profile)
        assert theta.shape == (2,)

    def test_x_shape(self):
        """Test that x has correct shape."""
        mc_pair = [14.0, 5.0]
        profile = np.random.uniform(1e7, 1e9, size=30)
        theta, x = sbiutils.create_observation_nfw(mc_pair, profile)
        assert x.shape == (30,)


class TestComputeMcSummary:
    """Tests for _compute_mc_summary function."""

    def test_output_length(self):
        """Test that output has correct length."""
        np.random.seed(42)
        mc_pairs = np.column_stack([
            np.random.uniform(13, 15, size=100),
            np.random.uniform(3, 8, size=100),
        ])
        summary = sbiutils._compute_mc_summary(mc_pairs)
        # 2 * len(PERCENTILE_LEVELS) + 1 for correlation
        expected_length = 2 * len(sbiutils.PERCENTILE_LEVELS) + 1
        assert len(summary) == expected_length

    def test_correlation_in_range(self):
        """Test that correlation is in valid range [-1, 1]."""
        np.random.seed(42)
        mc_pairs = np.column_stack([
            np.random.uniform(13, 15, size=100),
            np.random.uniform(3, 8, size=100),
        ])
        summary = sbiutils._compute_mc_summary(mc_pairs)
        correlation = summary[-1]
        assert -1 <= correlation <= 1

    def test_percentiles_ordered(self):
        """Test that percentiles are correctly ordered."""
        np.random.seed(42)
        mc_pairs = np.column_stack([
            np.random.uniform(13, 15, size=100),
            np.random.uniform(3, 8, size=100),
        ])
        summary = sbiutils._compute_mc_summary(mc_pairs)
        n_levels = len(sbiutils.PERCENTILE_LEVELS)
        mass_percentiles = summary[:n_levels]
        conc_percentiles = summary[n_levels:2*n_levels]
        # Lower percentiles should be smaller than higher percentiles
        assert mass_percentiles[0] < mass_percentiles[-1]
        assert conc_percentiles[0] < conc_percentiles[-1]


class TestCreateJoinFitObservationNfw:
    """Tests for create_join_fit_observation_nfw function."""

    def test_returns_tensors(self):
        """Test that function returns PyTorch tensors."""
        np.random.seed(42)
        mc_pairs = np.column_stack([
            np.random.uniform(13, 15, size=20),
            np.random.uniform(3, 8, size=20),
        ])
        profiles = np.random.uniform(1e7, 1e9, size=(20, 30))
        theta, x = sbiutils.create_join_fit_observation_nfw(mc_pairs, profiles)
        assert isinstance(theta, torch.Tensor)
        assert isinstance(x, torch.Tensor)

    def test_x_is_median_profile(self):
        """Test that x is the median profile."""
        np.random.seed(42)
        mc_pairs = np.column_stack([
            np.random.uniform(13, 15, size=20),
            np.random.uniform(3, 8, size=20),
        ])
        profiles = np.random.uniform(1e7, 1e9, size=(20, 30))
        theta, x = sbiutils.create_join_fit_observation_nfw(mc_pairs, profiles)
        expected_median = np.median(profiles, axis=0)
        np.testing.assert_array_almost_equal(x.numpy(), expected_median.astype(np.float32))


class TestCreateFitJoinObservationNfw:
    """Tests for create_fit_join_observation_nfw function."""

    def test_returns_tensors(self):
        """Test that function returns PyTorch tensors."""
        np.random.seed(42)
        mc_pairs = np.column_stack([
            np.random.uniform(13, 15, size=20),
            np.random.uniform(3, 8, size=20),
        ])
        profiles = np.random.uniform(1e7, 1e9, size=(20, 30))
        theta, x = sbiutils.create_fit_join_observation_nfw(mc_pairs, profiles)
        assert isinstance(theta, torch.Tensor)
        assert isinstance(x, torch.Tensor)

    def test_x_is_flattened_profiles(self):
        """Test that x contains all profiles flattened."""
        np.random.seed(42)
        n_obs = 20
        n_bins = 30
        mc_pairs = np.column_stack([
            np.random.uniform(13, 15, size=n_obs),
            np.random.uniform(3, 8, size=n_obs),
        ])
        profiles = np.random.uniform(1e7, 1e9, size=(n_obs, n_bins))
        theta, x = sbiutils.create_fit_join_observation_nfw(mc_pairs, profiles)
        assert x.shape == (n_obs * n_bins,)
        np.testing.assert_array_almost_equal(x.numpy(), profiles.flatten().astype(np.float32))
