"""
Tests for population.py
"""

import pytest
import numpy as np
from numpy.testing import assert_array_less

from weaklensclustersbi.simulations import population


class TestGenerateConcentrationForSample:
    """Tests for generate_concentration_for_sample function."""

    def test_concentration_in_valid_range(self):
        """Test that generated concentrations fall within expected range."""
        np.random.seed(42)
        log10masses = np.random.uniform(13, 15, size=10)
        zs = np.zeros(10)
        concentrations = population.generate_concentration_for_sample(
            log10masses=log10masses, zs=zs
        )
        min_concentrations = np.ones(10) * 2
        max_concentrations = np.ones(10) * 12
        assert_array_less(min_concentrations, concentrations)
        assert_array_less(concentrations, max_concentrations)

    def test_output_shape(self):
        """Test that output has same shape as input."""
        np.random.seed(42)
        log10masses = np.random.uniform(13, 15, size=20)
        zs = np.random.uniform(0, 1, size=20)
        concentrations = population.generate_concentration_for_sample(
            log10masses=log10masses, zs=zs
        )
        assert concentrations.shape == log10masses.shape


class TestGenerateRichnessForSample:
    """Tests for generate_richness_for_sample function."""

    def test_richness_in_valid_range(self):
        """Test that generated richnesses fall within expected range."""
        np.random.seed(42)
        log10masses = np.random.uniform(14.4, 15, size=10)
        richnesses = population.generate_richness_for_sample(
            log10masses=log10masses
        )
        min_richnesses = np.ones(10) * 10
        max_richnesses = np.ones(10) * 200
        assert_array_less(min_richnesses, richnesses)
        assert_array_less(richnesses, max_richnesses)

    def test_output_shape(self):
        """Test that output has same shape as input."""
        np.random.seed(42)
        log10masses = np.random.uniform(14, 15, size=15)
        richnesses = population.generate_richness_for_sample(
            log10masses=log10masses
        )
        assert richnesses.shape == log10masses.shape


class TestDrawMassesInRichnessBin:
    """Tests for draw_masses_in_richness_bin function."""

    def test_returns_correct_number_of_samples(self):
        """Test that function returns correct number of samples."""
        np.random.seed(42)
        num_samples = 14
        lambdas, log10masses = population.draw_masses_in_richness_bin(
            30, 40, num_samples=num_samples
        )
        assert len(log10masses) == num_samples
        assert len(lambdas) == num_samples

    def test_masses_in_expected_range(self):
        """Test that drawn masses are in a reasonable range."""
        np.random.seed(42)
        num_samples = 50
        lambdas, log10masses = population.draw_masses_in_richness_bin(
            30, 40, num_samples=num_samples
        )
        # Masses should be roughly in the range 14-15 for this richness bin
        assert np.all(log10masses > 13.5)
        assert np.all(log10masses < 16)


class TestRandomMassConc:
    """Tests for random_mass_conc function."""

    def test_returns_list_of_tuples(self):
        """Test that function returns list of (mass, conc) tuples."""
        np.random.seed(42)
        mc_pairs = population.random_mass_conc(
            min_log10mass=13,
            max_log10mass=15,
            num_sims=10,
            mc_relation="child18",  # Use valid MC relation
        )
        assert len(mc_pairs) == 10
        assert all(len(pair) == 2 for pair in mc_pairs)

    def test_masses_in_specified_range(self):
        """Test that masses are within specified range."""
        np.random.seed(42)
        mc_pairs = population.random_mass_conc(
            min_log10mass=13.5,
            max_log10mass=14.5,
            num_sims=100,
            mc_relation="child18",  # Use valid MC relation
        )
        masses = [pair[0] for pair in mc_pairs]
        assert all(13.5 <= m <= 14.5 for m in masses)


class TestCalculateNoise:
    """Tests for calculate_noise function."""

    def test_no_noise_returns_original(self):
        """Test that dex=0 returns original values."""
        np.random.seed(42)
        sample = np.array([1.0, 10.0, 100.0])
        result = population.calculate_noise(sample, dex=0.0)
        np.testing.assert_array_almost_equal(result, sample)

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        np.random.seed(42)
        sample = np.random.uniform(1, 100, size=(5, 10))
        result = population.calculate_noise(sample, dex=0.1)
        assert result.shape == sample.shape


class TestGenMcPairsInRichnessBin:
    """Tests for gen_mc_pairs_in_richness_bin function."""

    def test_returns_correct_number_of_pairs(self):
        """Test that function returns correct number of mc pairs."""
        np.random.seed(42)
        num_samples = 20
        mc_pairs = population.gen_mc_pairs_in_richness_bin(
            lambda_min=30,
            lambda_max=45,
            num_samples=num_samples,
        )
        assert len(mc_pairs) == num_samples

    def test_pairs_have_two_elements(self):
        """Test that each pair has mass and concentration."""
        np.random.seed(42)
        mc_pairs = population.gen_mc_pairs_in_richness_bin(
            lambda_min=30,
            lambda_max=45,
            num_samples=10,
        )
        assert all(len(pair) == 2 for pair in mc_pairs)
