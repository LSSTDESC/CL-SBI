"""
Tests for populationutils.py
"""

import pytest
import numpy as np

from weaklensclustersbi.simulations import populationutils


class TestGetConcentration:
    """Tests for get_concentration function."""

    def test_concentration_positive(self):
        """Test that concentration is always positive."""
        conc = populationutils.get_concentration(14.0, z=0.0)
        assert conc > 0

    def test_concentration_decreases_with_mass(self):
        """Test that concentration decreases with increasing mass (general trend)."""
        conc_low_mass = populationutils.get_concentration(13.0, z=0.0)
        conc_high_mass = populationutils.get_concentration(15.0, z=0.0)
        assert conc_low_mass > conc_high_mass

    def test_different_models(self):
        """Test that different models return different values."""
        conc_child18 = populationutils.get_concentration(14.0, model="child18")
        conc_diemer19 = populationutils.get_concentration(14.0, model="diemer19")
        # Different models should give different results
        # (not exactly equal due to different fitting)
        assert conc_child18 != conc_diemer19


class TestGetRichness:
    """Tests for get_richness function."""

    def test_richness_positive(self):
        """Test that richness is always positive."""
        richness = populationutils.get_richness(14.5, z=0.0)
        assert richness > 0

    def test_richness_increases_with_mass(self):
        """Test that richness increases with mass."""
        richness_low = populationutils.get_richness(14.0, z=0.0)
        richness_high = populationutils.get_richness(15.0, z=0.0)
        assert richness_high > richness_low

    def test_murata17_model(self):
        """Test murata17 model returns values in expected range."""
        # Mass that should give richness around 50
        richness = populationutils.get_richness(14.5, model="murata17")
        assert 10 < richness < 200

    def test_mcclintock18_model(self):
        """Test mcclintock18 model returns values."""
        richness = populationutils.get_richness(14.5, model="mcclintock18")
        assert richness > 0

    def test_strict_mode_warns(self):
        """Test that strict mode warns for out-of-range values."""
        # Very low mass should give richness below 20
        with pytest.warns(UserWarning, match="outside the range"):
            populationutils.get_richness(13.0, model="murata17", strict=True)


class TestGetLog10MassFromRichness:
    """Tests for get_log10mass_from_richness function."""

    def test_inverse_of_get_richness(self):
        """Test that function is approximately inverse of get_richness."""
        original_mass = 14.5
        richness = populationutils.get_richness(original_mass, model="murata17")
        recovered_mass = populationutils.get_log10mass_from_richness(
            richness, model="murata17"
        )
        np.testing.assert_almost_equal(original_mass, recovered_mass, decimal=5)

    def test_mass_increases_with_richness(self):
        """Test that mass increases with richness."""
        mass_low = populationutils.get_log10mass_from_richness(30, model="murata17")
        mass_high = populationutils.get_log10mass_from_richness(80, model="murata17")
        assert mass_high > mass_low

    def test_strict_mode_warns(self):
        """Test that strict mode warns for out-of-range values."""
        with pytest.warns(UserWarning, match="outside the range"):
            populationutils.get_log10mass_from_richness(
                10, model="murata17", strict=True
            )


class TestMurata2017Parameters:
    """Tests for _get_murata2017_parameters function."""

    def test_returns_three_values(self):
        """Test that function returns three parameter values."""
        a, b, mass_pivot = populationutils._get_murata2017_parameters()
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(mass_pivot, float)

    def test_parameters_positive(self):
        """Test that parameters have expected signs."""
        a, b, mass_pivot = populationutils._get_murata2017_parameters()
        assert a > 0
        assert b > 0
        assert mass_pivot > 0


class TestMcclintock18Parameters:
    """Tests for _get_mcclintock18_parameters function."""

    def test_returns_three_values(self):
        """Test that function returns three parameter values."""
        m0, G, F = populationutils._get_mcclintock18_parameters()
        assert isinstance(m0, float)
        assert isinstance(G, float)
        assert isinstance(F, float)

    def test_m0_reasonable(self):
        """Test that pivot mass is reasonable."""
        m0, G, F = populationutils._get_mcclintock18_parameters()
        # Should be around 3e14 solar masses
        assert 1e14 < m0 < 1e15
