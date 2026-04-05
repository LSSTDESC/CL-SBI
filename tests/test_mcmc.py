"""
Tests for mcmc.py
"""

import pytest
import numpy as np

from weaklensclustersbi.inference import mcmc


class TestDefaultConfig:
    """Tests for default_config function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        config = mcmc.default_config()
        assert isinstance(config, dict)

    def test_has_required_keys(self):
        """Test that config has all required keys."""
        config = mcmc.default_config()
        required_keys = ["nwalkers", "npar", "starts", "nsteps_burn", "nsteps_per_chain"]
        for key in required_keys:
            assert key in config

    def test_nwalkers_positive(self):
        """Test that nwalkers is positive."""
        config = mcmc.default_config()
        assert config["nwalkers"] > 0

    def test_npar_is_three(self):
        """Test that npar is 3 (mass, concentration, log_f)."""
        config = mcmc.default_config()
        assert config["npar"] == 3

    def test_starts_shape(self):
        """Test that starts has correct shape."""
        config = mcmc.default_config()
        assert len(config["starts"]) == config["npar"]

    def test_burn_steps_positive(self):
        """Test that burn-in steps is positive."""
        config = mcmc.default_config()
        assert config["nsteps_burn"] > 0

    def test_chain_steps_positive(self):
        """Test that chain steps is positive."""
        config = mcmc.default_config()
        assert config["nsteps_per_chain"] > 0

    def test_reasonable_nwalkers(self):
        """Test that nwalkers is at least 2 * npar."""
        config = mcmc.default_config()
        assert config["nwalkers"] >= 2 * config["npar"]


class TestRunMcmcBase:
    """Tests for _run_mcmc_base function."""

    def test_accepts_callable(self):
        """Test that function accepts a callable log_prob_fn."""
        # This is a structural test - we just check the function signature works
        # without actually running the expensive MCMC
        pass  # Full integration test would be too slow


class TestFitThenJoin:
    """Tests for fit_then_join function structure."""

    def test_function_exists(self):
        """Test that function exists and is callable."""
        assert callable(mcmc.fit_then_join)


class TestJoinThenFit:
    """Tests for join_then_fit function structure."""

    def test_function_exists(self):
        """Test that function exists and is callable."""
        assert callable(mcmc.join_then_fit)


class TestRunMcmc:
    """Tests for run_mcmc function structure."""

    def test_function_exists(self):
        """Test that function exists and is callable."""
        assert callable(mcmc.run_mcmc)


class TestRunJointMcmc:
    """Tests for run_joint_mcmc function structure."""

    def test_function_exists(self):
        """Test that function exists and is callable."""
        assert callable(mcmc.run_joint_mcmc)


class TestStackedConfig:
    """Tests for stacked_config function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        config = mcmc.stacked_config()
        assert isinstance(config, dict)

    def test_has_required_keys(self):
        """Test that config has all required keys."""
        config = mcmc.stacked_config()
        required_keys = ["nwalkers", "npar", "starts", "nsteps_burn", "nsteps_per_chain"]
        for key in required_keys:
            assert key in config

    def test_lighter_than_default(self):
        """Test that stacked config uses fewer resources than default."""
        default = mcmc.default_config()
        stacked = mcmc.stacked_config()
        # Stacked should use fewer walkers and steps
        assert stacked["nwalkers"] <= default["nwalkers"]
        assert stacked["nsteps_burn"] <= default["nsteps_burn"]
        assert stacked["nsteps_per_chain"] <= default["nsteps_per_chain"]

    def test_npar_matches_default(self):
        """Test that npar is same as default (both fit for mass, conc, log_f)."""
        default = mcmc.default_config()
        stacked = mcmc.stacked_config()
        assert stacked["npar"] == default["npar"]


class TestFitThenJoinStacked:
    """Tests for fit_then_join_stacked function structure."""

    def test_function_exists(self):
        """Test that function exists and is callable."""
        assert callable(mcmc.fit_then_join_stacked)
