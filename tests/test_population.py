"""
tests for population.py
"""
from context import population
from numpy.testing import assert_array_less
import numpy as np


def test_generate_concentration_for_sample():
    log10masses = np.random.uniform(13, 15, size=10)
    concentrations = population.generate_concentration_for_sample(
        log10masses=log10masses)
    min_concentrations = np.ones(10) * 4
    max_concentrations = np.ones(10) * 10
    assert_array_less(min_concentrations, concentrations)
    assert_array_less(concentrations, max_concentrations)


def test_generate_richness_for_sample():
    log10masses = np.random.uniform(14.4, 15, size=10)
    richnesses = population.generate_richness_for_sample(
        log10masses=log10masses)
    min_richnesses = np.ones(10) * 20
    max_richnesses = np.ones(10) * 100
    assert_array_less(min_richnesses, richnesses)
    assert_array_less(richnesses, max_richnesses)


def test_draw_masses_in_richness_bin():
    num_clusters = 14
    log10masses = population.draw_masses_in_richness_bin(
        30, 40, num_clusters=num_clusters)
    min_log10masses = np.ones(num_clusters) * 14.5
    max_log10masses = np.ones(num_clusters) * 14.8
    assert_array_less(min_log10masses, log10masses)
    assert_array_less(log10masses, max_log10masses)


test_generate_concentration_for_sample()
test_generate_richness_for_sample()
test_draw_masses_in_richness_bin()
