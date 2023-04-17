"""
tests for population.py
"""
from weaklensingclustersbi.simulations import population
from numpy.testing import assert_array_less

def test_generate_concentration_for_sample() :
    log10masses = np.random.uniform(13,15,size=10)
    concentrations = population.generate_concentration_for_sample(log10masses=log10masses)
    min_concentrations = np.ones(10) * 4
    max_concentrations = np.ones(10) * 10
    assert_array_less(min_concentrations, concentrations)
    assert_array_less(concentrations, max_concentrations)
    
def test_generate_richness_for_sample() :
    pass
