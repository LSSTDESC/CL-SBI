"""
Example script to simulate a weak lensing profile using modules
"""

from context import population, wlprofile
import numpy as np

# Define a richness band from which we draw masses with some noise, and the
# concentration that scatters about that theoretical prediction
# TODO: this should go into a config file
num_samples = 10000
num_drawn = 10
min_richness = 30
max_richness = 40
noise_dex = 0
rbins = 10**np.arange(0, 3, 0.1)
'''
* simulated_nfw_profiles: needs to be 10k from randomly sampled log10masses in 
    range 13-15 and their corresponding concentrations
* drawn masses: should be a much smaller number ~10
'''

# ~10k randomly sampled log10masses and their corresponding concentrations.
# These are the "simulations" that we'll use for SBI
sample_mc_pairs = population.random_mass_conc(13, 15, num_samples)
simulated_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in sample_mc_pairs
])

# These are the ~10 log10masses that we've drawn from our richness bin of interest
drawn_mc_pairs = population.gen_mc_pairs_in_richness_bin(min_richness,
                                                         max_richness,
                                                         num_drawn=num_drawn,
                                                         noise_dex=noise_dex)
drawn_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in drawn_mc_pairs
])

# Output to an intermediate file to be read by inference example script
np.save('drawn_nfw_profiles.npy', drawn_nfw_profiles)
np.save('drawn_mc_pairs.npy', drawn_mc_pairs)

np.save('simulated_nfw_profiles.npy', simulated_nfw_profiles)
np.save('sample_mc_pairs.npy', sample_mc_pairs)
