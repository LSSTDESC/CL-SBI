"""
Example script to simulate a weak lensing profile using modules
"""

from context import population, wlprofile
import numpy as np

# Define a richness band from which we draw masses with some noise, and the
# concentration that scatters about that theoretical prediction

num_samples = 10000
min_richness = 30
max_richness = 40
rbins = 10**np.arange(0, 3, 0.1)

log10mass_sample = population.draw_masses_in_richness_bin(
    min_richness,
    max_richness,
    noise_dex=0,
    num_clusters=2,
)
concentration_sample = population.generate_concentration_for_sample(
    log10mass_sample, scatter_concentration_scale=0.2)
mc_pairs = list(zip(log10mass_sample, concentration_sample))
simulated_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in zip(log10mass_sample, concentration_sample)
])

# Output to an intermediate file to be read by inference example script
np.save('simulated_nfw_profiles.npy', simulated_nfw_profiles)
