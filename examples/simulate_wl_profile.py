"""
Example script to simulate a weak lensing profile using modules
"""

from context import population, wlprofile
import numpy as np
import json
from pathlib import Path

# Define a richness band from which we draw masses with some noise, and the
# concentration that scatters about that theoretical prediction

# Open the copy of sim_config in the same directory as this script
# TODO: have the input directory passed in as well
p = Path(__file__).with_name('sim_config.json')
with p.open('r') as f:
    sim_config = json.load(f)
'''
* simulated_nfw_profiles: needs to be 10k from randomly sampled log10masses in 
    range 13-15 and their corresponding concentrations
* drawn masses: should be a much smaller number ~10
'''

# ~10k randomly sampled log10masses and their corresponding concentrations.
# These are the "simulations" that we'll use for SBI
sample_mc_pairs = population.random_mass_conc(13, 15,
                                              sim_config['num_samples'])
rbins = 10**np.arange(0, sim_config['num_radial_bins'] / 10, 0.1)
simulated_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in sample_mc_pairs
])

# These are the ~10 log10masses that we've drawn from our richness bin of interest
drawn_mc_pairs = population.gen_mc_pairs_in_richness_bin(
    sim_config['min_richness'],
    sim_config['max_richness'],
    num_drawn=sim_config['num_drawn'],
    noise_dex=sim_config['sample_noise_dex'])

drawn_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in drawn_mc_pairs
])

sim_output_dir = sim_config['sim_output_dir']

# Output to an intermediate file to be read by inference example script
np.save(f'{sim_output_dir}/drawn_nfw_profiles.npy', drawn_nfw_profiles)
np.save(f'{sim_output_dir}/drawn_mc_pairs.npy', drawn_mc_pairs)

np.save(f'{sim_output_dir}/simulated_nfw_profiles.npy', simulated_nfw_profiles)
np.save(f'{sim_output_dir}/sample_mc_pairs.npy', sample_mc_pairs)
