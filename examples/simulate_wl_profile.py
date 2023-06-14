"""
Example script to simulate a weak lensing profile using modules
"""

from context import population, wlprofile
import numpy as np
import json
import os
import argparse

# Define a richness band from which we draw masses with some noise, and the
# concentration that scatters about that theoretical prediction

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
# parser.add_argument('--config_dir')
parser.add_argument('--sim_dir')
args = parser.parse_args()

# Open the copy of infer_config in the config_dir specified in the command line
script_dir = os.path.dirname(__file__)
config_rel_path = '../simulations/' + args.sim_dir
config_path = os.path.join(script_dir, config_rel_path)
config_filename = os.path.join(config_path, 'sim_config.json')

# Open the copy of sim_config in sim_dir
with open(config_filename, 'r') as f:
    sim_config = json.load(f)
'''
* simulated_nfw_profiles: needs to be 10k from randomly sampled log10masses in 
    range 13-15 and their corresponding concentrations
* drawn masses: should be a much smaller number ~10
'''

# ~10k randomly sampled log10masses and their corresponding concentrations.
# These are the "simulations" that we'll use for SBI
sample_mc_pairs = population.random_mass_conc(
    13, 15, sim_config['num_parameter_samples'])
rbins = 10**np.arange(0, sim_config['num_radial_bins'] / 10, 0.1)
simulated_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in sample_mc_pairs
])

# These are the ~10 log10masses that we've drawn from our richness bin of interest
drawn_mc_pairs = population.gen_mc_pairs_in_richness_bin(
    sim_config['min_richness'],
    sim_config['max_richness'],
    num_sims=sim_config['num_sims'],
    noise_dex=sim_config['sample_noise_dex'])

drawn_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in drawn_mc_pairs
])

measured_params = population.generate_richness_for_sample(
    [mc_pair[0] for mc_pair in sample_mc_pairs])  # log10masses

# Output to intermediate files in sim_dir to be read by inference example script
np.save(os.path.join(config_path, 'drawn_nfw_profiles.npy'),
        drawn_nfw_profiles)
np.save(os.path.join(config_path, 'drawn_mc_pairs.npy'), drawn_mc_pairs)
np.save(os.path.join(config_path, 'simulated_nfw_profiles.npy'),
        simulated_nfw_profiles)
np.save(os.path.join(config_path, 'sample_mc_pairs.npy'), sample_mc_pairs)
np.save(os.path.join(config_path, 'measured_params.npy'), measured_params)
