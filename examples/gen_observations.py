"""
Generate observations upon which we will run inference
"""

from context import population, wlprofile
import numpy as np
import json
import os
import argparse

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument('--obs_dir')
args = parser.parse_args()

# Open the copy of obs_config in the obs_dir specified in the command line
script_dir = os.path.dirname(__file__)
config_rel_path = '../configs/observations/' + args.obs_dir
config_path = os.path.join(script_dir, config_rel_path)
config_filename = os.path.join(config_path, 'obs_config.json')

# Open the copy of obs_config in obs_dir
with open(config_filename, 'r') as f:
    obs_config = json.load(f)

rbins = 10**np.arange(0, obs_config['num_radial_bins'] / 10, 0.1)

# These are the ~10 log10masses that we've drawn from our richness bin of interest
drawn_mc_pairs = population.gen_mc_pairs_in_richness_bin(
    obs_config['min_richness'],
    obs_config['max_richness'],
    rm_relation=obs_config['rm_relation'],
    mc_relation=obs_config['mc_relation'],
    num_obs=obs_config['num_obs'],
    mc_scatter=obs_config['mc_scatter'],
    rm_scatter=obs_config['rm_scatter'],
)

non_noisy_drawn_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins, obs_config['z'])
    for log10mass, concentration in drawn_mc_pairs
])
drawn_nfw_profiles = population.calculate_noise(
    non_noisy_drawn_nfw_profiles, obs_config['profile_noise_dex'])

# Output to intermediate files in obs_dir to be read by inference example script
np.save(os.path.join(config_path, 'drawn_nfw_profiles.npy'),
        drawn_nfw_profiles)
np.save(os.path.join(config_path, 'drawn_mc_pairs.npy'), drawn_mc_pairs)
