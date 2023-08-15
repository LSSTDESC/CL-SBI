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
parser.add_argument('--obs_id')
parser.add_argument('--num_obs')

# Add regenerate flag if we want to overwrite any existing observations.
# If false or not set, skip observation generation if they already exist from an earlier run.
parser.add_argument('--regenerate', action='store_true')
args = parser.parse_args()

# Open the copy of obs_config with the specified obs_id
script_dir = os.path.dirname(__file__)
config_rel_path = '../configs/observations/'
config_path = os.path.join(script_dir, config_rel_path)
config_filename = os.path.join(config_path, f'{args.obs_id}.json')

out_rel_path = f'../outputs/observations/{args.obs_id}.{args.num_obs}'
out_path = os.path.join(script_dir, out_rel_path)

# Checking if observations already exist from an earlier script run
if (os.path.isfile(os.path.join(out_path, 'drawn_nfw_profiles.npy'))):
    # Regenerating observations (continuing script)
    if args.regenerate:
        print('Overwriting existing observations because of --regenerate flag')
    # Using existing observations (terminating script)
    else:
        print(
            'Observations already exist. If you want to regenerate, re-run with the --regenerate flag'
        )
        quit()

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
    num_obs=int(args.num_obs),
    mc_scatter=obs_config['mc_scatter'],
    rm_scatter=obs_config['rm_scatter'],
    z=obs_config['z'],
)

noiseless_drawn_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins, obs_config['z'])
    for log10mass, concentration in drawn_mc_pairs
])
drawn_nfw_profiles = population.calculate_noise(
    noiseless_drawn_nfw_profiles, obs_config['profile_noise_dex'])

# Output to intermediate files in obs_dir to be read by inference example script
if not os.path.exists(out_path):
    os.makedirs(out_path)
np.save(os.path.join(out_path, 'noiseless_drawn_nfw_profiles.npy'),
        noiseless_drawn_nfw_profiles)
np.save(os.path.join(out_path, 'drawn_nfw_profiles.npy'), drawn_nfw_profiles)
np.save(os.path.join(out_path, 'drawn_mc_pairs.npy'), drawn_mc_pairs)
