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
parser.add_argument('--sim_id')
parser.add_argument('--num_sims')
args = parser.parse_args()

# Open the copy of sim_config with the specified sim_id
script_dir = os.path.dirname(__file__)
sim_config_rel_path = '../configs/simulations/'
sim_config_path = os.path.join(script_dir, sim_config_rel_path)
sim_config_filename = os.path.join(sim_config_path, f'{args.sim_id}.json')

out_rel_path = f'../outputs/simulations/{args.sim_id}.{args.num_sims}'
out_path = os.path.join(script_dir, out_rel_path)

# Open the copy of sim_config in sim_dir
with open(sim_config_filename, 'r') as f:
    sim_config = json.load(f)
'''
* simulated_nfw_profiles: needs to be 10k from randomly sampled log10masses in 
    range specified in sim_config and their corresponding concentrations
* drawn masses: should be a much smaller number ~10
'''

# ~10k randomly sampled log10masses and their corresponding concentrations.
# These are the "simulations" that we'll use for SBI
sample_mc_pairs = population.random_mass_conc(
    sim_config['min_log10mass'],
    sim_config['max_log10mass'],
    int(args.num_sims),
    sim_config['sample_noise_dex'],
    mc_relation=sim_config['mc_relation'],
    mc_scatter=sim_config['mc_scatter'],
    mc_relation=sim_config['mc_relation'],
    z=sim_config['z'],
)

# Apply filtering criteria to subselect mc_pairs
filtered_mc_pairs = population.filter_mc_pairs(sample_mc_pairs,
                                               sim_config['mc_pair_subselect'])

# Simulate NFW profiles for each of the mc_pairs
rbins = 10**np.arange(0, sim_config['num_radial_bins'] / 10, 0.1)
simulated_nfw_profiles = np.array([
    wlprofile.simulate_nfw(log10mass, concentration, rbins)
    for log10mass, concentration in filtered_mc_pairs
])

# Output to intermediate files in sim_dir to be read by inference example script
if not os.path.exists(out_path):
    os.makedirs(out_path)
np.save(os.path.join(out_path, 'simulated_nfw_profiles.npy'),
        simulated_nfw_profiles)
np.save(os.path.join(out_path, 'sample_mc_pairs.npy'), filtered_mc_pairs)
