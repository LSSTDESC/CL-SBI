"""
Example script to use MCMC to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import mcmc, plotutils
import numpy as np
import os
import argparse
import json

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir')
parser.add_argument('--sim_dir')
args = parser.parse_args()

# Open the copy of infer_config in the config_dir specified in the command line
script_dir = os.path.dirname(__file__)
config_rel_path = '../configs/' + args.config_dir
config_path = os.path.join(script_dir, config_rel_path)
config_filename = os.path.join(config_path, 'infer_config.json')

with open(config_filename, 'r') as f:
    infer_config = json.load(f)

# Read in the simulations in the sim_dir specified in the command line
sim_rel_path = '../simulations/' + args.sim_dir
sim_path = os.path.join(script_dir, sim_rel_path)
sample_mc_pairs = np.load(os.path.join(sim_path, 'sample_mc_pairs.npy'))
drawn_nfw_profiles = np.load(os.path.join(sim_path, 'drawn_nfw_profiles.npy'))
truths2d = (np.mean(sample_mc_pairs.T[0]), np.mean(sample_mc_pairs.T[1]))

join_then_fit_chain = mcmc.join_then_fit(drawn_nfw_profiles, infer_config)
fit_then_join_chain = mcmc.fit_then_join(drawn_nfw_profiles, infer_config)

np.save(os.path.join(config_path, 'jtf_chain.npy'), join_then_fit_chain)
np.save(os.path.join(config_path, 'ftj_chain.npy'), fit_then_join_chain)