"""
Example script to use MCMC to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import mcmc, plotutils, inferutils
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
filtered_mc_pairs = inferutils.filter_mc_pairs(
    sample_mc_pairs, infer_config['mc_pair_subselect'])
drawn_mc_pairs = np.load(os.path.join(sim_path, 'drawn_mc_pairs.npy'))
drawn_nfw_profiles = np.load(os.path.join(sim_path, 'drawn_nfw_profiles.npy'))
true_param_mean = (np.mean(drawn_mc_pairs.T[0]), np.mean(drawn_mc_pairs.T[1]))

join_then_fit_chain = mcmc.join_then_fit(drawn_nfw_profiles)
fit_then_join_chain = mcmc.fit_then_join(drawn_nfw_profiles)

np.save(os.path.join(config_path, 'jtf_chain.npy'), join_then_fit_chain)
np.save(os.path.join(config_path, 'ftj_chain.npy'), fit_then_join_chain)
np.save(os.path.join(config_path, 'true_param_mean.npy'), true_param_mean)

# Let's also copy the sim_config to our output directory for cleaner provenance
shutil.copyfile(os.path.join(sim_path, 'sim_config.json'),
                os.path.join(config_path, 'sim_config.json'))
