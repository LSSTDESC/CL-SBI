"""
Example script to use MCMC to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import mcmc, plotutils
import numpy as np
import os
import argparse
import json
import shutil

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument('--sim_dir')
parser.add_argument('--obs_dir')
parser.add_argument('--infer_dir')
args = parser.parse_args()

# Open the copy of infer_config in the infer_dir specified in the command line
script_dir = os.path.dirname(__file__)
infer_rel_path = '../configs/inference/' + args.infer_dir
infer_path = os.path.join(script_dir, infer_rel_path)
infer_filename = os.path.join(infer_path, 'infer_config.json')

with open(infer_filename, 'r') as f:
    infer_config = json.load(f)

# Read in the simulations in the sim_dir specified in the command line
sim_rel_path = '../configs/simulations/' + args.sim_dir
sim_path = os.path.join(script_dir, sim_rel_path)
sample_mc_pairs = np.load(os.path.join(sim_path, 'sample_mc_pairs.npy'))

# Read in the observations in the obs_dir specified in the command line
obs_rel_path = '../configs/observations/' + args.obs_dir
obs_path = os.path.join(script_dir, obs_rel_path)
drawn_mc_pairs = np.load(os.path.join(obs_path, 'drawn_mc_pairs.npy'))
drawn_nfw_profiles = np.load(os.path.join(obs_path, 'drawn_nfw_profiles.npy'))
true_param_mean = (np.mean(drawn_mc_pairs.T[0]), np.mean(drawn_mc_pairs.T[1]))

join_then_fit_chain = mcmc.join_then_fit(drawn_nfw_profiles)
fit_then_join_chain = mcmc.fit_then_join(drawn_nfw_profiles)

np.save(os.path.join(infer_path, 'jtf_chain.npy'), join_then_fit_chain)
np.save(os.path.join(infer_path, 'ftj_chain.npy'), fit_then_join_chain)
np.save(os.path.join(infer_path, 'true_param_mean.npy'), true_param_mean)

# Let's also copy the sim_config and obs_config to our output directory for cleaner provenance
shutil.copyfile(os.path.join(sim_path, 'sim_config.json'),
                os.path.join(infer_path, 'sim_config.json'))
shutil.copyfile(os.path.join(obs_path, 'obs_config.json'),
                os.path.join(infer_path, 'obs_config.json'))
