"""
Example script to use SBI to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import sbi_, plotutils
import numpy as np
from pathlib import Path
import json

# Open the copy of infer_config in the same directory as this script
# TODO: pass along the path to the infer config. Maybe as a command line arg?
p = Path(__file__).with_name('infer_config.json')
with p.open('r') as f:
    infer_config = json.load(f)

# Read the output of the simulate_wl_profile.py
simulated_nfw_profiles = np.load('simulated_nfw_profiles.npy')
sample_mc_pairs = np.load('sample_mc_pairs.npy')
drawn_nfw_profiles = np.load('drawn_nfw_profiles.npy')
drawn_mc_pairs = np.load('drawn_mc_pairs.npy')

truths2d = (np.mean(sample_mc_pairs.T[0]), np.mean(sample_mc_pairs.T[1]))

chains = sbi_.run_sbi(simulated_nfw_profiles, sample_mc_pairs,
                      drawn_nfw_profiles, drawn_mc_pairs,
                      infer_config['profile_noise_dex'])

# TODO: write to the same directory as the infer_config
np.save('sbi_chains.npy', chains)
np.save('truths2d.npy', truths2d)