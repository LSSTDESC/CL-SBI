"""
Example script to use SBI to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import sbi_, plotutils
import numpy as np

# Read the output of the simulate_wl_profile.py
simulated_nfw_profiles = np.load('simulated_nfw_profiles.npy')
sample_mc_pairs = np.load('sample_mc_pairs.npy')
drawn_nfw_profiles = np.load('drawn_nfw_profiles.npy')
drawn_mc_pairs = np.load('drawn_mc_pairs.npy')

# TODO: should be in a config
noise_dex = 0.3

truths2d = (np.mean(sample_mc_pairs.T[0]), np.mean(sample_mc_pairs.T[1]))

chains = sbi_.run_sbi(simulated_nfw_profiles, sample_mc_pairs,
                      drawn_nfw_profiles, drawn_mc_pairs, noise_dex)
plotutils.plot_pygtc(chains, 'sbi_pygtc', truths2d)
plotutils.plot_chainconsumer(chains, 'sbi_cc', list(truths2d))