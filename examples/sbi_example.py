"""
Example script to use SBI to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import sbi_, plotutils
import numpy as np

# Read the output of the simulate_wl_profile.py
simulated_nfw_profiles = np.load('simulated_nfw_profiles.npy')
log10mass_sample = np.load('log10mass_sample.npy')
concentration_sample = np.load('concentration_sample.npy')

drawn_nfw_profiles = np.load('drawn_nfw_profiles.npy')
drawn_log10masses = np.load('drawn_log10masses.npy')
drawn_concentrations = np.load('drawn_concentrations.npy')

# mc_pairs = np.load('mc_pairs.npy')
# print(np.shape(mc_pairs))

# TODO: should be in a config
noise_dex = 0.3

sbi_.run_sbi(simulated_nfw_profiles, log10mass_sample, concentration_sample,
             drawn_nfw_profiles, drawn_log10masses, drawn_concentrations,
             noise_dex)