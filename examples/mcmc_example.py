"""
Example script to use MCMC to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import mcmc, plotutils
import numpy as np

# Read the output of the simulate_wl_profile.py
drawn_nfw_profiles = np.load('drawn_nfw_profiles.npy')
sample_mc_pairs = np.load('sample_mc_pairs.npy')
truths2d = (np.mean(sample_mc_pairs.T[0]), np.mean(sample_mc_pairs.T[1]))

join_then_fit_chain = mcmc.join_then_fit(drawn_nfw_profiles)
fit_then_join_chain = mcmc.fit_then_join(drawn_nfw_profiles)

np.save('jtf_chain.npy', join_then_fit_chain)
np.save('ftj_chain.npy', fit_then_join_chain)