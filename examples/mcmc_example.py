"""
Example script to use MCMC to recover log10mass and concentration from NFW profiles. 
This is using two approaches - join&fit, fit&join and plotting both together.
"""

from context import mcmc, plotutils
import numpy as np

# Read the output of the simulate_wl_profile.py
drawn_nfw_profiles = np.load('drawn_nfw_profiles.npy')

# sampler = mcmc.run_mcmc(simulated_nfw_profiles[0])
# plotutils.plot_chainconsumer(sampler)

join_then_fit_chain = mcmc.join_then_fit(drawn_nfw_profiles)
fit_then_join_chains = mcmc.fit_then_join(drawn_nfw_profiles)
plotutils.combine_chains_pygtc(join_then_fit_chain, fit_then_join_chains)
plotutils.combine_chains_chainconsumer(join_then_fit_chain,
                                       fit_then_join_chains)