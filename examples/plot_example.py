from context import plotutils
import numpy as np

mcmc_jtf = np.load('mcmc_jtf.npy')
mcmc_ftj = np.load('mcmc_ftj.npy')
mcmc_chains = [mcmc_jtf, mcmc_ftj]

sbi_chains = np.load('sbi_chains.npy')
truths2d = np.load('truths2d.npy')

# TODO: add a wrapper function so we have a single interface with 'pygtc' or 'cc' as a param
plotutils.plot_pygtc(mcmc_chains, 'mcmc_pygtc', truths2d)
plotutils.plot_chainconsumer(mcmc_chains, 'mcmc_cc', list(truths2d))

plotutils.plot_pygtc(sbi_chains, 'sbi_pygtc', truths2d)
plotutils.plot_chainconsumer(sbi_chains, 'sbi_cc', list(truths2d))