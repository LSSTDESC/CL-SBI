from context import plotutils
import numpy as np
import argparse
import os

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir')
args = parser.parse_args()

# Open the config_dir specified in the command line
script_dir = os.path.dirname(__file__)
config_rel_path = '../configs/' + args.config_dir
config_path = os.path.join(script_dir, config_rel_path)

mcmc_jtf = np.load(os.path.join(config_path, 'jtf_chain.npy'))
mcmc_ftj = np.load(os.path.join(config_path, 'ftj_chain.npy'))
mcmc_chains = [mcmc_jtf, mcmc_ftj]

sbi_chains = np.load(os.path.join(config_path, 'sbi_chains.npy'))
true_param_mean = np.load(os.path.join(config_path, 'true_param_mean.npy'))

# TODO: add a wrapper function so we have a single interface with 'pygtc' or 'cc' as a param
plotutils.plot_pygtc(mcmc_chains, config_path, 'mcmc', true_param_mean)
plotutils.plot_chainconsumer(mcmc_chains, config_path, 'mcmc',
                             list(true_param_mean))

plotutils.plot_pygtc(sbi_chains, config_path, 'sbi', true_param_mean)
plotutils.plot_chainconsumer(sbi_chains, config_path, 'sbi',
                             list(true_param_mean))
