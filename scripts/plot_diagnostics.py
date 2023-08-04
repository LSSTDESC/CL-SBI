from context import plotutils
import numpy as np
import argparse
import os
import pickle
import json

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument('--sim_id')
parser.add_argument('--infer_id')
parser.add_argument('--obs_id')
parser.add_argument('--num_sims')
parser.add_argument('--num_obs')

# Add regenerate flag if we want to overwrite any existing plots.
# If false or not set, skip plot generation if they already exist from an earlier run.
parser.add_argument('--regenerate', action='store_true')
args = parser.parse_args()

# Open the infer_dir specified in the command line
script_dir = os.path.dirname(__file__)
infer_rel_path = f'../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}'
infer_path = os.path.join(script_dir, infer_rel_path)

out_rel_path = f'../outputs/plots/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}/diagnostics'
out_path = os.path.join(script_dir, out_rel_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Checking if plots already exist from an earlier script run
if (os.path.isfile(os.path.join(out_path, 'mcmc_ftj_cc.png'))):
    # Regenerating plots (continuing script)
    if args.regenerate:
        print(
            'Overwriting existing diagnostic plots because of --regenerate flag'
        )
    # Using existing plots (terminating script)
    else:
        print(
            'Diagnostic plots already exist. If you want to regenerate, re-run with the --regenerate flag'
        )
        quit()

with open(os.path.join(infer_path, 'mcmc_jtf_sampler.pickle'), 'rb') as handle:
    mcmc_jtf_sampler = pickle.load(handle)
with open(os.path.join(infer_path, 'mcmc_ftj_samplers.pickle'),
          'rb') as handle:
    mcmc_ftj_samplers = pickle.load(handle)
with open(os.path.join(infer_path, 'sbi_ftj_chains.pickle'), 'rb') as handle:
    sbi_ftj_chains = pickle.load(handle)

true_param_mean = np.load(os.path.join(infer_path, 'true_param_mean.npy'))

# Plot the walkers for jtf sampler
plotutils.plot_walkers(mcmc_jtf_sampler, out_path, 'mcmc_jtf_')

mcmc_ftj_chains = []
for i in range(len(mcmc_ftj_samplers)):
    # Plot the walkers for each of the ftj samplers
    mcmc_ftj_chains.append(mcmc_ftj_samplers[i].flatchain)

# Plotting contour plots for each of the observations (that we later join in fit_then_join)
plotutils.plot_cc_diagnostic(mcmc_ftj_chains, out_path, 'mcmc_ftj',
                             list(true_param_mean))
plotutils.plot_cc_diagnostic(sbi_ftj_chains, out_path, 'sbi_ftj',
                             list(true_param_mean))

# Load observations
obs_rel_path = f'../outputs/observations/{args.obs_id}.{args.num_obs}'
obs_path = os.path.join(script_dir, obs_rel_path)

obs_config_rel_path = '../configs/observations/'
obs_config_path = os.path.join(script_dir, obs_config_rel_path)
obs_config_filename = os.path.join(obs_config_path, f'{args.obs_id}.json')
with open(obs_config_filename, 'r') as f:
    obs_config = json.load(f)

drawn_mc_pairs_filename = os.path.join(obs_path, 'drawn_mc_pairs.npy')
drawn_nfw_profiles_filename = os.path.join(obs_path, 'drawn_nfw_profiles.npy')
drawn_mc_pairs = np.load(drawn_mc_pairs_filename)
drawn_nfw_profiles = np.load(drawn_nfw_profiles_filename)

# Plotting drawn m-c pairs
plotutils.plot_mc_pairs(drawn_mc_pairs, obs_path)

# Plotting drawn NFW profiles
plotutils.plot_nfw_profiles(
    drawn_nfw_profiles,
    obs_path,
    obs_config['num_radial_bins'],
    obs_config["min_richness"],
    obs_config["max_richness"],
)
