from context import sbi_, mcmc
import numpy as np
import json
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--sim_id')
parser.add_argument('--infer_id')
parser.add_argument('--obs_id')
parser.add_argument('--num_sims')
parser.add_argument('--num_obs')

# Add regenerate flag if we want to overwrite any existing posterior.
# If false or not set, skip posterior generation if they already exist from an earlier run.
parser.add_argument('--regenerate', action='store_true')
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
# Go to/create output directory
out_rel_path = f'../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}'
out_path = os.path.join(script_dir, out_rel_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
# Checking if output already exists from an earlier script run
if (os.path.isfile(os.path.join(out_path, 'true_param_mean.npy'))):
    # Rerunning inference (continuing script)
    if args.regenerate:
        print(
            'Overwriting existing inference outputs because of --regenerate flag'
        )
    # Using existing inference outputs (terminating script)
    else:
        print(
            'Inference output already exists. If you want to regenerate, re-run with the --regenerate flag'
        )
        quit()

# Load posterior
posterior_rel_path = f'../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}'
posterior_path = os.path.join(script_dir, posterior_rel_path)
posterior_filename = os.path.join(posterior_path, 'posterior.pickle')
with open(posterior_filename, 'rb') as handle:
    posterior = pickle.load(handle)

# Load infer config
infer_config_rel_path = '../configs/inference/'
infer_config_path = os.path.join(script_dir, infer_config_rel_path)
infer_config_filename = os.path.join(infer_config_path,
                                     f'{args.infer_id}.json')
with open(infer_config_filename, 'r') as f:
    infer_config = json.load(f)

# Load observations
obs_rel_path = f'../outputs/observations/{args.obs_id}.{args.num_obs}'
obs_path = os.path.join(script_dir, obs_rel_path)
drawn_mc_pairs_filename = os.path.join(obs_path, 'drawn_mc_pairs.npy')
drawn_nfw_profiles_filename = os.path.join(obs_path, 'drawn_nfw_profiles.npy')
drawn_mc_pairs = np.load(drawn_mc_pairs_filename)
drawn_nfw_profiles = np.load(drawn_nfw_profiles_filename)

# Run SBI inference
sbi_chains = sbi_.apply_observations(posterior, drawn_mc_pairs,
                                     drawn_nfw_profiles)

# Output SBI chains
with open(os.path.join(out_path, 'sbi_chains.pickle'), 'wb') as handle:
    pickle.dump(sbi_chains, handle, protocol=4)

# Run MCMC inference
mcmc_jtf = mcmc.join_then_fit(drawn_nfw_profiles, infer_config['priors'])
mcmc_ftj = mcmc.fit_then_join(drawn_nfw_profiles, infer_config['priors'])

# Output MCMC chains (pickling because diff sizes)
with open(os.path.join(out_path, 'mcmc_chains.pickle'), 'wb') as handle:
    pickle.dump([mcmc_jtf, mcmc_ftj], handle, protocol=4)

# Output mean of drawn m-c pairs as "truth" value for plotting
true_param_mean = (np.mean(drawn_mc_pairs.T[0]), np.mean(drawn_mc_pairs.T[1]))
np.save(os.path.join(out_path, 'true_param_mean.npy'), true_param_mean)
