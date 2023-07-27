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
args = parser.parse_args()

# Load posterior
script_dir = os.path.dirname(__file__)
posterior_rel_path = f'../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}'
posterior_path = os.path.join(script_dir, posterior_rel_path)
posterior_filename = os.path.join(posterior_path, 'posterior.pickle')
with open(posterior_filename, 'rb') as handle:
    posterior = pickle.load(handle)

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

# Go to/create output directory
out_rel_path = f'../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}'
out_path = os.path.join(script_dir, out_rel_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Output SBI chains
with open(os.path.join(out_path, 'sbi_chains.pickle'), 'wb') as handle:
    pickle.dump(sbi_chains, handle, protocol=4)

# Run MCMC inference
mcmc_jtf = mcmc.join_then_fit(drawn_nfw_profiles)
mcmc_ftj = mcmc.fit_then_join(drawn_nfw_profiles)

# Output MCMC chains (pickling because diff sizes)
with open(os.path.join(out_path, 'mcmc_chains.pickle'), 'wb') as handle:
    pickle.dump([mcmc_jtf, mcmc_ftj], handle, protocol=4)

# Output mean of drawn m-c pairs as "truth" value for plotting
true_param_mean = (np.mean(drawn_mc_pairs.T[0]), np.mean(drawn_mc_pairs.T[1]))
np.save(os.path.join(out_path, 'true_param_mean.npy'), true_param_mean)
