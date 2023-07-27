from context import sbi_
import numpy as np
import json
import argparse
import os
import pickle

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument('--infer_id')
parser.add_argument('--sim_id')
parser.add_argument('--num_sims')
args = parser.parse_args()

script_dir = os.path.dirname(__file__)

# Open the copy of infer_config with the specified infer_id
script_dir = os.path.dirname(__file__)
infer_config_rel_path = '../configs/inference/'
infer_config_path = os.path.join(script_dir, infer_config_rel_path)
infer_config_filename = os.path.join(infer_config_path,
                                     f'{args.infer_id}.json')
with open(infer_config_filename, 'r') as f:
    infer_config = json.load(f)
inferrer = sbi_.gen_inferrer(infer_config['priors'])

# Open simulations output
sim_rel_path = f'../outputs/simulations/{args.sim_id}.{args.num_sims}'
sim_path = os.path.join(script_dir, sim_rel_path)
sample_mc_pairs_filename = os.path.join(sim_path, 'sample_mc_pairs.npy')
simulated_nfw_profiles_filename = os.path.join(sim_path,
                                               'simulated_nfw_profiles.npy')
sample_mc_pairs = np.load(sample_mc_pairs_filename)
simulated_nfw_profiles = np.load(simulated_nfw_profiles_filename)

posterior = sbi_.gen_posterior(inferrer, sample_mc_pairs,
                               simulated_nfw_profiles)

# Pickle posterior
posterior_out_rel_path = f'../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}'
posterior_out_path = os.path.join(script_dir, posterior_out_rel_path)
if not os.path.exists(posterior_out_path):
    os.makedirs(posterior_out_path)
pickle.dump(posterior,
            open(os.path.join(posterior_out_path, 'posterior.pickle'), "wb"))
