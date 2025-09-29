from context import sbi_
import numpy as np
import json
import argparse
import os
import pickle

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument("--infer_id")
parser.add_argument("--sim_id")
parser.add_argument("--num_sims")
parser.add_argument("--num_obs")

# Add regenerate flag if we want to overwrite any existing posterior.
# If false or not set, skip posterior generation if they already exist from an earlier run.
parser.add_argument("--regenerate", action="store_true")
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
out_rel_path = f"../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}.{args.num_obs}"
out_path = os.path.join(script_dir, out_rel_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
# Checking if posterior already exists from an earlier script run
if os.path.isfile(os.path.join(out_path, "posterior.pickle")):
    # Regenerating posterior (continuing script)
    if args.regenerate:
        print("Overwriting existing posterior because of --regenerate flag")
    # Using existing posterior (terminating script)
    else:
        print(
            "Posterior already exists. If you want to regenerate, re-run with the --regenerate flag"
        )
        quit()

# Open the copy of infer_config with the specified infer_id
infer_config_rel_path = "../configs/inference/"
infer_config_path = os.path.join(script_dir, infer_config_rel_path)
infer_config_filename = os.path.join(infer_config_path, f"{args.infer_id}.json")
with open(infer_config_filename, "r") as f:
    infer_config = json.load(f)

# Open simulations output
sim_rel_path = f"../outputs/simulations/{args.sim_id}.{args.num_sims}.{args.num_obs}"
sim_path = os.path.join(script_dir, sim_rel_path)
sample_mc_pairs_filename = os.path.join(sim_path, "sample_mc_pairs.npy")
# sample_jtf_mc_pairs_filename = os.path.join(sim_path, "sample_jtf_mc_pairs.npy")
simulated_nfw_profiles_filename = os.path.join(sim_path, "simulated_nfw_profiles.npy")
simulated_jtf_nfw_profiles_filename = os.path.join(
    sim_path, "simulated_jtf_nfw_profiles.npy"
)
sample_mc_pairs = np.load(sample_mc_pairs_filename)
# sample_jtf_mc_pairs = np.load(sample_jtf_mc_pairs_filename)
simulated_nfw_profiles = np.load(simulated_nfw_profiles_filename)
simulated_jtf_nfw_profiles = np.load(simulated_jtf_nfw_profiles_filename)

inferrer = sbi_.gen_inferrer(infer_config["priors"])
posterior = sbi_.gen_posterior(inferrer, sample_mc_pairs, simulated_nfw_profiles)
inferrer = sbi_.gen_inferrer(infer_config["priors"])
posterior_range = sbi_.gen_posterior(
    inferrer, sample_mc_pairs, simulated_jtf_nfw_profiles
)

# Pickle posterior
pickle.dump(posterior, open(os.path.join(out_path, "posterior.pickle"), "wb"))
pickle.dump(posterior_range, open(os.path.join(out_path, "posterior_jtf.pickle"), "wb"))
