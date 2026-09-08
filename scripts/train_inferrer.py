import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weaklensclustersbi.inference import stackutils
from weaklensclustersbi.inference import sbi_
import numpy as np
import json
import argparse
import os
import pickle
import time
from runtime_log import append_runtime_log

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument("--infer_id")
parser.add_argument("--sim_id")
parser.add_argument("--num_sims")
parser.add_argument("--num_obs")
# Observable: "surface_density" (default) or "delta_sigma". Must match the observable
# used for gen_simulations; reads suffixed sims and writes suffixed posteriors.
parser.add_argument("--observable", default="surface_density")
# JTF stack estimator: "median" (default) or "corrected_mean". Must match gen_simulations.
parser.add_argument("--stack_estimator", default="median")
# Optional torch/numpy training seed. If set, seeds the network initialization/training RNG
# and writes the posteriors to a .seed{N}-suffixed dir (for seed-ensemble studies; the
# training DATA is unchanged -- only the network init and batch order vary).
parser.add_argument("--train_seed", type=int, default=None)

# Add regenerate flag if we want to overwrite any existing posterior.
# If false or not set, skip posterior generation if they already exist from an earlier run.
parser.add_argument("--regenerate", action="store_true")
args = parser.parse_args()

script_start = time.perf_counter()


def log_runtime(status="success", details=""):
    append_runtime_log(
        stage="train_inferrer",
        seconds=time.perf_counter() - script_start,
        sim_id=args.sim_id,
        infer_id=args.infer_id,
        num_sims=args.num_sims,
        num_obs=args.num_obs,
        details=details,
        status=status,
    )


script_dir = os.path.dirname(__file__)
obs_suffix = "" if args.observable == "surface_density" else f".{args.observable}"
obs_suffix += stackutils.stack_suffix(args.stack_estimator)
seed_suffix = f".seed{args.train_seed}" if args.train_seed is not None else ""
if args.train_seed is not None:
    import torch
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
out_rel_path = f"../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}.{args.num_obs}{obs_suffix}{seed_suffix}"
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
        log_runtime(status="skipped", details="existing posterior")
        quit()

# Open the copy of infer_config with the specified infer_id
infer_config_rel_path = "../configs/inference/"
infer_config_path = os.path.join(script_dir, infer_config_rel_path)
infer_config_filename = os.path.join(infer_config_path, f"{args.infer_id}.json")
with open(infer_config_filename, "r") as f:
    infer_config = json.load(f)

# Open simulations output
sim_rel_path = f"../outputs/simulations/{args.sim_id}.{args.num_sims}.{args.num_obs}{obs_suffix}"
sim_path = os.path.join(script_dir, sim_rel_path)
sample_mc_pairs_filename = os.path.join(sim_path, "sample_mc_pairs.npy")
# sample_jtf_mc_pairs_filename = os.path.join(sim_path, "sample_jtf_mc_pairs.npy")
simulated_nfw_profiles_filename = os.path.join(sim_path, "simulated_nfw_profiles.npy")
simulated_jtf_nfw_profiles_filename = os.path.join(
    sim_path, "simulated_jtf_nfw_profiles.npy"
)
sample_mc_percentiles_filename = os.path.join(sim_path, "sample_mc_percentiles.npy")
sample_mc_correlations_filename = os.path.join(sim_path, "sample_mc_correlations.npy")
sample_mc_pairs = np.load(sample_mc_pairs_filename)

if sample_mc_pairs.ndim == 3:
    percentiles = np.concatenate(
        (sample_mc_pairs[:, :, 0], sample_mc_pairs[:, :, 1]), axis=1
    )
    if os.path.exists(sample_mc_correlations_filename):
        correlations = np.load(sample_mc_correlations_filename)
    else:
        correlations = np.array(
            [
                np.nan_to_num(
                    np.corrcoef(block.T)[0, 1], nan=0.0, posinf=0.0, neginf=0.0
                )
                for block in sample_mc_pairs
            ]
        )
    correlations = np.clip(correlations, -0.99, 0.99)
    sample_mc_pairs = np.column_stack((percentiles, correlations))
simulated_nfw_profiles = np.load(simulated_nfw_profiles_filename)
simulated_jtf_nfw_profiles = np.load(simulated_jtf_nfw_profiles_filename)
if os.path.exists(sample_mc_percentiles_filename):
    sample_jtf_mc_pairs = np.load(sample_mc_percentiles_filename)
    if sample_jtf_mc_pairs.ndim == 3:
        percentiles = np.concatenate(
            (sample_jtf_mc_pairs[:, :, 0], sample_jtf_mc_pairs[:, :, 1]), axis=1
        )
        if os.path.exists(sample_mc_correlations_filename):
            correlations = np.load(sample_mc_correlations_filename)
        else:
            correlations = np.array(
                [
                    np.nan_to_num(
                        np.corrcoef(block.T)[0, 1], nan=0.0, posinf=0.0, neginf=0.0
                    )
                    for block in sample_jtf_mc_pairs
                ]
            )
        correlations = np.clip(correlations, -0.99, 0.99)
        sample_jtf_mc_pairs = np.column_stack((percentiles, correlations))
else:
    # Backwards compatibility: older simulation outputs only store the stack
    # medians, so reuse them when the new file is absent.
    sample_jtf_mc_pairs = sample_mc_pairs

inferrer = sbi_.gen_inferrer(infer_config["priors"], sample_mc_pairs.shape[1])
posterior = sbi_.train_inferrer(inferrer, sample_mc_pairs, simulated_nfw_profiles)
inferrer = sbi_.gen_inferrer(infer_config["priors"], sample_jtf_mc_pairs.shape[1])
posterior_range = sbi_.train_inferrer(
    inferrer, sample_jtf_mc_pairs, simulated_jtf_nfw_profiles
)

# Pickle posterior
pickle.dump(posterior, open(os.path.join(out_path, "posterior.pickle"), "wb"))
pickle.dump(posterior_range, open(os.path.join(out_path, "posterior_jtf.pickle"), "wb"))

log_runtime()
