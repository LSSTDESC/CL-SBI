from context import plotutils
import numpy as np
import argparse
import os
import pickle
import json

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument("--sim_id")
parser.add_argument("--infer_id")
parser.add_argument("--obs_id")
parser.add_argument("--num_sims")
parser.add_argument("--num_obs")

# Add regenerate flag if we want to overwrite any existing plots.
# If false or not set, skip plot generation if they already exist from an earlier run.
parser.add_argument("--regenerate", action="store_true")
args = parser.parse_args()

script_dir = os.path.dirname(__file__)

# Load observations
obs_rel_path = f"../outputs/observations/{args.obs_id}.{args.num_obs}"
obs_path = os.path.join(script_dir, obs_rel_path)
drawn_mc_pairs_filename = os.path.join(obs_path, "drawn_mc_pairs.npy")
drawn_mc_pairs = np.load(drawn_mc_pairs_filename)

# Load infer config
infer_config_rel_path = "../configs/inference/"
infer_config_path = os.path.join(script_dir, infer_config_rel_path)
infer_config_filename = os.path.join(infer_config_path, f"{args.infer_id}.json")
with open(infer_config_filename, "r") as f:
    infer_config = json.load(f)

# Open the infer_dir specified in the command line
infer_rel_path = f"../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}"
infer_path = os.path.join(script_dir, infer_rel_path)

with open(os.path.join(infer_path, "mcmc_chains.pickle"), "rb") as handle:
    mcmc_chains = pickle.load(handle)
with open(os.path.join(infer_path, "sbi_chains.pickle"), "rb") as handle:
    sbi_chains = pickle.load(handle)

true_param_median = np.load(os.path.join(infer_path, "true_param_median.npy"))
true_param_25 = np.load(os.path.join(infer_path, "true_param_25th_percentile.npy"))
true_param_75 = np.load(os.path.join(infer_path, "true_param_75th_percentile.npy"))

out_rel_path = f"../outputs/plots/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}"
out_path = os.path.join(script_dir, out_rel_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
# Checking if plots already exist from an earlier script run
if os.path.isfile(os.path.join(out_path, "sbi_cc.pdf")) or os.path.isfile(
    os.path.join(out_path, "sbi_cc_agg.pdf")
):
    # Regenerating plots (continuing script)
    if args.regenerate:
        print("Overwriting existing plots because of --regenerate flag")
    # Using existing plots (terminating script)
    else:
        print(
            "Plots already exist. If you want to regenerate, re-run with the --regenerate flag"
        )
        quit()


if "agg" not in infer_config:
    # TODO: add a wrapper function so we have a single interface with 'pygtc' or 'cc' as a param
    # plotutils.plot_pygtc(mcmc_chains, out_path, "mcmc", true_param_median)
    plotutils.plot_chainconsumer(
        mcmc_chains,
        out_path,
        "mcmc",
        [true_param_median, true_param_25, true_param_75],
        drawn_mc_pairs,
    )

    # plotutils.plot_pygtc(sbi_chains, out_path, "sbi", true_param_median)
    plotutils.plot_chainconsumer(
        sbi_chains,
        out_path,
        "sbi",
        [true_param_median, true_param_25, true_param_75],
        drawn_mc_pairs,
    )

    plotutils.plot_chainconsumer_combined(
        mcmc_chains,
        sbi_chains,
        out_path,
        [true_param_median, true_param_25, true_param_75],
    )
# else:
#     plotutils.plot_chainconsumer_agg(
#         sbi_agg_chains,
#         out_path,
#         # "sbi_agg",
#         [true_param_median, true_param_25, true_param_75],
#     )
