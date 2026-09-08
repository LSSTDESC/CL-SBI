import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plot import plotutils
import numpy as np
import argparse
import os
import pickle
import json
import time
from runtime_log import append_runtime_log

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument("--sim_id")
parser.add_argument("--infer_id")
parser.add_argument("--obs_id")
parser.add_argument("--num_sims")
parser.add_argument("--num_obs")
# Observable: "surface_density" (default) or "delta_sigma". Non-default reads/writes
# .delta_sigma-suffixed dirs so the Sigma figures are preserved.
parser.add_argument("--observable", default="surface_density")
parser.add_argument("--stack_estimator", default="median")

# Add regenerate flag if we want to overwrite any existing plots.
# If false or not set, skip plot generation if they already exist from an earlier run.
parser.add_argument("--regenerate", action="store_true")
args = parser.parse_args()

script_start = time.perf_counter()


def log_runtime(status="success", details=""):
    append_runtime_log(
        stage="plot_chains",
        seconds=time.perf_counter() - script_start,
        sim_id=args.sim_id,
        infer_id=args.infer_id,
        obs_id=args.obs_id,
        num_sims=args.num_sims,
        num_obs=args.num_obs,
        details=details,
        status=status,
    )

script_dir = os.path.dirname(__file__)
obs_suffix = "" if args.observable == "surface_density" else f".{args.observable}"
obs_suffix += "" if args.stack_estimator == "median" else f".{args.stack_estimator}"

# Truth reference = the TRUE population distribution (large seeded sample from the obs
# config), not the finite N_c observed draw. The posteriors claim to recover the intrinsic
# population dispersion, so the reference contour/crosshairs should be that population;
# the single N_c draw carries ~1/sqrt(N_c) finite-sample noise (Payerne P2-d). The data
# vector itself (and hence the posteriors) remains a single realization, as in a real survey.
from weaklensclustersbi.simulations import population as _population
obs_config_filename_truth = os.path.join(script_dir, f"../configs/observations/{args.obs_id}.json")
with open(obs_config_filename_truth, "r") as f:
    _obs_config = json.load(f)
np.random.seed(4242)  # fixed seed: reproducible reference population
drawn_mc_pairs = np.asarray(_population.gen_mc_pairs_in_richness_bin(
    _obs_config["min_richness"],
    _obs_config["max_richness"],
    rm_relation=_obs_config["rm_relation"],
    mc_relation=_obs_config["mc_relation"],
    num_samples=40000,  # large sample: smooth 95/99.7% truth contours (tails from 5k are jagged)
    mc_scatter=_obs_config["mc_scatter"],
    rm_scatter=_obs_config["rm_scatter"],
    min_z=_obs_config["min_z"],
    max_z=_obs_config["max_z"],
    richness_contam_frac=_obs_config.get("richness_contam_frac", 0.0),
    lambda_min_contam=_obs_config.get("min_richness_contam", None),
))

# Load infer config
infer_config_rel_path = "../configs/inference/"
infer_config_path = os.path.join(script_dir, infer_config_rel_path)
infer_config_filename = os.path.join(infer_config_path, f"{args.infer_id}.json")
with open(infer_config_filename, "r") as f:
    infer_config = json.load(f)

# Open the infer_dir specified in the command line
infer_rel_path = f"../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}{obs_suffix}"
infer_path = os.path.join(script_dir, infer_rel_path)

with open(os.path.join(infer_path, "mcmc_chains.pickle"), "rb") as handle:
    mcmc_chains = pickle.load(handle)
with open(os.path.join(infer_path, "sbi_chains.pickle"), "rb") as handle:
    sbi_chains = pickle.load(handle)

# Crosshair / percentile reference lines from the same population sample as the contour
# (previously loaded from the inference dir = the single N_c draw's median/percentiles).
true_param_median = (np.median(drawn_mc_pairs[:, 0]), np.median(drawn_mc_pairs[:, 1]))
true_param_25 = (np.percentile(drawn_mc_pairs[:, 0], 25), np.percentile(drawn_mc_pairs[:, 1], 25))
true_param_75 = (np.percentile(drawn_mc_pairs[:, 0], 75), np.percentile(drawn_mc_pairs[:, 1], 75))

out_rel_path = f"../outputs/plots/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}{obs_suffix}"
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
        log_runtime(status="skipped", details="existing plots")
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
        mc_pairs=drawn_mc_pairs,
    )

    # Load and plot two-stage FTJ comparison if available
    population_samples_file = os.path.join(infer_path, "mcmc_ftj_population_samples.pickle")
    population_params_file = os.path.join(infer_path, "mcmc_ftj_population_params.pickle")
    individual_samplers_file = os.path.join(infer_path, "mcmc_ftj_individual_samplers.pickle")
    if os.path.exists(population_samples_file):
        with open(population_samples_file, "rb") as handle:
            mcmc_ftj_population_samples = pickle.load(handle)

        # Load population params for displaying fitted values
        population_params = None
        if os.path.exists(population_params_file):
            with open(population_params_file, "rb") as handle:
                population_params = pickle.load(handle)

        # Load individual samplers and create naive stacked samples
        mcmc_ftj_naive_samples = None
        if os.path.exists(individual_samplers_file):
            with open(individual_samplers_file, "rb") as handle:
                individual_samplers = pickle.load(handle)
            # Concatenate flatchain from each individual sampler (only M, c columns)
            naive_chains = [s.flatchain[:, :2] for s in individual_samplers]
            mcmc_ftj_naive_samples = np.concatenate(naive_chains, axis=0)

        plotutils.plot_chainconsumer_ftj_comparison(
            mcmc_chains[1],  # FTJ joint likelihood chain
            mcmc_ftj_population_samples,  # FTJ two-stage population samples
            out_path,
            [true_param_median, true_param_25, true_param_75],
            drawn_mc_pairs,
            population_params=population_params,
            mcmc_ftj_naive_samples=mcmc_ftj_naive_samples,
        )

        # Plot all MCMC methods comparison (JTF, FTJ joint likelihood, FTJ two-stage, naive stacking)
        plotutils.plot_chainconsumer_mcmc_all_methods(
            mcmc_chains[0],  # JTF chain
            mcmc_chains[1],  # FTJ joint likelihood chain
            mcmc_ftj_population_samples,  # FTJ two-stage population samples
            out_path,
            [true_param_median, true_param_25, true_param_75],
            drawn_mc_pairs,
            mcmc_ftj_naive_samples=mcmc_ftj_naive_samples,
        )
# else:
#     plotutils.plot_chainconsumer_agg(
#         sbi_agg_chains,
#         out_path,
#         # "sbi_agg",
#         [true_param_median, true_param_25, true_param_75],
#     )

log_runtime()
