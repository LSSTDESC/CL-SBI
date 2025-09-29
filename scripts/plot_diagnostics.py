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

# Open the infer_dir specified in the command line
script_dir = os.path.dirname(__file__)
infer_rel_path = f"../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}"
infer_path = os.path.join(script_dir, infer_rel_path)

out_rel_path = f"../outputs/plots/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}/diagnostics"
out_path = os.path.join(script_dir, out_rel_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Checking if plots already exist from an earlier script run
if os.path.isfile(os.path.join(out_path, "mcmc_cc.png")):
    # Regenerating plots (continuing script)
    if args.regenerate:
        print("Overwriting existing diagnostic plots because of --regenerate flag")
    # Using existing plots (terminating script)
    else:
        print(
            "Diagnostic plots already exist. If you want to regenerate, re-run with the --regenerate flag"
        )
        quit()

with open(os.path.join(infer_path, "mcmc_jtf_sampler.pickle"), "rb") as handle:
    mcmc_jtf_sampler = pickle.load(handle)
with open(os.path.join(infer_path, "mcmc_ftj_samplers.pickle"), "rb") as handle:
    mcmc_ftj_samplers = pickle.load(handle)
# with open(os.path.join(infer_path, "sbi_ftj_chains.pickle"), "rb") as handle:
#     sbi_ftj_chains = pickle.load(handle)

# Load SBI inferred m-c pairs
with open(os.path.join(infer_path, "sbi_ftj_mc.pickle"), "rb") as handle:
    sbi_ftj_mc = pickle.load(handle)
with open(os.path.join(infer_path, "sbi_jtf_mc.pickle"), "rb") as handle:
    sbi_jtf_mc = pickle.load(handle)

with open(os.path.join(infer_path, "mcmc_chains.pickle"), "rb") as handle:
    mcmc_chains = pickle.load(handle)
with open(os.path.join(infer_path, "sbi_chains.pickle"), "rb") as handle:
    sbi_chains = pickle.load(handle)

with open(os.path.join(infer_path, "sbi_jtf_logprob.pickle"), "rb") as handle:
    sbi_jtf_logprob = pickle.load(handle)
with open(os.path.join(infer_path, "sbi_ftj_logprob.pickle"), "rb") as handle:
    sbi_ftj_logprob = pickle.load(handle)

true_param_median = np.load(os.path.join(infer_path, "true_param_median.npy"))

# Plot the walkers for jtf sampler
plotutils.plot_walkers(mcmc_jtf_sampler, out_path, "mcmc_jtf_")

# Load observations
obs_rel_path = f"../outputs/observations/{args.obs_id}.{args.num_obs}"
obs_path = os.path.join(script_dir, obs_rel_path)

obs_config_rel_path = "../configs/observations/"
obs_config_path = os.path.join(script_dir, obs_config_rel_path)
obs_config_filename = os.path.join(obs_config_path, f"{args.obs_id}.json")
with open(obs_config_filename, "r") as f:
    obs_config = json.load(f)

drawn_mc_pairs_filename = os.path.join(obs_path, "drawn_mc_pairs.npy")
drawn_nfw_profiles_filename = os.path.join(obs_path, "drawn_nfw_profiles.npy")
sigmas_filename = os.path.join(obs_path, "sigmas.npy")
drawn_mc_pairs = np.load(drawn_mc_pairs_filename)
drawn_nfw_profiles = np.load(drawn_nfw_profiles_filename)
sigmas = np.load(sigmas_filename)

noiseless_drawn_nfw_profiles_filename = os.path.join(
    obs_path, "noiseless_drawn_nfw_profiles.npy"
)
noiseless_drawn_nfw_profiles = np.load(noiseless_drawn_nfw_profiles_filename)

# Load posterior
posterior_rel_path = f"../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}.{args.num_obs}"
posterior_path = os.path.join(script_dir, posterior_rel_path)
posterior_filename = os.path.join(posterior_path, "posterior.pickle")
with open(posterior_filename, "rb") as handle:
    posterior = pickle.load(handle)

# Plotting drawn m-c pairs
plotutils.plot_mc_pairs(drawn_mc_pairs, obs_path)

z = (obs_config["min_z"] + obs_config["max_z"]) / 2

# Plotting drawn NFW profiles in observations directory
plotutils.plot_nfw_profiles(
    drawn_nfw_profiles,
    sigmas,
    obs_path,
    obs_config["num_radial_bins"],
    obs_config["min_richness"],
    obs_config["max_richness"],
    z,
    is_noisy=True,
    true_param_median=true_param_median,
)

plotutils.plot_nfw_profiles(
    noiseless_drawn_nfw_profiles,
    None,
    obs_path,
    obs_config["num_radial_bins"],
    obs_config["min_richness"],
    obs_config["max_richness"],
    z,
    is_noisy=False,
    true_param_median=true_param_median,
)

# Plotting drawn AND inferred profiles in plots directory
# plotutils.plot_nfw_profiles(
#     drawn_nfw_profiles,
#     sigmas,
#     out_path,
#     obs_config["num_radial_bins"],
#     obs_config["min_richness"],
#     obs_config["max_richness"],
#     z,
#     is_noisy=True,
#     mcmc_chains=mcmc_chains,
#     sbi_chains=sbi_chains,
#     mcmc_jtf_sampler=mcmc_jtf_sampler,
#     mcmc_ftj_samplers=mcmc_ftj_samplers,
#     sbi_ftj_mc=sbi_ftj_mc,
#     sbi_jtf_mc=sbi_jtf_mc,
# )

plotutils.plot_mcmc_nfw_profiles(
    drawn_nfw_profiles,
    sigmas,
    out_path,
    obs_config["num_radial_bins"],
    obs_config["min_richness"],
    obs_config["max_richness"],
    z,
    mcmc_chains=mcmc_chains,
    mcmc_jtf_sampler=mcmc_jtf_sampler,
    mcmc_ftj_samplers=mcmc_ftj_samplers,
    true_param_median=true_param_median,
)

plotutils.plot_sbi_nfw_profiles(
    drawn_nfw_profiles,
    sigmas,
    out_path,
    obs_config["num_radial_bins"],
    obs_config["min_richness"],
    obs_config["max_richness"],
    z,
    sbi_chains=sbi_chains,
    sbi_ftj_mc=sbi_ftj_mc,
    sbi_jtf_mc=sbi_jtf_mc,
    sbi_ftj_logprob=sbi_ftj_logprob,
    sbi_jtf_logprob=sbi_jtf_logprob,
    true_param_median=true_param_median,
)

plotutils.plot_frac_diff(
    drawn_nfw_profiles,
    sigmas,
    out_path,
    obs_config["num_radial_bins"],
    obs_config["min_richness"],
    obs_config["max_richness"],
    z,
    mcmc_chains=mcmc_chains,
    mcmc_jtf_sampler=mcmc_jtf_sampler,
    mcmc_ftj_samplers=mcmc_ftj_samplers,
    sbi_chains=sbi_chains,
    # sbi_ftj_mc=sbi_ftj_mc,
    # sbi_jtf_mc=sbi_jtf_mc,
    true_param_median=true_param_median,
    sbi_ftj_logprob=sbi_ftj_logprob,
    sbi_jtf_logprob=sbi_jtf_logprob,
)

# plotutils.plot_nfw_profiles(
#     noiseless_drawn_nfw_profiles,
#     out_path,
#     obs_config['num_radial_bins'],
#     obs_config["min_richness"],
#     obs_config["max_richness"],
#     is_noisy=False,
#     mcmc_chains=mcmc_chains,
#     sbi_chains=sbi_chains,
# )

# plotutils.plot_ppc(
#     drawn_nfw_profiles,
#     np.median(drawn_mc_pairs, axis=0),
#     posterior,
#     out_path,
#     # TODO: this should be from the infer config i think?
#     obs_config['num_radial_bins'],
#     z,
# )
