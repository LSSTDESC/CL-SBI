"""
Generate observations upon which we will run inference
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weaklensclustersbi.simulations import population, wlprofile
import numpy as np
import json
import os
import argparse
import time
from runtime_log import append_runtime_log

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument("--obs_id")
parser.add_argument("--num_obs")

# Add regenerate flag if we want to overwrite any existing observations.
# If false or not set, skip observation generation if they already exist from an earlier run.
parser.add_argument("--regenerate", action="store_true")
args = parser.parse_args()

script_start = time.perf_counter()


def log_runtime(status="success", details=""):
    append_runtime_log(
        stage="gen_observations",
        seconds=time.perf_counter() - script_start,
        obs_id=args.obs_id,
        num_obs=args.num_obs,
        details=details,
        status=status,
    )


# Open the copy of obs_config with the specified obs_id
script_dir = os.path.dirname(__file__)
config_rel_path = "../configs/observations/"
config_path = os.path.join(script_dir, config_rel_path)
config_filename = os.path.join(config_path, f"{args.obs_id}.json")

out_rel_path = f"../outputs/observations/{args.obs_id}.{args.num_obs}"
out_path = os.path.join(script_dir, out_rel_path)

# Checking if observations already exist from an earlier script run
if os.path.isfile(os.path.join(out_path, "drawn_nfw_profiles.npy")):
    # Regenerating observations (continuing script)
    if args.regenerate:
        print("Overwriting existing observations because of --regenerate flag")
    # Using existing observations (terminating script)
    else:
        print(
            "Observations already exist. If you want to regenerate, re-run with the --regenerate flag"
        )
        log_runtime(status="skipped", details="existing observations")
        quit()

# Open the copy of obs_config in obs_dir
with open(config_filename, "r") as f:
    obs_config = json.load(f)

rbins = 10 ** np.arange(0, obs_config["num_radial_bins"] / 10, 0.1)

# These are the ~10 log10masses that we've drawn from our richness bin of interest
drawn_mc_pairs = population.gen_mc_pairs_in_richness_bin(
    obs_config["min_richness"],
    obs_config["max_richness"],
    rm_relation=obs_config["rm_relation"],
    mc_relation=obs_config["mc_relation"],
    num_samples=int(args.num_obs),
    mc_scatter=obs_config["mc_scatter"],
    rm_scatter=obs_config["rm_scatter"],
    min_z=obs_config["min_z"],
    max_z=obs_config["max_z"],
    richness_contam_frac=obs_config.get("richness_contam_frac", 0.0),
    lambda_min_contam=obs_config.get("min_richness_contam", None),
)

z_sample = np.random.uniform(
    obs_config["min_z"], obs_config["max_z"], size=int(args.num_obs)
)
noiseless_drawn_nfw_profiles = np.array(
    [
        wlprofile.simulate_nfw(log10mass, concentration, rbins, z)
        for log10mass, concentration, z in np.column_stack((drawn_mc_pairs, z_sample))
    ]
)

drawn_nfw_profiles = population.calculate_noise(
    noiseless_drawn_nfw_profiles, obs_config["profile_noise_dex"]
)

# Observational error is defined as the standard deviation of the observable for each radial bin
sigmas = np.std(drawn_nfw_profiles, axis=0)
log_sigmas = np.std(np.log10(drawn_nfw_profiles), axis=0)

# Observables to logspace
drawn_nfw_profiles = np.log10(drawn_nfw_profiles)
noiseless_drawn_nfw_profiles = np.log10(noiseless_drawn_nfw_profiles)

# Output to intermediate files in obs_dir to be read by inference example script
if not os.path.exists(out_path):
    os.makedirs(out_path)
np.save(
    os.path.join(out_path, "noiseless_drawn_nfw_profiles.npy"),
    noiseless_drawn_nfw_profiles,
)
np.save(os.path.join(out_path, "drawn_nfw_profiles.npy"), drawn_nfw_profiles)
np.save(os.path.join(out_path, "drawn_mc_pairs.npy"), drawn_mc_pairs)
# np.save(os.path.join(out_path, "sigmas.npy"), sigmas)
np.save(os.path.join(out_path, "sigmas.npy"), log_sigmas)

log_runtime()
