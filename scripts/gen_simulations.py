"""
Example script to simulate a weak lensing profile using modules
"""

from context import population, wlprofile
import numpy as np
import json
import os
import argparse

# Define a richness band from which we draw masses with some noise, and the
# concentration that scatters about that theoretical prediction

# Read command line arguments for the directory with the infer_config
parser = argparse.ArgumentParser()
parser.add_argument("--sim_id")
parser.add_argument("--num_sims")

# Add regenerate flag if we want to overwrite any existing simulations.
# If false or not set, skip simulation generation if they already exist from an earlier run.
parser.add_argument("--regenerate", action="store_true")
args = parser.parse_args()

# Open the copy of sim_config with the specified sim_id
script_dir = os.path.dirname(__file__)
sim_config_rel_path = "../configs/simulations/"
sim_config_path = os.path.join(script_dir, sim_config_rel_path)
sim_config_filename = os.path.join(sim_config_path, f"{args.sim_id}.json")

out_rel_path = f"../outputs/simulations/{args.sim_id}.{args.num_sims}"
out_path = os.path.join(script_dir, out_rel_path)

# Checking if simulations already exist from an earlier script run
if os.path.isfile(os.path.join(out_path, "simulated_nfw_profiles.npy")):
    # Regenerating simulations (continuing script)
    if args.regenerate:
        print("Overwriting existing simulation because of --regenerate flag")
    # Using existing simulations (terminating script)
    else:
        print(
            "Simulations already exist. If you want to regenerate, re-run with the --regenerate flag"
        )
        quit()

# Open the copy of sim_config in sim_dir
with open(sim_config_filename, "r") as f:
    sim_config = json.load(f)
"""
* simulated_nfw_profiles: needs to be 10k from randomly sampled log10masses in 
    range specified in sim_config and their corresponding concentrations
* drawn masses: should be a much smaller number ~10
"""

# # ~10k randomly sampled log10masses and their corresponding concentrations.
# # These are the "simulations" that we'll use for SBI
# sample_mc_pairs = population.random_mass_conc(
#     sim_config["min_log10mass"],
#     sim_config["max_log10mass"],
#     int(args.num_sims),
#     mc_scatter=sim_config["mc_scatter"],
#     mc_relation=sim_config["mc_relation"],
#     min_z=sim_config["min_z"],
#     max_z=sim_config["max_z"],
# )

sample_mc_pairs = population.gen_mc_pairs_in_richness_bin(
    sim_config["min_richness"],
    sim_config["max_richness"],
    rm_relation=sim_config["rm_relation"],
    mc_relation=sim_config["mc_relation"],
    num_samples=int(args.num_sims),
    mc_scatter=sim_config["mc_scatter"],
    rm_scatter=sim_config["rm_scatter"],
    min_z=sim_config["min_z"],
    max_z=sim_config["max_z"],
)

z_sample = np.random.uniform(
    sim_config["min_z"], sim_config["max_z"], size=int(args.num_sims)
)

# Apply filtering criteria to subselect mc_pairs
filtered_mc_pairs = population.filter_mc_pairs(
    sample_mc_pairs, sim_config["mc_pair_subselect"]
)

# Simulate NFW profiles for each of the mc_pairs
rbins = 10 ** np.arange(0, sim_config["num_radial_bins"] / 10, 0.1)
non_noisy_simulated_nfw_profiles = np.array(
    [
        wlprofile.simulate_nfw(log10mass, concentration, rbins, z)
        for log10mass, concentration, z in np.column_stack(
            (filtered_mc_pairs, z_sample)
        )
    ]
)

simulated_nfw_profiles = population.calculate_noise(
    # select first half of non_noisy_simulated_nfw_profiles to add noise to
    non_noisy_simulated_nfw_profiles,
    sim_config["profile_noise_dex"],
)

simulated_nfw_profiles_range = population.calculate_noise_range(
    non_noisy_simulated_nfw_profiles,
    sim_config["profile_noise_dex"],
)

# Output to intermediate files in sim_dir to be read by inference example script
if not os.path.exists(out_path):
    os.makedirs(out_path)
np.save(os.path.join(out_path, "simulated_nfw_profiles.npy"), simulated_nfw_profiles)
np.save(
    os.path.join(out_path, "simulated_nfw_profiles_range.npy"),
    simulated_nfw_profiles_range,
)
np.save(os.path.join(out_path, "sample_mc_pairs.npy"), filtered_mc_pairs)
