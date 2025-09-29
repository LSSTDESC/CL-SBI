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
parser.add_argument("--num_obs")

# Add regenerate flag if we want to overwrite any existing simulations.
# If false or not set, skip simulation generation if they already exist from an earlier run.
parser.add_argument("--regenerate", action="store_true")
args = parser.parse_args()

# Open the copy of sim_config with the specified sim_id
script_dir = os.path.dirname(__file__)
sim_config_rel_path = "../configs/simulations/"
sim_config_path = os.path.join(script_dir, sim_config_rel_path)
sim_config_filename = os.path.join(sim_config_path, f"{args.sim_id}.json")

out_rel_path = f"../outputs/simulations/{args.sim_id}.{args.num_sims}.{args.num_obs}"
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

rbins = 10 ** np.arange(0, sim_config["num_radial_bins"] / 10, 0.1)

# Richness bin ranges:[5, 10], [10, 14], [14, 20], [20, 30], [30, 45], [45, 60], [60, 100]
lambda_bins = [(5, 10), (10, 14), (14, 20), (20, 30), (30, 45), (45, 60), (60, 100)]
# lambda_bins_fuzzy = [
#     (5, 15),
#     (8, 16),
#     (12, 25),
#     (16, 35),
#     (25, 50),
#     (40, 70),
#     (50, 100),
# ]

num_sims = int(args.num_sims)
num_obs = int(args.num_obs)
# sims_per_bin = num_sims * num_obs // (len(z_bins) * len(lambda_bins))
sims_per_bin = num_sims * num_obs // len(lambda_bins)

all_sample_mc_pairs = []
all_simulated_nfw_profiles = []
all_non_noisy_simulated_nfw_profiles = []
all_jtf_simulated_nfw_profiles = []
# for min_z, max_z in z_bins:
for min_lambda, max_lambda in lambda_bins:
    # print(
    #     f"Simulating {sims_per_bin} m-c pairs in z bin [{min_z}, {max_z}], lambda bin [{min_lambda}, {max_lambda}]"
    # )

    # Generate sims_per_bin * num_obs m-c pairs
    sample_mc_pairs = population.gen_mc_pairs_in_richness_bin(
        min_lambda,
        max_lambda,
        rm_relation=sim_config["rm_relation"],
        mc_relation=sim_config["mc_relation"],
        num_samples=sims_per_bin,
        mc_scatter=sim_config["mc_scatter"],
        rm_scatter=sim_config["rm_scatter"],
        min_z=sim_config["min_z"],
        max_z=sim_config["max_z"],
    )

    z_sample = np.random.uniform(
        sim_config["min_z"], sim_config["max_z"], size=sims_per_bin
    )

    # # Apply filtering criteria to subselect mc_pairs
    # sample_mc_pairs = population.filter_mc_pairs(
    #     sample_mc_pairs, sim_config["mc_pair_subselect"]
    # )

    # Simulate NFW profiles for each of the mc_pairs
    non_noisy_simulated_nfw_profiles = np.array(
        [
            wlprofile.simulate_nfw(log10mass, concentration, rbins, z)
            for log10mass, concentration, z in np.column_stack(
                (sample_mc_pairs, z_sample)
            )
        ]
    )

    # Add a "fixed" amount of noise to each profile
    simulated_nfw_profiles = population.calculate_noise(
        non_noisy_simulated_nfw_profiles,
        sim_config["profile_noise_dex"],
    )

    # # Add a range of noise to each profile (used for join-then-fit)
    # simulated_nfw_profiles_range = population.calculate_noise_range(
    #     non_noisy_simulated_nfw_profiles,
    #     sim_config["profile_noise_dex"],
    # )

    # print(np.shape(simulated_nfw_profiles))
    # print(np.shape(simulated_nfw_profiles_range))
    # simulated_jtf_nfw_profiles = np.median(simulated_nfw_profiles, axis=0)

    # Simulations to logspace
    simulated_nfw_profiles = np.log10(simulated_nfw_profiles)
    # simulated_nfw_profiles_range = np.log10(simulated_nfw_profiles_range)

    for i in range(sims_per_bin // num_obs):
        # print(
        #     np.median(
        #         sample_mc_pairs[i * num_obs : (i + 1) * num_obs],
        #         axis=0,
        #     )
        # )
        all_sample_mc_pairs.append(
            np.median(
                sample_mc_pairs[i * num_obs : (i + 1) * num_obs],
                axis=0,
            )
        )
        all_simulated_nfw_profiles.append(
            simulated_nfw_profiles[i * num_obs : (i + 1) * num_obs]
        )
        all_non_noisy_simulated_nfw_profiles.append(
            non_noisy_simulated_nfw_profiles[i * num_obs : (i + 1) * num_obs]
        )
        all_jtf_simulated_nfw_profiles.append(
            np.median(
                simulated_nfw_profiles[i * num_obs : (i + 1) * num_obs],
                axis=0,
            )
            # simulated_nfw_profiles_range[i]  # * num_obs : (i + 1) * num_obs]
        )

# TODO: fill in remainder number of sims
# print(np.shape(all_sample_mc_pairs))
# print(np.shape(all_simulated_nfw_profiles))
# print(np.shape(all_jtf_simulated_nfw_profiles))

# For jtf, don't use the redshift-richness bins at all
# sample_mc_pairs = population.gen_mc_pairs_in_richness_bin(
#     sim_config["min_richness"],
#     sim_config["max_richness"],
#     rm_relation=sim_config["rm_relation"],
#     mc_relation=sim_config["mc_relation"],
#     num_samples=len(all_sample_mc_pairs),
#     mc_scatter=sim_config["mc_scatter"],
#     rm_scatter=sim_config["rm_scatter"],
#     min_z=sim_config["min_z"],
#     max_z=sim_config["max_z"],
# )
# z_sample = np.random.uniform(
#     sim_config["min_z"], sim_config["max_z"], size=len(all_sample_mc_pairs)
# )
# non_noisy_simulated_nfw_profiles = np.array(
#     [
#         wlprofile.simulate_nfw(log10mass, concentration, rbins, z)
#         for log10mass, concentration, z in np.column_stack((sample_mc_pairs, z_sample))
#     ]
# )
# simulated_nfw_profiles_range = population.calculate_noise_range(
#     non_noisy_simulated_nfw_profiles,
#     sim_config["profile_noise_dex"],
# )
# simulated_nfw_profiles_range = np.log10(simulated_nfw_profiles_range)

# Output to intermediate files in sim_dir to be read by inference example script
if not os.path.exists(out_path):
    os.makedirs(out_path)
np.save(
    os.path.join(out_path, "simulated_nfw_profiles.npy"), all_simulated_nfw_profiles
)
np.save(
    os.path.join(out_path, "simulated_jtf_nfw_profiles.npy"),
    all_jtf_simulated_nfw_profiles,
)
np.save(os.path.join(out_path, "sample_mc_pairs.npy"), all_sample_mc_pairs)
# np.save(os.path.join(out_path, "sample_jtf_mc_pairs.npy"), sample_mc_pairs)
