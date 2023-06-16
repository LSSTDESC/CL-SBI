weaklensingclustersbi
========================

This repository contains tools for experiments for Weak Lensing Galaxy Clusters using Simulation Based Inference.



---------------

Summary:

We've split our workflow into four parts:

1) Simulation (/configs/simulations) - sim_config tells us about how our simulations were generated/obtained (e.g. the mass-concentration relation used, the log10mass range, the number of simulations, the noise dex, etc).
2) Observation(/configs/observations) - obs_config tells us about how our observations were generated/obtained (e.g. the richness-mass relation used, the richness range, the number of observations, the noise dex, etc). 
3) Inference (/configs/inference) - infer_config tells us about how our inference was done (e.g. are we using SBI or MCMC, our priors, etc.)
4) Plotting - we produce plots using both ChainConsumer and PyGTC.

For each step, the script reading the config will save the output files to the same directory as the config (i.e. gen_observations script will output to the obs_dir). 

To run inference, you need to specify which sim_dir and which obs_dir to read intermediate files from, in addition to the infer_dir.


---------------

Running instructions:

SIMULATIONS:
First, generate the sim_config file in the directory of interest:
``python sim_config_example.py``
Then, use this sim_config to generate simulations in the sim_dir folder:
``python simulate_wl_profile.py --sim_dir example``

OBSERVATIONS:
First, generate the obs_config file in the directory of interest:
``python obs_config_example.py``
Then, use this obs_config to generate our observations in the obs_dir folder:
``python gen_observations.py --obs_dir example``

INFERENCE:
First, generate the infer_config file in the directory of interest:
``python infer_config_example.py``
Next, run inference using the sim_config in sim_dir and the infer_config in config_dir. This will output to the infer_config (so if you want to run the same inference with different simulations, you should specify multiple infer_configs for that).
``python mcmc_example.py --sim_dir example --infer_dir example --obs_dir example
python sbi_example.py --sim_dir example --infer_dir example --obs_dir example``

Finally, we can plot the results using pygtc and chainconsumer
``python plot_example.py --infer_dir example``
