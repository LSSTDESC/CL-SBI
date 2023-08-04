weaklensingclustersbi
========================

This repository contains tools for experiments for Weak Lensing Galaxy Clusters using Simulation Based Inference.

---------------

Summary:

We've split our workflow into four parts:

1) Simulation (/configs/simulations) - sim_config tells us about how our simulations were generated/obtained (e.g. the mass-concentration relation used, how much scatter the m-c relation has, the log10mass range, etc).
2) Observation(/configs/observations) - obs_config tells us about how our observations were generated/obtained (e.g. the richness-mass relation used, the m-c relation used, the scatter for each of those relations, the richness range, the number of observations, etc). 
3) Inference (/configs/inference) - infer_config tells us about how our inference was done (e.g. our priors, etc.)
4) Plotting - we produce plots using both ChainConsumer and PyGTC.

For each step, the script reading the config will save the output files to the corresponding outputs directory as the config.

---------------

Running instructions:

1) Create a simulation config in the ``configs/simulations`` directory

2) Generate simulations using the following script: 
	``python scripts/gen_simulations.py --sim_id {SIM_ID} --num_sims {NUM_SIMS}``. 
	
	This will output to the ``outputs/simulations/{SIM_ID}.{NUM_SIMS}`` directory

3) Create an inference config in the ``configs/inference`` directory

4) Generate a posterior using the following script: 
	``python scripts/gen_posterior.py --sim_id {SIM_ID} --infer_id {INFER_ID} --num_sims {NUM_SIMS}``.
	
	This will output to the ``outputs/posteriors/{SIM_ID}.{INFER_ID}.{NUM_SIMS}`` directory

5) Create an observation config in the ``configs/observations`` directory

6) Generate observations using the following script: 
	``python scripts/gen_observations.py --obs_id {OBS_ID} --num_obs {NUM_OBS}``. 
	
	This will output to the ``outputs/observations/{OBS_ID}.{NUM_OBS}`` directory

7) Run inference using the following script:
	``python scripts/run_inference.py --sim_id {SIM_ID} --infer_id {INFER_ID} --obs_id {OBS_ID} --num_sims {NUM_SIMS} --num_obs {NUM_OBS}``.
	
	This will output to the ``outputs/inference/{SIM_ID}.{INFER_ID}.{OBS_ID}.{NUM_SIMS}.{NUM_OBS}`` directory

8) Plot the contours from the SBI/MCMC chains using the following script:
	``python scripts/plot_chains.py --sim_id {SIM_ID} --infer_id {INFER_ID} --obs_id {OBS_ID} --num_sims {NUM_SIMS} --num_obs {NUM_OBS}``.
	
	This will output to the ``outputs/plots/{SIM_ID}.{INFER_ID}.{OBS_ID}.{NUM_SIMS}.{NUM_OBS}`` directory

Note: any of the above steps of the pipeline can be run with a --regenerate flag to overwrite existing output (with the exact same input parameters) for that step of the pipeline. This is a time saving step so we don't unnecessarily regenerate some output that we already have.


---------------

Bash script running instructions:

1) Modify bash_run.sh with whatever combinations of {SIM_ID}, {INFER_ID}, {OBS_ID}, {NUM_SIMS}, and {NUM_OBS} you'd like to experiment with
2) Navigate to the ``CL-SBI/scripts/`` directory and run ``./bash_run.sh``
