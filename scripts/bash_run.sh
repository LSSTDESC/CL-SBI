#!/bin/bash

run_config() {
	SIM_ID=$1
	INFER_ID=$2
	OBS_ID=$3
	NUM_SIMS=$4
	NUM_OBS=$5
	python3 gen_simulations.py --sim_id $SIM_ID --num_sims $NUM_SIMS
	python3 gen_observations.py --obs_id $OBS_ID --num_obs $NUM_OBS
	python3 gen_posterior.py --sim_id $SIM_ID --infer_id $INFER_ID --num_sims $NUM_SIMS
	python3 run_inference.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS
	python3 plot_chains.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS
	python3 plot_diagnostics.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS

}

# # Add lines below for any configuration to run
# #  run_config	"{SIM_ID}"	"{INFER_ID}"	"{OBS_ID}"	{NUM_SIMS}	{NUM_OBS}


# # CONFIG 1: Ideal
# 	# Simulations: minimal scatter
# 	# Observations: no scatter, no profile noise
# 	# Inference: wide priors
# run_config	"sim_1"	"infer_1"	"obs_1"	10000	10

# # CONFIG 2: Config 1 + 0.01 NFW noise (obs)
# 	# Simulations: minimal scatter
# 	# Observations: no scatter, minimal profile noise
# 	# Inference: wide priors
# run_config	"sim_1"	"infer_1"	"obs_2"	10000	10

# # CONFIG 3: Config 2 + 0.1 r-m scatter (obs)
# 	# Simulations: minimal scatter
# 	# Observations: no m-c scatter, minimal r-m scatter, minimal profile noise
# 	# Inference: wide priors
# run_config	"sim_1"	"infer_1"	"obs_3"	10000	10

# # CONFIG 4: Config 3 + 0.1 m-c scatter (obs)
# 	# Simulations: minimal scatter
# 	# Observations: minimal m-c scatter, minimal r-m scatter, minimal profile noise
# 	# Inference: wide priors
# run_config	"sim_1"	"infer_1"	"obs_4"	10000	10

# # [VERY SLOW] CONFIG 5: Config 4 + 100 observations
# 	# Simulations: minimal scatter
# 	# Observations: minimal m-c scatter, minimal r-m scatter, minimal profile noise
# 	# Inference: wide priors
# # run_config	"sim_1"	"infer_1"	"obs_5"	10000	100

# # CONFIG 6: Config 1 + Wrong m-c model (obs)
# 	# Simulations: minimal scatter
# 	# Observations: different m-c model, no noise/scatter
# 	# Inference: wide priors
# run_config	"sim_1"	"infer_1"	"obs_6"	10000	10


# # [DOESN'T CONVERGE] CONFIG 7: Config 5 + 0.1 NFW noise (obs)
# 	# Simulations: minimal scatter
# 	# Observations: no scatter, minimal profile noise
# 	# Inference: wide priors
# # run_config	"sim_1"	"infer_1"	"obs_7"	10000	10

# # CONFIG 8: Config 7 + 0.1 NFW noise (sim)
# 	# Simulations: minimal scatter, minimal profile noise
# 	# Observations: no scatter, minimal profile noise
# 	# Inference: wide priors
# run_config	"sim_2"	"infer_1"	"obs_7"	10000	10

# # CONFIG 9: Config 1 + 0.1 NFW noise (sim)
# 	# Simulations: minimal scatter
# 	# Observations: no scatter, no profile noise
# 	# Inference: wide priors
# run_config	"sim_2"	"infer_1"	"obs_1"	10000	10

# # CONFIG 10: Config 2 + 0.01 NFW noise (sim)
# 	# Simulations: minimal scatter
# 	# Observations: no scatter, minimal profile noise
# 	# Inference: wide priors
# run_config	"sim_2"	"infer_1"	"obs_2"	10000	10

# # CONFIG 11: Config 3 + 0.01 NFW noise (sim)
# 	# Simulations: minimal scatter
# 	# Observations: no m-c scatter, minimal r-m scatter, minimal profile noise
# 	# Inference: wide priors
# run_config	"sim_2"	"infer_1"	"obs_3"	10000	10

# # CONFIG 12: Config 4 + 0.01 NFW noise (sim)
# 	# Simulations: minimal scatter
# 	# Observations: minimal m-c scatter, minimal r-m scatter, minimal profile noise
# 	# Inference: wide priors
# run_config	"sim_2"	"infer_1"	"obs_4"	10000	10

# # [VERY SLOW] CONFIG 13: Config 5 + 0.01 NFW noise (sim)
# 	# Simulations: minimal scatter
# 	# Observations: minimal m-c scatter, minimal r-m scatter, minimal profile noise
# 	# Inference: wide priors
# # run_config	"sim_2"	"infer_1"	"obs_5"	10000	100

# # CONFIG 14: Config 6 + 0.01 NFW noise (sim)
# 	# Simulations: minimal scatter
# 	# Observations: different m-c model, no noise/scatter
# 	# Inference: wide priors
# run_config	"sim_2"	"infer_1"	"obs_6"	10000	10

# # CONFIG 15: Config 7 + few observations
# 	# Simulations: minimal scatter
# 	# Observations: different m-c model, no noise/scatter
# 	# Inference: wide priors
# run_config	"sim_2"	"infer_1"	"obs_6"	10000	5
