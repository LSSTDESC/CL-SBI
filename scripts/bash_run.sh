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
}

# Add lines below for any configuration to run
#  run_config	"{SIM_ID}"	"{INFER_ID}"	"{OBS_ID}"	{NUM_SIMS}	{NUM_OBS}
run_config	"sim_1"	"infer_1"	"obs_1"	1000	10
run_config	"sim_1"	"infer_1"	"obs_2"	1000	10
