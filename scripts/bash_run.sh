#!/bin/bash

run_config() {
	SIM_ID=$1
	INFER_ID=$2
	OBS_ID=$3
	NUM_SIMS=$4
	NUM_OBS=$5

	echo -e "Running experiment:\t$SIM_ID\t\t$INFER_ID\t\t$OBS_ID\t\t$NUM_SIMS\t\t$NUM_OBS"

	# python3 gen_simulations.py --sim_id $SIM_ID --num_sims $NUM_SIMS #--regenerate
	# python3 gen_observations.py --obs_id $OBS_ID --num_obs $NUM_OBS #--regenerate
	# python3 train_inferrer.py --sim_id $SIM_ID --infer_id $INFER_ID --num_sims $NUM_SIMS #--regenerate
	# python3 run_inference.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate
	python3 plot_chains.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate
	python3 plot_diagnostics.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate

	echo -e "Completed experiment: $SIM_ID\t\t$INFER_ID\t\t$OBS_ID\t\t$NUM_SIMS\t\t$NUM_OBS\n\n\n"

}

# Add lines below for any configuration to run
#  run_config	"{SIM_ID}"	"{INFER_ID}"	"{OBS_ID}"	{NUM_SIMS}	{NUM_OBS}

# # McClintock richness-redshift bins, all with near-ideal conditions
# run_config	"sim_z1"			"infer_z1"				"obs_z1_lambda1"	10000	10636
# run_config	"sim_z1"			"infer_z1"				"obs_z1_lambda2"	10000	2089
# run_config	"sim_z1"			"infer_z1"				"obs_z1_lambda3"	10000	1375
# run_config	"sim_z1"			"infer_z1"				"obs_z1_lambda4"	10000	10
# # run_config	"sim_z1"			"infer_z1"				"obs_z1_lambda5"	10000	10
# run_config	"sim_z1"			"infer_z1"				"obs_z1_lambda6"	10000	10
# run_config	"sim_z1"			"infer_z1"				"obs_z1_lambda7"	10000	10

# # run_config	"sim_z2"			"infer_z2"				"obs_z2_lambda1"	10000	18331
# # run_config	"sim_z2"			"infer_z2"				"obs_z2_lambda2"	10000	4135
# # run_config	"sim_z2"			"infer_z2"				"obs_z2_lambda3"	10000	2612
# run_config	"sim_z2"			"infer_z2"				"obs_z2_lambda4"	10000	10
# run_config	"sim_z2"			"infer_z2"				"obs_z2_lambda5"	10000	10
# run_config	"sim_z2"			"infer_z2"				"obs_z2_lambda6"	10000	10
# run_config	"sim_z2"			"infer_z2"				"obs_z2_lambda7"	10000	10

# # run_config	"sim_z3"			"infer_z3"				"obs_z3_lambda1"	10000	22991
# # run_config	"sim_z3"			"infer_z3"				"obs_z3_lambda2"	10000	4974
# # run_config	"sim_z3"			"infer_z3"				"obs_z3_lambda3"	10000	2927
# run_config	"sim_z3"			"infer_z3"				"obs_z3_lambda4"	10000	10
# run_config	"sim_z3"			"infer_z3"				"obs_z3_lambda5"	10000	10
# run_config	"sim_z3"			"infer_z3"				"obs_z3_lambda6"	10000	10
# run_config	"sim_z3"			"infer_z3"				"obs_z3_lambda7"	10000	10


# Add more noise/error/scatter to sims and obs. For more scatter, also add to MCMC priors
run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	376
run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	376
run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	376
run_config	"sim_z1_mid_mc_scatter"		"infer_z1_mid_mc_scatter"	"obs_z1_lambda5_mid_mc_scatter"		10000	376
run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_rm_scatter"	10000	376
run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_mid_rm_scatter"		10000	376


# Wrong m-c relation - out of distribution
run_config	"sim_z1"	"infer_z1"		"obs_z1_lambda5_prada"		10000	376
run_config	"sim_z1"	"infer_z1"		"obs_z1_lambda5_ludlow"		10000	376