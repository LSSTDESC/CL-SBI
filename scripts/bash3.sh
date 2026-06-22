#!/bin/bash

run_config() {
	SIM_ID=$1
	INFER_ID=$2
	OBS_ID=$3
	NUM_SIMS=$4
	NUM_OBS=$5

	echo -e "Running experiment:\t$SIM_ID\t\t$INFER_ID\t\t$OBS_ID\t\t$NUM_SIMS\t\t$NUM_OBS"

	# python3 gen_simulations.py --sim_id $SIM_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS #--regenerate
	# python3 gen_observations.py --obs_id $OBS_ID --num_obs $NUM_OBS #--regenerate
	# python3 train_inferrer.py --sim_id $SIM_ID --infer_id $INFER_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS #--regenerate
	# python3 run_inference.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate --run_mcmc_stacked
	python3 plot_chains.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate
	python3 plot_diagnostics.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate
	# python3 plot_calibration.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --ftj-only #--regenerate

	echo -e "Completed experiment: $SIM_ID\t\t$INFER_ID\t\t$OBS_ID\t\t$NUM_SIMS\t\t$NUM_OBS\n\n\n"
}

# ### ADJUSTING NUM OBS ###

# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	10
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	10
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	10
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				10000	10
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				10000	10
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					10000	10
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			10000	10

# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	20
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	20
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	20
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				10000	20
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				10000	20
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					10000	20
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			10000	20

# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	50
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	50
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	50
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				10000	50
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				10000	50
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					10000	50
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			10000	50

# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	100
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	100
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	100
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				10000	100
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				10000	100
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					10000	100
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			10000	100

# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	200
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	200
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	200
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				10000	200
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				10000	200
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					10000	200
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			10000	200


# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	300
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	300
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	300
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				10000	300
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				10000	300
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					10000	300
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			10000	300

# ###### MAIN EXPERIMENTS ######
run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					10000	376
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	10000	376
# run_config	"sim_z1_high_rm_scatter"	"infer_z1_high_rm_scatter"	"obs_z1_lambda5_high_rm_scatter"	10000	376
run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			10000	376

# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				10000	376
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				10000	376
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					10000	376
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			10000	376
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_mc_scatter"	10000	376
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5"					10000	376

# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_rm_scatter"	10000	376
# run_config	"sim_z1_high_rm_scatter"	"infer_z1_high_rm_scatter"	"obs_z1_lambda5"					10000	376
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_low_richness_contam"					10000	376
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_richness_contam"					10000	376

# # ##############################

# # ### ADJUSTING NUM SIMS ###
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					1000	10
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	1000	10
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			1000	10
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				1000	10
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				1000	10
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					1000	10
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			1000	10

# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5"					100000	10
# run_config	"sim_z1_high_mc_scatter"	"infer_z1_high_mc_scatter"	"obs_z1_lambda5_high_mc_scatter"	100000	10
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5_high_noise"			100000	10
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_prada"				100000	10
# run_config	"sim_z1"        			"infer_z1"					"obs_z1_lambda5_ludlow"				100000	10
# run_config	"sim_z1_high_noise"			"infer_z1"					"obs_z1_lambda5"					100000	10
# run_config	"sim_z1"					"infer_z1"					"obs_z1_lambda5_high_noise"			100000	10