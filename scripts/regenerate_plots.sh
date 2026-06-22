#!/bin/bash
# Regenerate all plots for paper experiments

cd /Users/akumgill/Documents/GitHub/CL-SBI/scripts

run_plots() {
    SIM_ID=$1
    INFER_ID=$2
    OBS_ID=$3
    echo "=== Plotting $SIM_ID / $INFER_ID / $OBS_ID ==="
    python3 plot_chains.py --sim_id "$SIM_ID" --infer_id "$INFER_ID" --obs_id "$OBS_ID" --num_sims 10000 --num_obs 376 --regenerate 2>&1 | grep -v "INFO\|FutureWarning\|ArviZ\|warn("
    python3 plot_diagnostics.py --sim_id "$SIM_ID" --infer_id "$INFER_ID" --obs_id "$OBS_ID" --num_sims 10000 --num_obs 376 --regenerate 2>&1 | grep -v "INFO\|FutureWarning\|ArviZ\|warn("
}

# In-distribution
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5"
run_plots "sim_z1_high_noise" "infer_z1" "obs_z1_lambda5_high_noise"
run_plots "sim_z1_high_mc_scatter" "infer_z1_high_mc_scatter" "obs_z1_lambda5_high_mc_scatter"
run_plots "sim_z1_high_rm_scatter" "infer_z1_high_rm_scatter" "obs_z1_lambda5_high_rm_scatter"

# OOD
run_plots "sim_z1_high_noise" "infer_z1" "obs_z1_lambda5"
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_high_noise"
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_ludlow"
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_low_richness_contam"
run_plots "sim_z1_high_rm_scatter" "infer_z1_high_rm_scatter" "obs_z1_lambda5"
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_high_rm_scatter"

echo "Done!"
