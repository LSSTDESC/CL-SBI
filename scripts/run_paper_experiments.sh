#!/bin/bash
# Run all paper experiments with corrected joint likelihood (no /N division)
# Usage: cd scripts && ./run_paper_experiments.sh

set -e

run_exp() {
    SIM_ID=$1
    INFER_ID=$2
    OBS_ID=$3
    echo "=========================================="
    echo "Running: $SIM_ID / $INFER_ID / $OBS_ID"
    echo "=========================================="
    python3 run_inference.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims 10000 --num_obs 376 --regenerate
    echo "Completed: $SIM_ID / $INFER_ID / $OBS_ID"
    echo ""
}

echo "Starting paper experiments at $(date)"
echo ""

# In-distribution experiments
echo "=== IN-DISTRIBUTION EXPERIMENTS ==="
run_exp "sim_z1" "infer_z1" "obs_z1_lambda5"
run_exp "sim_z1_high_noise" "infer_z1" "obs_z1_lambda5_high_noise"
run_exp "sim_z1_high_mc_scatter" "infer_z1_high_mc_scatter" "obs_z1_lambda5_high_mc_scatter"
run_exp "sim_z1_high_rm_scatter" "infer_z1_high_rm_scatter" "obs_z1_lambda5_high_rm_scatter"

# OOD experiments
echo "=== OUT-OF-DISTRIBUTION EXPERIMENTS ==="
run_exp "sim_z1_high_noise" "infer_z1" "obs_z1_lambda5"
run_exp "sim_z1" "infer_z1" "obs_z1_lambda5_high_noise"
run_exp "sim_z1" "infer_z1" "obs_z1_lambda5_ludlow"
run_exp "sim_z1" "infer_z1" "obs_z1_lambda5_low_richness_contam"
run_exp "sim_z1_high_rm_scatter" "infer_z1_high_rm_scatter" "obs_z1_lambda5"
run_exp "sim_z1" "infer_z1" "obs_z1_lambda5_high_rm_scatter"

echo ""
echo "All experiments completed at $(date)"
