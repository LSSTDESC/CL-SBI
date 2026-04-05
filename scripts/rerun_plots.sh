#!/bin/bash
# Rerun all paper plots with updated axis labels (Σ instead of ΔΣ)
# Skipping calibration plots as requested

cd "$(dirname "$0")"

run_plots() {
    SIM_ID=$1
    INFER_ID=$2
    OBS_ID=$3
    NUM_SIMS=$4
    NUM_OBS=$5

    # Check if inference output exists
    if [ ! -d "../outputs/inference/${SIM_ID}.${INFER_ID}.${OBS_ID}.${NUM_SIMS}.${NUM_OBS}" ]; then
        echo "SKIP (no data): $SIM_ID | $OBS_ID"
        return
    fi

    echo "==========================================="
    echo "Plotting: $SIM_ID | $OBS_ID"
    echo "==========================================="

    python3 plot_chains.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate
    python3 plot_diagnostics.py --sim_id $SIM_ID --infer_id $INFER_ID --obs_id $OBS_ID --num_sims $NUM_SIMS --num_obs $NUM_OBS --regenerate

    echo ""
}

echo "Starting plot regeneration with updated Σ labels..."
echo ""

# All experiments with 10000.376 outputs (based on existing data)
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5" 10000 376
run_plots "sim_z1_high_mc_scatter" "infer_z1_high_mc_scatter" "obs_z1_lambda5_high_mc_scatter" 10000 376
run_plots "sim_z1_high_mc_scatter" "infer_z1_high_mc_scatter" "obs_z1_lambda5" 10000 376
run_plots "sim_z1_high_noise" "infer_z1" "obs_z1_lambda5_high_noise" 10000 376
run_plots "sim_z1_high_noise" "infer_z1" "obs_z1_lambda5" 10000 376
run_plots "sim_z1_high_rm_scatter" "infer_z1_high_rm_scatter" "obs_z1_lambda5_high_rm_scatter" 10000 376
run_plots "sim_z1_high_rm_scatter" "infer_z1" "obs_z1_lambda5_high_rm_scatter" 10000 376
run_plots "sim_z1_high_rm_scatter" "infer_z1" "obs_z1_lambda5" 10000 376
run_plots "sim_z1" "infer_z1_high_mc_scatter" "obs_z1_lambda5_high_mc_scatter" 10000 376
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_high_mc_scatter" 10000 376
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_high_noise" 10000 376
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_high_richness_contam" 10000 376
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_high_rm_scatter" 10000 376
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_low_richness_contam" 10000 376
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_ludlow" 10000 376
run_plots "sim_z1" "infer_z1" "obs_z1_lambda5_prada" 10000 376

echo "==========================================="
echo "All plots regenerated!"
echo "==========================================="
