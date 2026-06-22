#!/bin/bash
# Regenerate Paper I inference (MCMC + SBI eval) for all 16 experiment configs with the
# adaptive (autocorr-converged) MCMC sampler and the corrected richness-selection prior.
# Posteriors already exist, so this does NOT retrain SBI -- it re-runs inference only.
# Convergence status per run is printed by the adaptive sampler and captured in this log.
cd "$(dirname "$0")"

run() {
    echo "=================================================================="
    echo "REGEN: sim=$1 infer=$2 obs=$3"
    echo "=================================================================="
    python3 run_inference.py --sim_id "$1" --infer_id "$2" --obs_id "$3" \
        --num_sims "$4" --num_obs "$5" --regenerate
    echo ""
}

# All 16 experiment configurations (matches outputs/inference/*.10000.376)
run sim_z1                 infer_z1                 obs_z1_lambda5                      10000 376
run sim_z1                 infer_z1                 obs_z1_lambda5_high_mc_scatter      10000 376
run sim_z1                 infer_z1                 obs_z1_lambda5_high_noise           10000 376
run sim_z1                 infer_z1                 obs_z1_lambda5_high_richness_contam 10000 376
run sim_z1                 infer_z1                 obs_z1_lambda5_high_rm_scatter      10000 376
run sim_z1                 infer_z1                 obs_z1_lambda5_low_richness_contam  10000 376
run sim_z1                 infer_z1                 obs_z1_lambda5_ludlow               10000 376
run sim_z1                 infer_z1                 obs_z1_lambda5_prada                10000 376
run sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5_high_mc_scatter      10000 376
run sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5                      10000 376
run sim_z1_high_noise      infer_z1                 obs_z1_lambda5_high_noise           10000 376
run sim_z1_high_noise      infer_z1                 obs_z1_lambda5                      10000 376
run sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5_high_rm_scatter      10000 376
run sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5                      10000 376
run sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5_high_rm_scatter      10000 376
run sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5                      10000 376

echo "ALL DONE."
