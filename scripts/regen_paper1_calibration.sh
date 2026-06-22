#!/bin/bash
# Regenerate calibration coverage plots + KS tables (stable N_c-subsampled KS) for all 16 configs.
cd "$(dirname "$0")"
run() {
    DIR="../outputs/inference/${1}.${2}.${3}.10000.376"
    [ -d "$DIR" ] || { echo "SKIP: $1 | $3"; return; }
    echo "=== calibration: $1 | $2 | $3 ==="
    python3 plot_calibration.py --sim_id $1 --infer_id $2 --obs_id $3 --num_sims 10000 --num_obs 376 --regenerate --ftj-only
}
run sim_z1                 infer_z1                 obs_z1_lambda5
run sim_z1                 infer_z1                 obs_z1_lambda5_high_mc_scatter
run sim_z1                 infer_z1                 obs_z1_lambda5_high_noise
run sim_z1                 infer_z1                 obs_z1_lambda5_high_richness_contam
run sim_z1                 infer_z1                 obs_z1_lambda5_high_rm_scatter
run sim_z1                 infer_z1                 obs_z1_lambda5_low_richness_contam
run sim_z1                 infer_z1                 obs_z1_lambda5_ludlow
run sim_z1                 infer_z1                 obs_z1_lambda5_prada
run sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5_high_mc_scatter
run sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5
run sim_z1_high_noise      infer_z1                 obs_z1_lambda5_high_noise
run sim_z1_high_noise      infer_z1                 obs_z1_lambda5
run sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5_high_rm_scatter
run sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5
run sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5_high_rm_scatter
run sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5
echo "ALL CALIBRATION DONE."
