#!/usr/bin/env bash
# M=50 SBI-only sampling-robustness ensemble for all 16 ΔΣ experiments.
# (MCMC ensemble is a separate overnight run: add --run_mcmc --n_mcmc N.)
cd "$(dirname "$0")"
M=${1:-50}; EXTRA="${2:-}"
echo "### SBI ensemble (M=$M) started $(date) ###"
run(){ echo "== $1/$2/$3 =="; python3 run_ensemble.py --sim_id "$1" --infer_id "$2" --obs_id "$3" --num_sims 10000 --num_obs 376 --observable delta_sigma --n_realizations $M $EXTRA 2>&1 | grep -E "saved ->|ERROR|Error|Traceback"; }
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
echo "### SBI ensemble DONE $(date) ###"
