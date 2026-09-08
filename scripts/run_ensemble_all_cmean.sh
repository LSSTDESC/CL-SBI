#!/usr/bin/env bash
# Corrected-mean ensemble sweep for the ensemble-recovery figure: SBI-only M=50 realization
# ensembles for all 16 experiment rows (baseline already has MCMC n=12 via the seeded rerun),
# then regenerate the figure and copy it into the paper's path. Detached-safe; resumable
# (run_ensemble skips completed realizations).
set -u
cd "$(dirname "$0")"
NS=10000; NO=376; PY=python3
run() { echo "=== ensemble $1/$2 x $3 $(date) ==="; $PY run_ensemble.py --sim_id "$1" --infer_id "$2" --obs_id "$3" --num_sims $NS --num_obs $NO --observable delta_sigma --stack_estimator corrected_mean --n_realizations 50; }
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
echo "=== all ensembles done; regenerating figure $(date) ==="
$PY plot_ensemble.py --observable delta_sigma --stack_estimator corrected_mean
cp ../outputs/plots/ensemble_recovery.delta_sigma.corrected_mean.pdf \
   ../tex_source/figures_new/ensemble/ensemble_recovery.delta_sigma.pdf \
   && echo "figure copied into paper path"
echo "=== ensemble sweep DONE $(date) ==="
