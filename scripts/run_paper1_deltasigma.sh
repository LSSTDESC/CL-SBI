#!/bin/bash
# Full Paper-I pipeline rebuilt on the DeltaSigma (excess surface density) observable.
# Mirrors the 16 experiments in regen_paper1_mcmc.sh, but generates sims/observations,
# trains SBI posteriors, and runs inference all with --observable delta_sigma.
#
# Outputs are written to .delta_sigma-suffixed directories, so the existing Sigma
# outputs are preserved untouched for the observable A/B comparison.
#
# Resumable: stages skip if their output already exists (no --regenerate), so this
# script can be safely re-launched if interrupted.
#
# Usage: cd scripts && ./run_paper1_deltasigma.sh   (best run in the background / nohup)
set -u
cd "$(dirname "$0")"

NS=10000
NO=376
OBSV=delta_sigma
PY=python3

echo "############ Paper-I DeltaSigma rebuild started $(date) ############"

echo "===== STAGE 1/4: generate training simulations (4 sim sets) ====="
for SIM in sim_z1 sim_z1_high_mc_scatter sim_z1_high_noise sim_z1_high_rm_scatter; do
    echo "--- gen_simulations $SIM ---"
    $PY gen_simulations.py --sim_id "$SIM" --num_sims $NS --num_obs $NO --observable $OBSV
done

echo "===== STAGE 2/4: generate observation mocks (8 obs sets) ====="
for OBS in obs_z1_lambda5 obs_z1_lambda5_high_mc_scatter obs_z1_lambda5_high_noise \
           obs_z1_lambda5_high_richness_contam obs_z1_lambda5_high_rm_scatter \
           obs_z1_lambda5_low_richness_contam obs_z1_lambda5_ludlow obs_z1_lambda5_prada; do
    echo "--- gen_observations $OBS ---"
    $PY gen_observations.py --obs_id "$OBS" --num_obs $NO --observable $OBSV
done

echo "===== STAGE 3/4: train SBI posteriors (5 sim/infer pairs) ====="
train() { echo "--- train_inferrer sim=$1 infer=$2 ---"; $PY train_inferrer.py --sim_id "$1" --infer_id "$2" --num_sims $NS --num_obs $NO --observable $OBSV; }
train sim_z1                 infer_z1
train sim_z1_high_mc_scatter infer_z1_high_mc_scatter
train sim_z1_high_noise      infer_z1
train sim_z1_high_rm_scatter infer_z1_high_rm_scatter
train sim_z1_high_rm_scatter infer_z1

echo "===== STAGE 4/4: run inference (16 experiments) ====="
infer() {
    echo "--- run_inference sim=$1 infer=$2 obs=$3 ---"
    $PY run_inference.py --sim_id "$1" --infer_id "$2" --obs_id "$3" --num_sims $NS --num_obs $NO --observable $OBSV
}
infer sim_z1                 infer_z1                 obs_z1_lambda5
infer sim_z1                 infer_z1                 obs_z1_lambda5_high_mc_scatter
infer sim_z1                 infer_z1                 obs_z1_lambda5_high_noise
infer sim_z1                 infer_z1                 obs_z1_lambda5_high_richness_contam
infer sim_z1                 infer_z1                 obs_z1_lambda5_high_rm_scatter
infer sim_z1                 infer_z1                 obs_z1_lambda5_low_richness_contam
infer sim_z1                 infer_z1                 obs_z1_lambda5_ludlow
infer sim_z1                 infer_z1                 obs_z1_lambda5_prada
infer sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5_high_mc_scatter
infer sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5
infer sim_z1_high_noise      infer_z1                 obs_z1_lambda5_high_noise
infer sim_z1_high_noise      infer_z1                 obs_z1_lambda5
infer sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5_high_rm_scatter
infer sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5
infer sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5_high_rm_scatter
infer sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5

echo "############ Paper-I DeltaSigma rebuild DONE $(date) ############"
