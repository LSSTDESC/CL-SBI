#!/usr/bin/env bash
# Parallelized Paper-I DeltaSigma rebuild.
#   Stage 1 (4 sim sets)  -> run CONCURRENTLY (each is single-core; maps to the 4 P-cores)
#   Stage 2 (8 obs sets)  -> run CONCURRENTLY (each is cheap, ~seconds)
#   Stage 3 (5 posteriors)-> SEQUENTIAL (torch training is multi-threaded)
#   Stage 4 (16 inferences)-> SEQUENTIAL (each already uses all cores via its MCMC pool)
# Writes to .delta_sigma-suffixed dirs; Sigma outputs untouched. Resumable (skip-if-exists).
set -u
cd "$(dirname "$0")"
NS=10000; NO=376; OBSV=delta_sigma; PY=python3
LOG=../outputs/logs; mkdir -p "$LOG"

echo "############ Paper-I DeltaSigma (parallel) started $(date) ############"

echo "===== STAGE 1/4: 4 sim sets (concurrent) ====="
for SIM in sim_z1 sim_z1_high_mc_scatter sim_z1_high_noise sim_z1_high_rm_scatter; do
    echo "  launch gen_simulations $SIM"
    $PY gen_simulations.py --sim_id "$SIM" --num_sims $NS --num_obs $NO --observable $OBSV \
        > "$LOG/gensim_${SIM}.deltasigma.log" 2>&1 &
done
wait
echo "STAGE 1 done $(date)"

echo "===== STAGE 2/4: 8 obs sets (concurrent) ====="
for OBS in obs_z1_lambda5 obs_z1_lambda5_high_mc_scatter obs_z1_lambda5_high_noise \
           obs_z1_lambda5_high_richness_contam obs_z1_lambda5_high_rm_scatter \
           obs_z1_lambda5_low_richness_contam obs_z1_lambda5_ludlow obs_z1_lambda5_prada; do
    echo "  launch gen_observations $OBS"
    $PY gen_observations.py --obs_id "$OBS" --num_obs $NO --observable $OBSV \
        > "$LOG/genobs_${OBS}.deltasigma.log" 2>&1 &
done
wait
echo "STAGE 2 done $(date)"

echo "===== STAGE 3/4: 5 posteriors (sequential) ====="
train() { echo "  train sim=$1 infer=$2 $(date)"; $PY train_inferrer.py --sim_id "$1" --infer_id "$2" --num_sims $NS --num_obs $NO --observable $OBSV; }
train sim_z1                 infer_z1
train sim_z1_high_mc_scatter infer_z1_high_mc_scatter
train sim_z1_high_noise      infer_z1
train sim_z1_high_rm_scatter infer_z1_high_rm_scatter
train sim_z1_high_rm_scatter infer_z1
echo "STAGE 3 done $(date)"

echo "===== STAGE 4/4: 16 inferences (sequential) ====="
infer() { echo "  infer sim=$1 infer=$2 obs=$3 $(date)"; $PY run_inference.py --sim_id "$1" --infer_id "$2" --obs_id "$3" --num_sims $NS --num_obs $NO --observable $OBSV; }
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

echo "############ Paper-I DeltaSigma (parallel) DONE $(date) ############"
