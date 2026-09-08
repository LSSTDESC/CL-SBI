#!/usr/bin/env bash
# Paper-I rebuild on the bias-corrected-mean JTF stack (IR2/Zhou response, ZC1 option c).
# Mirrors run_paper1_deltasigma_par.sh but with --stack_estimator corrected_mean.
# Observations are estimator-independent (reuse existing .delta_sigma dirs); only the
# sims (JTF training stacks), posteriors, and inferences are regenerated, into
# .delta_sigma.corrected_mean-suffixed dirs. Median-stack outputs untouched (A/B).
# Resumable (skip-if-exists).
set -u
cd "$(dirname "$0")"
NS=10000; NO=376; OBSV=delta_sigma; EST=corrected_mean; PY=python3
LOG=../outputs/logs; mkdir -p "$LOG"

echo "############ Paper-I corrected-mean rebuild started $(date) ############"

echo "===== STAGE 1/3: 4 sim sets (concurrent) ====="
for SIM in sim_z1 sim_z1_high_mc_scatter sim_z1_high_noise sim_z1_high_rm_scatter; do
    echo "  launch gen_simulations $SIM"
    $PY gen_simulations.py --sim_id "$SIM" --num_sims $NS --num_obs $NO --observable $OBSV \
        --stack_estimator $EST > "$LOG/gensim_${SIM}.cmean.log" 2>&1 &
done
wait
echo "STAGE 1 done $(date)"

echo "===== STAGE 2/3: 5 posteriors (sequential) ====="
train() { echo "  train sim=$1 infer=$2 $(date)"; $PY train_inferrer.py --sim_id "$1" --infer_id "$2" --num_sims $NS --num_obs $NO --observable $OBSV --stack_estimator $EST; }
train sim_z1                 infer_z1
train sim_z1_high_mc_scatter infer_z1_high_mc_scatter
train sim_z1_high_noise      infer_z1
train sim_z1_high_rm_scatter infer_z1_high_rm_scatter
train sim_z1_high_rm_scatter infer_z1
echo "STAGE 2 done $(date)"

echo "===== STAGE 3/3: 16 inferences (sequential) ====="
infer() { echo "  infer sim=$1 infer=$2 obs=$3 $(date)"; $PY run_inference.py --sim_id "$1" --infer_id "$2" --obs_id "$3" --num_sims $NS --num_obs $NO --observable $OBSV --stack_estimator $EST; }
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

echo "############ Paper-I corrected-mean rebuild DONE $(date) ############"
