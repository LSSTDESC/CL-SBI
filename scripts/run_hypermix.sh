#!/usr/bin/env bash
# ZI (IR2/Zhou) prior-marginalized experiment: train on sim_z1_hypermix (per-stack draws of
# M-c relation [child18/ludlow16/prada12], R-M relation [mcclintock18/murata17], and both
# scatters U[0.1,0.7]) and infer against in-distribution + previously-OOD observations.
# MCMC FTJ is reused from the existing sim_z1 dirs (same obs + priors; estimator-independent).
set -u
cd "$(dirname "$0")"
NS=10000; NO=376; OBSV=delta_sigma; EST=corrected_mean; PY=python3
LOG=../outputs/logs; mkdir -p "$LOG"

echo "#### hypermix (ZI) started $(date) ####"
$PY gen_simulations.py --sim_id sim_z1_hypermix --num_sims $NS --num_obs $NO \
    --observable $OBSV --stack_estimator $EST > "$LOG/gensim_hypermix.log" 2>&1
echo "gen done $(date)"
$PY train_inferrer.py --sim_id sim_z1_hypermix --infer_id infer_z1 --num_sims $NS --num_obs $NO \
    --observable $OBSV --stack_estimator $EST > "$LOG/train_hypermix.log" 2>&1
echo "train done $(date)"
for OBS in obs_z1_lambda5 obs_z1_lambda5_prada obs_z1_lambda5_ludlow \
           obs_z1_lambda5_high_mc_scatter obs_z1_lambda5_high_rm_scatter; do
  echo "infer vs $OBS $(date)"
  $PY run_inference.py --sim_id sim_z1_hypermix --infer_id infer_z1 --obs_id "$OBS" \
      --num_sims $NS --num_obs $NO --observable $OBSV --stack_estimator $EST \
      --reuse_mcmc_ftj_from "../outputs/inference/sim_z1.infer_z1.${OBS}.${NS}.${NO}.delta_sigma" \
      > "$LOG/infer_hypermix_${OBS}.log" 2>&1
done
echo "#### hypermix (ZI) DONE $(date) ####"
