#!/usr/bin/env bash
# Training-seed ensemble (IR2/Zhou L586-594): retrain both baseline inferrers K=5 times on the
# IDENTICAL training set (only network init / batch order vary via --train_seed), then evaluate
# each net's FTJ population-summary bias and coverage across M=20 fresh observation
# realizations (seed_ensemble_eval.py). Establishes the seed floor for the reported biases —
# in particular whether the median-era "persistent +0.02-0.03 c offset" was a seed artifact.
set -u
cd "$(dirname "$0")"
NS=10000; NO=376; OBSV=delta_sigma; EST=corrected_mean; PY=python3
for SEED in 101 102 103 104 105; do
    echo "=== train seed $SEED $(date) ==="
    $PY train_inferrer.py --sim_id sim_z1 --infer_id infer_z1 --num_sims $NS --num_obs $NO \
        --observable $OBSV --stack_estimator $EST --train_seed $SEED \
        > "../outputs/logs/train_seed${SEED}.log" 2>&1
done
echo "=== all seeds trained; evaluating $(date) ==="
$PY seed_ensemble_eval.py
echo "=== seed ensemble DONE $(date) ==="
