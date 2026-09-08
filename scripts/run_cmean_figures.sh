#!/usr/bin/env bash
# Corrected-mean figure batch (IR2/ZC1): regenerate per-experiment corner, diagnostic, and
# FTJ calibration/KS plots for EVERY completed .delta_sigma.corrected_mean experiment, into
# .delta_sigma.corrected_mean-suffixed plot dirs (median-stack figures preserved for A/B).
# Launch after run_paper1_cmean_par.sh completes. Aggregate + calibration-overlay figures:
# run make_aggregate_figures.py / make_calibration_overlay.py with
# --stack_estimator corrected_mean separately.
cd "$(dirname "$0")"
echo "### corrected-mean figure batch started $(date) ###"
n=0
for d in ../outputs/inference/*.10000.376.delta_sigma.corrected_mean; do
  [ -f "$d/mcmc_chains.pickle" ] || continue
  b=$(basename "$d"); IFS='.' read -r sim infer obs ns no rest <<< "$b"
  n=$((n+1)); echo "=== [$n] $sim / $infer / $obs $(date +%H:%M:%S) ==="
  python3 plot_chains.py       --sim_id "$sim" --infer_id "$infer" --obs_id "$obs" --num_sims 10000 --num_obs 376 --observable delta_sigma --stack_estimator corrected_mean --regenerate 2>&1 | tail -1
  python3 plot_diagnostics.py  --sim_id "$sim" --infer_id "$infer" --obs_id "$obs" --num_sims 10000 --num_obs 376 --observable delta_sigma --stack_estimator corrected_mean --regenerate 2>&1 | tail -1
  python3 plot_calibration.py  --sim_id "$sim" --infer_id "$infer" --obs_id "$obs" --num_sims 10000 --num_obs 376 --observable delta_sigma --stack_estimator corrected_mean --ftj-only --regenerate 2>&1 | tail -1
done
echo "### corrected-mean figure batch DONE ($n experiments) $(date) ###"
