#!/usr/bin/env bash
# Post-recovery ΔΣ figure batch: regenerate per-experiment corner, diagnostic (profile +
# frac-diff, now ΔΣ-labeled), and FTJ calibration/KS plots for EVERY completed .delta_sigma
# experiment, into .delta_sigma-suffixed plot dirs (Sigma figures preserved for A/B).
#
# Launch ONLY after the sim_z1_high_noise recovery finishes (so it covers all 16 and doesn't
# oversubscribe). NOTE: the aggregate (aggregate_ftj_*) and calibration-OVERLAY figures are
# built by notebooks and are NOT covered here -- they need separate ΔΣ adaptation.
cd "$(dirname "$0")"
echo "### ΔΣ figure batch started $(date) ###"
n=0
for d in ../outputs/inference/*.10000.376.delta_sigma; do
  [ -f "$d/mcmc_chains.pickle" ] || continue
  b=$(basename "$d"); IFS='.' read -r sim infer obs ns no rest <<< "$b"
  n=$((n+1)); echo "=== [$n] $sim / $infer / $obs $(date +%H:%M:%S) ==="
  python3 plot_chains.py       --sim_id "$sim" --infer_id "$infer" --obs_id "$obs" --num_sims 10000 --num_obs 376 --observable delta_sigma --regenerate 2>&1 | tail -1
  python3 plot_diagnostics.py  --sim_id "$sim" --infer_id "$infer" --obs_id "$obs" --num_sims 10000 --num_obs 376 --observable delta_sigma --regenerate 2>&1 | tail -1
  python3 plot_calibration.py  --sim_id "$sim" --infer_id "$infer" --obs_id "$obs" --num_sims 10000 --num_obs 376 --observable delta_sigma --ftj-only --regenerate 2>&1 | tail -1
done
echo "### ΔΣ figure batch DONE ($n experiments) $(date) ###"
