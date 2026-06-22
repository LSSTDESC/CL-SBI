#!/bin/bash
# Regenerate ONLY the posterior + PPC plots (plot_chains.py produces mcmc_cc, sbi_cc,
# mcmc/sbi_drawn_nfw_profiles, frac_diff). Skips calibration (slow/flaky) and diagnostics.
# This carries both recent fixes: the c-marginal true-distribution overlay and the blue/red colors.
cd "$(dirname "$0")"

run() {
    SIM=$1; INF=$2; OBS=$3
    DIR="../outputs/inference/${SIM}.${INF}.${OBS}.10000.376"
    if [ ! -d "$DIR" ]; then echo "SKIP (no data): $SIM | $OBS"; return; fi
    echo "=== plot_chains: $SIM | $INF | $OBS ==="
    python3 plot_chains.py --sim_id $SIM --infer_id $INF --obs_id $OBS --num_sims 10000 --num_obs 376 --regenerate
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
echo "ALL CHAINS PLOTS DONE."
