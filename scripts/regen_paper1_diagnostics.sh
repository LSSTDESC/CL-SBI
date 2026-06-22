#!/bin/bash
# Regenerate PPC drawn-NFW-profile plots (mcmc/sbi_drawn_nfw_profiles, frac_diff) via plot_diagnostics.py.
# These land in each output dir's diagnostics/ subfolder and now use the blue/red (JTF/FTJ) colors.
cd "$(dirname "$0")"
run() {
    DIR="../outputs/inference/${1}.${2}.${3}.10000.376"
    if [ ! -d "$DIR" ]; then echo "SKIP (no data): $1 | $3"; return; fi
    echo "=== plot_diagnostics: $1 | $2 | $3 ==="
    python3 plot_diagnostics.py --sim_id $1 --infer_id $2 --obs_id $3 --num_sims 10000 --num_obs 376 --regenerate
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
echo "ALL DIAGNOSTICS DONE."
