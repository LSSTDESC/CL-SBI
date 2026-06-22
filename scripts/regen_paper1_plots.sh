#!/bin/bash
# Regenerate ALL Paper I plots (chains, diagnostics, calibration+KS) from the freshly
# regenerated adaptive-sampler MCMC chains. Covers all 16 experiment configs.
cd "$(dirname "$0")"

run_plots() {
    SIM=$1; INF=$2; OBS=$3; NS=$4; NO=$5
    DIR="../outputs/inference/${SIM}.${INF}.${OBS}.${NS}.${NO}"
    if [ ! -d "$DIR" ]; then echo "SKIP (no data): $SIM | $OBS"; return; fi
    echo "=== Plotting: $SIM | $INF | $OBS ==="
    python3 plot_chains.py      --sim_id $SIM --infer_id $INF --obs_id $OBS --num_sims $NS --num_obs $NO --regenerate
    python3 plot_diagnostics.py --sim_id $SIM --infer_id $INF --obs_id $OBS --num_sims $NS --num_obs $NO --regenerate
    python3 plot_calibration.py --sim_id $SIM --infer_id $INF --obs_id $OBS --num_sims $NS --num_obs $NO --regenerate
    echo ""
}

run_plots sim_z1                 infer_z1                 obs_z1_lambda5                      10000 376
run_plots sim_z1                 infer_z1                 obs_z1_lambda5_high_mc_scatter      10000 376
run_plots sim_z1                 infer_z1                 obs_z1_lambda5_high_noise           10000 376
run_plots sim_z1                 infer_z1                 obs_z1_lambda5_high_richness_contam 10000 376
run_plots sim_z1                 infer_z1                 obs_z1_lambda5_high_rm_scatter      10000 376
run_plots sim_z1                 infer_z1                 obs_z1_lambda5_low_richness_contam  10000 376
run_plots sim_z1                 infer_z1                 obs_z1_lambda5_ludlow               10000 376
run_plots sim_z1                 infer_z1                 obs_z1_lambda5_prada                10000 376
run_plots sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5_high_mc_scatter      10000 376
run_plots sim_z1_high_mc_scatter infer_z1_high_mc_scatter obs_z1_lambda5                      10000 376
run_plots sim_z1_high_noise      infer_z1                 obs_z1_lambda5_high_noise           10000 376
run_plots sim_z1_high_noise      infer_z1                 obs_z1_lambda5                      10000 376
run_plots sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5_high_rm_scatter      10000 376
run_plots sim_z1_high_rm_scatter infer_z1_high_rm_scatter obs_z1_lambda5                      10000 376
run_plots sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5_high_rm_scatter      10000 376
run_plots sim_z1_high_rm_scatter infer_z1                 obs_z1_lambda5                      10000 376

echo "ALL PLOTS DONE."
