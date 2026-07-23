#!/bin/bash
# Copy the DeltaSigma (population-truth) figures into tex_source/figures_new/, replacing the
# Sigma originals AT THE SAME PATHS so main.tex figure references are unchanged.
# A one-time backup of the Sigma originals is kept in tex_source/figures_new_sigma_backup/.
cd "$(dirname "$0")/.."
OUT=outputs/plots
FIG=tex_source/figures_new
BAK=tex_source/figures_new_sigma_backup

# one-time backup of the Sigma figure tree
if [ ! -d "$BAK" ]; then
    cp -R "$FIG" "$BAK" && echo "Backed up Sigma figures -> $BAK"
fi

copy() {  # $1 = output dir basename (without .10000.376), $2 = figures_new subdir
    src="$OUT/$1.10000.376.delta_sigma"
    dst="$FIG/$2"
    if [ ! -d "$src" ]; then echo "SKIP src missing: $src"; return; fi
    mkdir -p "$dst"
    for f in mcmc_cc.pdf sbi_cc.pdf; do
        [ -f "$src/$f" ] && cp "$src/$f" "$dst/" && echo "  $2/$f"
    done
    for f in mcmc_drawn_nfw_profiles.pdf sbi_drawn_nfw_profiles.pdf frac_diff.pdf; do
        [ -f "$src/diagnostics/$f" ] && cp "$src/diagnostics/$f" "$dst/" && echo "  $2/$f (diagnostics)"
    done
}

echo "Copying DeltaSigma per-experiment plots into paper..."
copy "sim_z1.infer_z1.obs_z1_lambda5"                                                  "near_ideal"
copy "sim_z1_high_mc_scatter.infer_z1_high_mc_scatter.obs_z1_lambda5_high_mc_scatter"  "high_mc_scatter"
copy "sim_z1_high_noise.infer_z1.obs_z1_lambda5_high_noise"                            "high_noise"
copy "sim_z1_high_rm_scatter.infer_z1_high_rm_scatter.obs_z1_lambda5_high_rm_scatter"  "high_rm_scatter"
copy "sim_z1.infer_z1.obs_z1_lambda5_prada"                                            "mc_relations/prada"
copy "sim_z1.infer_z1.obs_z1_lambda5_ludlow"                                           "mc_relations/ludlow"
copy "sim_z1_high_noise.infer_z1.obs_z1_lambda5"                                       "misspec/high_noise_sims"
copy "sim_z1.infer_z1.obs_z1_lambda5_high_noise"                                       "misspec/low_noise_sims"
copy "sim_z1.infer_z1.obs_z1_lambda5_low_richness_contam"                              "misspec/rm_contam"
copy "sim_z1.infer_z1.obs_z1_lambda5_high_rm_scatter"                                  "misspec/low_rm_scatter_sims"
copy "sim_z1_high_rm_scatter.infer_z1.obs_z1_lambda5"                                  "misspec/high_rm_scatter_sims"

echo "Copying aggregate figures (Figs 8/10/11)..."
for f in aggregate_ftj_in_distro aggregate_ftj_ood_noise_mc aggregate_ftj_ood_rm; do
    [ -f "$OUT/infer_agg.delta_sigma/$f.pdf" ] && cp "$OUT/infer_agg.delta_sigma/$f.pdf" "$FIG/infer_agg/" && echo "  infer_agg/$f.pdf"
done

echo "Copying calibration overlays (Fig 13)..."
for f in calibration_ftj_overlay_10k_376 calibration_ftj_overlay_10k_376_ood; do
    [ -f "$OUT/calibration_overlay.delta_sigma/$f.pdf" ] && cp "$OUT/calibration_overlay.delta_sigma/$f.pdf" "$FIG/calibration/" && echo "  calibration/$f.pdf"
done

echo "Copying ensemble + residual figures (new)..."
mkdir -p "$FIG/ensemble"
[ -f "$OUT/ensemble_recovery.delta_sigma.pdf" ] && cp "$OUT/ensemble_recovery.delta_sigma.pdf" "$FIG/ensemble/" && echo "  ensemble/ensemble_recovery.delta_sigma.pdf"
[ -f "$OUT/residual_consistency.delta_sigma/residual_sim_z1.obs_z1_lambda5.pdf" ] && \
    cp "$OUT/residual_consistency.delta_sigma/residual_sim_z1.obs_z1_lambda5.pdf" "$FIG/ensemble/" && echo "  ensemble/residual_sim_z1.obs_z1_lambda5.pdf"
echo "DONE."
