#!/bin/bash
# Copy regenerated posterior (mcmc_cc, sbi_cc) and PPC (drawn_nfw_profiles, frac_diff) plots
# from outputs/plots/ into the paper's tex_source/figures_new/ tree.
# Maps each inference output dir -> its figures_new subdir (the manual mapping used in the paper).
cd "$(dirname "$0")/.."
OUT=outputs/plots
FIG=tex_source/figures_new

copy() {  # $1 = output dir basename (without .10000.376), $2 = figures_new subdir
    src="$OUT/$1.10000.376"
    dst="$FIG/$2"
    if [ ! -d "$src" ]; then echo "SKIP src missing: $src"; return; fi
    mkdir -p "$dst"
    # corner/posterior plots live in the output dir root
    for f in mcmc_cc.pdf sbi_cc.pdf; do
        [ -f "$src/$f" ] && cp "$src/$f" "$dst/" && echo "  $2/$f"
    done
    # PPC drawn-NFW + frac_diff plots live in the diagnostics/ subfolder
    for f in mcmc_drawn_nfw_profiles.pdf sbi_drawn_nfw_profiles.pdf frac_diff.pdf; do
        [ -f "$src/diagnostics/$f" ] && cp "$src/diagnostics/$f" "$dst/" && echo "  $2/$f (from diagnostics/)"
    done
}

echo "Copying regenerated plots into paper..."
copy "sim_z1.infer_z1.obs_z1_lambda5"                                            "near_ideal"
copy "sim_z1_high_mc_scatter.infer_z1_high_mc_scatter.obs_z1_lambda5_high_mc_scatter" "high_mc_scatter"
copy "sim_z1_high_noise.infer_z1.obs_z1_lambda5_high_noise"                      "high_noise"
copy "sim_z1_high_rm_scatter.infer_z1_high_rm_scatter.obs_z1_lambda5_high_rm_scatter" "high_rm_scatter"
copy "sim_z1.infer_z1.obs_z1_lambda5_prada"                                      "mc_relations/prada"
copy "sim_z1.infer_z1.obs_z1_lambda5_ludlow"                                     "mc_relations/ludlow"
# misspecification experiments
copy "sim_z1_high_noise.infer_z1.obs_z1_lambda5"                                 "misspec/high_noise_sims"
copy "sim_z1.infer_z1.obs_z1_lambda5_high_noise"                                 "misspec/low_noise_sims"
copy "sim_z1.infer_z1.obs_z1_lambda5_low_richness_contam"                        "misspec/rm_contam"
copy "sim_z1.infer_z1.obs_z1_lambda5_high_rm_scatter"                            "misspec/low_rm_scatter_sims"
copy "sim_z1_high_rm_scatter.infer_z1.obs_z1_lambda5"                            "misspec/high_rm_scatter_sims"
echo "DONE copying."
