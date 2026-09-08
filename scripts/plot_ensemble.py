"""
Plot the sampling-robustness ensemble (P2-d): recovered vs. true (M,c) with run-to-run
error bars from the M realizations, per experiment. Doubles as the mean-mass-recovery
figure (Payerne comment #6). Reads outputs/ensembles/*/ensemble.npz.
Usage: python3 plot_ensemble.py --observable delta_sigma
"""
import sys, os, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--observable", default="delta_sigma")
ap.add_argument("--stack_estimator", default="median")
args = ap.parse_args()
sd = os.path.dirname(__file__)
suf = "" if args.observable == "surface_density" else f".{args.observable}"
suf += "" if args.stack_estimator == "median" else f".{args.stack_estimator}"
ens_dir = os.path.join(sd, "../outputs/ensembles")

rows = []
for d in sorted(glob.glob(os.path.join(ens_dir, f"*.10000.376{suf}"))):
    z = np.load(os.path.join(d, "ensemble.npz"), allow_pickle=True)
    t, sf = z["truth"], z["sbi_ftj"]
    if len(t) == 0:
        continue
    key = os.path.basename(d).replace(".10000.376" + suf, "")
    # x labels = the paper's test names (matching the results-section titles / tab:num_exp)
    LABELS = {
        "sim_z1.infer_z1.obs_z1_lambda5": "Baseline",
        "sim_z1_high_mc_scatter.infer_z1_high_mc_scatter.obs_z1_lambda5_high_mc_scatter": "High M-c scatter",
        "sim_z1_high_noise.infer_z1.obs_z1_lambda5_high_noise": "High noise",
        "sim_z1_high_rm_scatter.infer_z1_high_rm_scatter.obs_z1_lambda5_high_rm_scatter": r"High $\lambda$-M scatter",
        "sim_z1.infer_z1.obs_z1_lambda5_high_mc_scatter": "Obs higher M-c scatter",
        "sim_z1.infer_z1.obs_z1_lambda5_prada": "Prada M-c",
        "sim_z1.infer_z1.obs_z1_lambda5_ludlow": "Ludlow M-c",
        "sim_z1.infer_z1.obs_z1_lambda5_high_noise": "Obs higher noise",
        "sim_z1_high_noise.infer_z1.obs_z1_lambda5": "Sims higher noise",
        "sim_z1.infer_z1.obs_z1_lambda5_low_richness_contam": "Richness contam.",
        "sim_z1.infer_z1.obs_z1_lambda5_high_richness_contam": "Richness contam. (high)",
        "sim_z1.infer_z1.obs_z1_lambda5_high_rm_scatter": r"Obs higher $\lambda$-M",
        "sim_z1_high_rm_scatter.infer_z1.obs_z1_lambda5": r"Sims higher $\lambda$-M",
        "sim_z1_high_rm_scatter.infer_z1.obs_z1_lambda5_high_rm_scatter": r"High $\lambda$-M (unmatched prior)",
        "sim_z1_high_rm_scatter.infer_z1_high_rm_scatter.obs_z1_lambda5": r"Sims higher $\lambda$-M (matched prior)",
        "sim_z1_high_rm_scatter.infer_z1_high_rm_scatter.obs_z1_lambda5_high_rm_scatter": r"High $\lambda$-M scatter",
        "sim_z1_high_mc_scatter.infer_z1_high_mc_scatter.obs_z1_lambda5": "Sims higher M-c scatter",
    }
    label = LABELS.get(key, key)
    rows.append((label, t, sf, z["mcmc_ftj"]))

n = len(rows)
fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.7 * n), 9), sharex=True)
x = np.arange(n)
for ax, (pi, name) in zip(axes, [(0, r"$\log_{10}M$"), (2, "concentration $c$")]):
    for i, (label, t, sf, mf) in enumerate(rows):
        # truth median: mean +/- run-to-run scatter
        ax.errorbar(i - 0.12, t[:, pi].mean(), yerr=t[:, pi].std(), fmt="ks", ms=6, capsize=3,
                    label="observed (truth) median" if i == 0 else None)
        # SBI FTJ recovered: mean +/- run-to-run scatter
        ax.errorbar(i + 0.12, sf[:, pi].mean(), yerr=sf[:, pi].std(), fmt="o", color="#7e3ff2", ms=6, capsize=3,
                    label="SBI FTJ recovered" if i == 0 else None)
        if len(mf):
            ax.errorbar(i, mf[:, pi].mean(), yerr=mf[:, pi].std(), fmt="^", color="#d62728", ms=6, capsize=3,
                        label="MCMC FTJ recovered" if i == 0 else None)
    ax.set_ylabel(f"{name} median", fontsize="large")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize="small", loc="best")
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([r[0] for r in rows], rotation=60, ha="right", fontsize="x-small")
axes[0].set_title(f"Sampling-robustness ensemble (M={len(rows[0][1])} realizations, {args.observable}): "
                  f"recovered vs. observed, error bars = run-to-run scatter", fontsize="medium")
fig.tight_layout()
outp = os.path.join(sd, f"../outputs/plots/ensemble_recovery{suf}.pdf")
fig.savefig(outp); fig.savefig(outp.replace(".pdf", ".png"), dpi=110)
print(f"saved -> {outp}")

# also print a compact bias table
print("\nSBI FTJ median bias (recovered - truth), mean +/- run-to-run scatter:")
for label, t, sf, mf in rows:
    bM = sf[:, 0] - t[:, 0]; bc = sf[:, 2] - t[:, 2]
    print(f"  {label:<40} logM {bM.mean():+.3f}±{bM.std():.3f}   c {bc.mean():+.3f}±{bc.std():.3f}")
