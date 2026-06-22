"""Figure: inferred per-bin profile noise vs. the true input noise (baseline)."""
import pickle, numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
})
import hierarchical_mcmc_poc as H

OUT = H.OUT_DIR
RBINS = np.array(H.RBINS_KPC)
d = pickle.load(open(f"{OUT}/hier_inferred_noise_obs_z1_lambda5.pkl", "rb"))
sig = d["sigma_noise"]            # (nsamp, 30)
true_sig = d["true_sig"]
med = np.median(sig, 0); lo, hi = np.percentile(sig, [16, 84], axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(RBINS, lo, hi, color="#2ca02c", alpha=0.3, label="inferred noise 16-84%")
ax.plot(RBINS, med, color="#2ca02c", lw=2, label="inferred noise (median)")
ax.plot(RBINS, true_sig, "k--", lw=2, marker="o", ms=3, label="true input noise")
ax.set_xscale("log")
ax.set_xlabel(r"$R$ [kpc/$h$]")
ax.set_ylabel(r"per-bin profile noise $\sigma_i$ [dex]")
ax.set_title("Jointly inferred per-bin measurement noise vs. truth (baseline)")
ax.legend()
ax.set_ylim(0, max(true_sig.max(), hi.max()) * 1.25)
txt = (f"$\\sigma_M$ recovered = {d['summary']['sig_M'][0]:.3f} (true {d['true_mc'][:,0].std():.3f})\n"
       f"per-bin noise RMS err = {np.sqrt(np.mean((med-true_sig)**2)):.3f} dex")
ax.text(0.03, 0.05, txt, transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
fig.tight_layout()
fig.savefig(f"{OUT}/inferred_noise.png", dpi=140)
print(f"wrote {OUT}/inferred_noise.png")
