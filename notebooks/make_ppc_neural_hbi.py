"""
Posterior predictive check for the fully-amortized neural HBI, one row per experiment (all 8).

For each experiment we take neural-HBI's inferred population hyperparameters (mu_M, sig_M, c0, beta,
sig_c), draw a population of clusters from that generative model
    logM ~ N(mu_M, sig_M);  c = c0 + beta*(logM - mu_M) + N(0, sig_c),
forward-model each to a noiseless NFW log10-Sigma profile, and compare the predicted population to
the observed profiles. This is the same construction as make_ppc_three_method.py but for the neural
HBI's RELATION parameterization (so it captures the M-c slope, unlike a diagonal-Gaussian draw).

The two richness-contamination experiments are MIS-SPECIFIED (single-Gaussian population on a
non-Gaussian in-bin mass distribution); their PPC is expected to show the predicted band missing the
observed profiles -- the visual motivation for the Option-1 mixture follow-up. They are drawn with a
dashed median + hatched band and flagged in the title.

Run under base env:
  /Users/akumgill/anaconda3/bin/python notebooks/make_ppc_neural_hbi.py
"""
import os, pickle, numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
})
import jax, jax.numpy as jnp
import hierarchical_mcmc_poc as H

OUT = H.OUT_DIR; REPO = H.REPO
RBINS = np.array(H.RBINS_KPC)
fwd = jax.vmap(H.nfw_logSigma)

d = pickle.load(open(f"{OUT}/afull_neural_hbi.pkl", "rb"))
EXP = d["experiments"]
CONTAM = set(d.get("contam", []))
ORDER = ["obs_z1_lambda5", "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_rm_scatter",
         "obs_z1_lambda5_high_noise", "obs_z1_lambda5_ludlow", "obs_z1_lambda5_prada",
         "obs_z1_lambda5_low_richness_contam", "obs_z1_lambda5_high_richness_contam"]
ORDER = [o for o in ORDER if o in EXP]
LABEL = {"obs_z1_lambda5": "Baseline", "obs_z1_lambda5_high_mc_scatter": "High M-c scatter",
         "obs_z1_lambda5_high_rm_scatter": r"High $\lambda$-M scatter", "obs_z1_lambda5_high_noise": "High noise",
         "obs_z1_lambda5_ludlow": "Ludlow M-c (OOD)", "obs_z1_lambda5_prada": "Prada M-c (OOD)",
         "obs_z1_lambda5_low_richness_contam": "Low richness contam. (mis-spec.)",
         "obs_z1_lambda5_high_richness_contam": "High richness contam. (mis-spec.)"}
C_NHBI = "#9467bd"


def pop_profiles(est, n_pop=400, seed=0):
    """Draw a population from the neural-HBI relation params and forward-model -> log10 profiles."""
    muM, sigM = est["mu_M"][0], est["sig_M"][0]
    c0, beta, sigc = est["c0"][0], est["beta"][0], est["sig_c"][0]
    rng = np.random.default_rng(seed)
    logM = rng.normal(muM, max(sigM, 1e-3), n_pop)
    c = np.clip(c0 + beta * (logM - muM) + rng.normal(0, max(sigc, 1e-3), n_pop), 2.0, 9.0)
    return np.array(fwd(jnp.array(logM), jnp.array(c)))


n = len(ORDER)
fig, axes = plt.subplots(n, 2, figsize=(12, 1.7 * n), squeeze=False)
for i, obs_id in enumerate(ORDER):
    obs = np.load(f"{REPO}/outputs/observations/{obs_id}.376/drawn_nfw_profiles.npy")
    obs_med = np.median(obs, 0)
    is_contam = obs_id in CONTAM

    axL, axR = axes[i]
    for p in obs[::12]:
        axL.plot(RBINS, p, color="0.8", lw=0.3, alpha=0.5, zorder=1)
    axL.plot(RBINS, obs_med, "k", lw=2, zorder=6, label="observed median" if i == 0 else None)
    axR.axhline(0, color="k", lw=1.2, zorder=6)

    prof = pop_profiles(EXP[obs_id]["est"])
    med = np.median(prof, 0); lo, hi = np.percentile(prof, [16, 84], axis=0)
    ls = "--" if is_contam else "-"
    hatch = "////" if is_contam else None
    axL.fill_between(RBINS, lo, hi, color=C_NHBI, alpha=0.22, zorder=2, hatch=hatch)
    axL.plot(RBINS, med, color=C_NHBI, lw=1.8, ls=ls, zorder=4,
             label="Neural HBI" if i == 0 else None)
    fr_lo = 10 ** lo / 10 ** obs_med - 1
    fr_hi = 10 ** hi / 10 ** obs_med - 1
    fr_med = 10 ** med / 10 ** obs_med - 1
    axR.fill_between(RBINS, fr_lo, fr_hi, color=C_NHBI, alpha=0.20, zorder=2, hatch=hatch)
    axR.plot(RBINS, fr_med, color=C_NHBI, lw=1.6, ls=ls, zorder=4)

    axL.set_xscale("log"); axL.set_ylabel(LABEL[obs_id], fontsize=12)
    axR.set_xscale("log"); axR.set_ylim(-0.6, 0.6)
    if i == 0:
        axL.set_title("profiles: observed vs predicted population")
        axR.set_title("fractional residual (pred/obs $-$ 1)")
    if i == n - 1:
        axL.set_xlabel(r"$R$ [kpc/$h$]"); axR.set_xlabel(r"$R$ [kpc/$h$]")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.005), fontsize=13)
fig.suptitle("Neural HBI posterior predictive checks "
             r"($\dagger$ contam.\ rows are mis-specified: single-Gaussian on non-Gaussian mass)",
             y=1.015, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"{OUT}/ppc_neural_hbi.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/ppc_neural_hbi.png ({n} experiments)")
