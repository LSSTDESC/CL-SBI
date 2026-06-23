"""
Three-method posterior predictive check, one row per experiment (like the aggregate comparison fig).

For each experiment and each method (MCMC joint-likelihood, SBI FTJ, hierarchical HMC) we draw a
population of clusters from the method's inferred (mu, Sigma) of (logM, c), forward-model each to a
noiseless NFW profile, and compare the predicted population to the observed profiles.

Layout per experiment (one row, two columns):
  Left  : profiles -- grey observed, black median observed, and each method's predicted-population
          16-84% band + median (colored).
  Right : fractional residual (predicted median - observed median)/observed median for each method,
          with the method population bands, so over/under-prediction is visible near zero.
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

rows = pickle.load(open(f"{OUT}/_all_methods_rows.pkl", "rb"))
ORDER = ["obs_z1_lambda5", "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_noise",
         "obs_z1_lambda5_high_rm_scatter", "obs_z1_lambda5_low_richness_contam",
         "obs_z1_lambda5_high_richness_contam", "obs_z1_lambda5_prada", "obs_z1_lambda5_ludlow"]
LABEL = {"obs_z1_lambda5": "Baseline", "obs_z1_lambda5_high_mc_scatter": "High M-c scatter",
         "obs_z1_lambda5_high_noise": "High noise", "obs_z1_lambda5_high_rm_scatter": r"High $\lambda$-M scatter",
         "obs_z1_lambda5_low_richness_contam": "Low richness contam.",
         "obs_z1_lambda5_high_richness_contam": "High richness contam.",
         "obs_z1_lambda5_prada": "Prada M-c (OOD)", "obs_z1_lambda5_ludlow": "Ludlow M-c (OOD)"}
COL = {"mcmc": "#1f77b4", "sbi": "#ff7f0e", "hmc": "#2ca02c", "nhbi": "#9467bd"}
NAME = {"mcmc": "MCMC joint", "sbi": "SBI FTJ", "hmc": "Hierarchical HMC", "nhbi": "Neural HBI"}

# Neural HBI (architecture A): population drawn from its inferred RELATION params (c = c0 + beta*dM
# + scatter), keyed by obs. Available for all 8 experiments from the fully-amortized run.
try:
    _nhbi = pickle.load(open(f"{OUT}/afull_neural_hbi.pkl", "rb"))["experiments"]
except FileNotFoundError:
    _nhbi = {}

# canonical sim_z1/infer_z1 row per obs
by_obs = {}
for r in rows:
    if r["sim"] == "sim_z1" and r["infer"] == "infer_z1" and r["obs"] not in by_obs:
        by_obs[r["obs"]] = r
avail = [o for o in ORDER if o in by_obs]

def pop_profiles(r, method, n_pop=400, seed=0):
    """Sample a population from method's inferred population and forward-model -> log10 profiles."""
    rng = np.random.default_rng(seed)
    if method == "nhbi":
        # neural-HBI infers the RELATION: draw logM~N(muM,sigM); c = c0 + beta*(logM-muM) + N(0,sigc)
        if r["obs"] not in _nhbi:
            return None
        e = _nhbi[r["obs"]]["est"]
        muM, sigM = e["mu_M"][0], e["sig_M"][0]
        c0, beta, sigc = e["c0"][0], e["beta"][0], e["sig_c"][0]
        logM = rng.normal(muM, max(sigM, 1e-3), n_pop)
        c = np.clip(c0 + beta * (logM - muM) + rng.normal(0, max(sigc, 1e-3), n_pop), 2.0, 9.0)
        return np.array(fwd(jnp.array(logM), jnp.array(c)))
    muM, sigM = r.get(method + "_muM"), r.get(method + "_sigM")
    muc, sigc = r.get(method + "_muc"), r.get(method + "_sigc")
    if muM is None:
        return None
    logM = rng.normal(muM, max(sigM, 1e-3), n_pop)
    c = np.clip(rng.normal(muc, max(sigc, 1e-3), n_pop), 2.0, 9.0)
    return np.array(fwd(jnp.array(logM), jnp.array(c)))

n = len(avail)
fig, axes = plt.subplots(n, 2, figsize=(12, 1.7 * n), squeeze=False)
for i, obs_id in enumerate(avail):
    r = by_obs[obs_id]
    obs = np.load(f"{REPO}/outputs/observations/{obs_id}.376/drawn_nfw_profiles.npy")
    obs_med = np.median(obs, 0)

    axL, axR = axes[i]
    for p in obs[::12]:
        axL.plot(RBINS, p, color="0.8", lw=0.3, alpha=0.5, zorder=1)
    axL.plot(RBINS, obs_med, "k", lw=2, zorder=6, label="observed median")
    axR.axhline(0, color="k", lw=1.2, zorder=6)

    for method in ["mcmc", "sbi", "hmc", "nhbi"]:
        if method == "mcmc" and not r.get("mcmc_converged", True):
            continue
        prof = pop_profiles(r, method)
        if prof is None:
            continue
        med = np.median(prof, 0); lo, hi = np.percentile(prof, [16, 84], axis=0)
        axL.fill_between(RBINS, lo, hi, color=COL[method], alpha=0.25, zorder=2)
        axL.plot(RBINS, med, color=COL[method], lw=1.8, zorder=4,
                 label=NAME[method] if i == 0 else None)
        # fractional residual vs observed median (in linear Sigma space)
        fr_med = 10 ** med / 10 ** obs_med - 1
        fr_lo = 10 ** lo / 10 ** obs_med - 1
        fr_hi = 10 ** hi / 10 ** obs_med - 1
        axR.fill_between(RBINS, fr_lo, fr_hi, color=COL[method], alpha=0.22, zorder=2)
        axR.plot(RBINS, fr_med, color=COL[method], lw=1.8, zorder=4)

    axL.set_xscale("log"); axR.set_xscale("log")
    axL.set_ylabel(LABEL[obs_id] + "\n" + r"$\log_{10}\Sigma$", fontsize=14)
    axR.set_ylabel(r"$\Sigma_{\rm pred}/\Sigma_{\rm obs}-1$", fontsize=14)
    axR.set_ylim(-0.5, 0.5)
    if i == n - 1:
        axL.set_xlabel(r"$R$ [kpc/$h$]"); axR.set_xlabel(r"$R$ [kpc/$h$]")
    if i == 0:
        axL.set_title("predicted vs observed profiles"); axR.set_title("fractional residual")
        axL.legend(fontsize=13, loc="lower left")

fig.suptitle("Posterior predictive checks across methods (population pushed through the forward model)",
             fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"{OUT}/aggregate_ppc_three_method.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/aggregate_ppc_three_method.png ({n} experiments)")
