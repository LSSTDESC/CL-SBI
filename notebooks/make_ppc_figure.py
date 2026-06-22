"""
Aggregate posterior predictive check (PPC) for the hierarchical HMC fits.

For each experiment: the hierarchical model gives a posterior on the population hyperparameters
(mu_M, sig_M, c0, beta, sig_c). A PPC asks: if we draw a population of clusters from that inferred
posterior and forward-model them, do the predicted profiles reproduce the observed ones?

For each experiment panel we show:
  - thin grey lines: the observed NFW profiles (the data)
  - black solid: the median observed profile
  - green band: the 16-84 percentile spread of PPC-predicted profiles (population draws pushed
    through the NFW forward model), i.e. the predicted population scatter
  - green dashed: the median PPC-predicted profile
A good fit has the black median sitting inside the green band and the green median tracking it.
"""
import os, pickle, numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
import hierarchical_mcmc_poc as H

OUT = H.OUT_DIR
REPO = H.REPO
RBINS = np.array(H.RBINS_KPC)

# experiment order + labels (match the other aggregate figure)
ORDER = ["obs_z1_lambda5", "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_noise",
         "obs_z1_lambda5_high_rm_scatter", "obs_z1_lambda5_low_richness_contam",
         "obs_z1_lambda5_high_richness_contam", "obs_z1_lambda5_prada", "obs_z1_lambda5_ludlow"]
LABEL = {"obs_z1_lambda5": "Baseline", "obs_z1_lambda5_high_mc_scatter": "High M-c scatter",
         "obs_z1_lambda5_high_noise": "High noise", "obs_z1_lambda5_high_rm_scatter": r"High $\lambda$-M scatter",
         "obs_z1_lambda5_low_richness_contam": "Low richness contam.",
         "obs_z1_lambda5_high_richness_contam": "High richness contam.",
         "obs_z1_lambda5_prada": "Prada M-c (OOD)", "obs_z1_lambda5_ludlow": "Ludlow M-c (OOD)"}

# vmapped forward model: (logM, c) -> log10 Sigma profile
fwd = jax.vmap(H.nfw_logSigma)

def ppc_profiles(post, obs_sigmas, n_pop=600, seed=0):
    """Draw n_pop population clusters from the hyperparameter posterior, forward-model each.

    Returns (noiseless, noisy): the population-only predicted profiles, and the full
    posterior-predictive profiles with per-bin observation noise added (sqrt(sigma_obs^2 +
    sig_extra^2)) so they are directly comparable to the noisy observed data.
    """
    rng = np.random.default_rng(seed)
    n = len(post["mu_M"])
    idx = rng.integers(0, n, n_pop)
    muM, sigM = post["mu_M"][idx], post["sig_M"][idx]
    c0, beta, sigc = post["c0"][idx], post["beta"][idx], post["sig_c"][idx]
    sig_extra = post["sig_extra"][idx]
    logM = muM + sigM * rng.standard_normal(n_pop)
    c = c0 + beta * (logM - muM) + sigc * rng.standard_normal(n_pop)
    c = np.clip(c, 2.0, 9.0)
    noiseless = np.array(fwd(jnp.array(logM), jnp.array(c)))      # (n_pop, 30) log10 Sigma
    sig_tot = np.sqrt(obs_sigmas[None, :] ** 2 + sig_extra[:, None] ** 2)
    noisy = noiseless + rng.standard_normal(noiseless.shape) * sig_tot
    return noiseless, noisy

avail = [o for o in ORDER if os.path.exists(f"{OUT}/hier_post_{o}.pkl")]
n = len(avail)
ncol = 4; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)

for k, obs_id in enumerate(avail):
    ax = axes[k // ncol][k % ncol]
    d = pickle.load(open(f"{OUT}/hier_post_{obs_id}.pkl", "rb"))
    post = d["samples"]
    obs = np.load(f"{REPO}/outputs/observations/{obs_id}.376/drawn_nfw_profiles.npy")  # (376,30) log10
    obs_sig = np.load(f"{REPO}/outputs/observations/{obs_id}.376/sigmas.npy")

    # observed data
    for p in obs[::8]:
        ax.plot(RBINS, p, color="0.7", lw=0.4, alpha=0.5, zorder=1)
    ax.plot(RBINS, np.median(obs, 0), color="k", lw=2, zorder=5, label="median observed")

    # PPC: full predictive (with obs noise) should match the grey data envelope;
    # population-only band shows the inferred intrinsic scatter.
    noiseless, noisy = ppc_profiles(post, obs_sig)
    lo_n, hi_n = np.percentile(noisy, [16, 84], axis=0)
    ax.fill_between(RBINS, lo_n, hi_n, color="#2ca02c", alpha=0.18, zorder=2,
                    label="PPC 16-84% (with noise)")
    lo_p, hi_p = np.percentile(noiseless, [16, 84], axis=0)
    ax.fill_between(RBINS, lo_p, hi_p, color="#2ca02c", alpha=0.45, zorder=3,
                    label="PPC 16-84% (population)")
    ax.plot(RBINS, np.median(noiseless, 0), color="#2ca02c", lw=2, ls="--", zorder=4,
            label="PPC median")

    ax.set_xscale("log")
    ax.set_title(LABEL.get(obs_id, obs_id), fontsize=10)
    if k % ncol == 0:
        ax.set_ylabel(r"$\log_{10}\Sigma$ [$M_\odot h/\mathrm{kpc}^2$]")
    if k // ncol == nrow - 1:
        ax.set_xlabel(r"$R$ [kpc/$h$]")
    if k == 0:
        ax.legend(fontsize=7, loc="lower left")

# hide unused axes
for k in range(n, nrow * ncol):
    axes[k // ncol][k % ncol].set_visible(False)

fig.suptitle("Posterior predictive checks: hierarchical HMC population vs. observed profiles", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f"{OUT}/aggregate_ppc.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/aggregate_ppc.png  ({n} experiments)")
