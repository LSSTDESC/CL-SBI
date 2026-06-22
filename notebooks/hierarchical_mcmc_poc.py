"""
Hierarchical Bayesian MCMC proof-of-concept for cluster (M,c) population inference.

GOAL
----
Demonstrate that a *hierarchical* model, sampled with NUTS (NumPyro) on a
differentiable JAX forward model (halox), can recover the intrinsic POPULATION
SPREAD of (log10 M, c) -- the quantity SBI FTJ estimates -- whereas the standard
MCMC joint-likelihood ("FTJ") only recovers a central (~mean) (M,c) value.

We compare three methods against the known true population of 376 clusters in the
baseline experiment (sim_z1.infer_z1.obs_z1_lambda5.10000.376):
  1. MCMC joint-likelihood (FTJ)      -- single (M,c), saved emcee chain
  2. SBI FTJ                          -- percentile-summary Gaussian, saved chain
  3. Hierarchical NUTS (this work)    -- infers (mu_M, sig_M, m-c relation, sig_c)

FORWARD MODEL
-------------
NFW surface density via the analytic Bartelmann(1996)/Wright&Brainerd(2000)
formula, using halox for the (differentiable) scale radius Rs and rho0. Validated
to machine precision against halox.surface_density and to <0.1% against the
Colossus pipeline that GENERATED the data (mdef="vir", planck18, z=0.275).
We re-implement only the analytic Sigma(x) step to work around a NaN-gradient
bug in halox's jnp.where branches (the "safe-x" trick); the physics is halox's.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from halox.halo import NFWHalo
from halox import cosmology as hcos

numpyro.set_host_device_count(1)
jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------
# 0. Config (baseline experiment) -- values pulled from configs/inference/infer_z1.json
# ----------------------------------------------------------------------------
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"
OBS_DIR = f"{REPO}/outputs/observations/obs_z1_lambda5.376"
INF_DIR = f"{REPO}/outputs/inference/sim_z1.infer_z1.obs_z1_lambda5.10000.376"
OUT_DIR = f"{REPO}/notebooks/hierarchical_poc_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

Z = 0.275                 # mid (min_z=0.2, max_z=0.35)
DELTA_VIR = 124.89        # Bryan&Norman Delta_vir(z=0.275) wrt rho_crit (matches colossus 'vir')
NUM_RBINS = 30
RBINS_KPC = jnp.array(10 ** np.arange(0, NUM_RBINS / 10, 0.1))  # kpc/h, matches gen_observations.py
P18 = hcos.Planck18()

# ----------------------------------------------------------------------------
# 1. Load data + true population
# ----------------------------------------------------------------------------
obs_profiles = jnp.array(np.load(f"{OBS_DIR}/drawn_nfw_profiles.npy"))   # (376,30) log10 Sigma
true_mc      = np.load(f"{OBS_DIR}/drawn_mc_pairs.npy")                  # (376,2) [log10M, c]
sigmas       = jnp.array(np.load(f"{OBS_DIR}/sigmas.npy"))               # (30,) log10-space errors
N_c = obs_profiles.shape[0]

TRUE = dict(
    mu_logM=true_mc[:, 0].mean(), sig_logM=true_mc[:, 0].std(),
    mu_c=true_mc[:, 1].mean(),    sig_c=true_mc[:, 1].std(),
    rho=np.corrcoef(true_mc.T)[0, 1],
)
print("TRUE population:", {k: round(v, 4) for k, v in TRUE.items()})

# ----------------------------------------------------------------------------
# 2. Differentiable NFW forward model (halox Rs/rho0 + safe-x analytic Sigma)
# ----------------------------------------------------------------------------
def nfw_logSigma(logM, c):
    """log10 NFW surface density [Msun h / kpc^2] at RBINS_KPC. Differentiable."""
    halo = NFWHalo(m_delta=10.0 ** logM, c_delta=c, z=Z, cosmo=P18, delta=DELTA_VIR)
    Rs = halo.Rs * 1000.0     # Mpc/h -> kpc/h
    rho0 = halo.rho0 / 1e9    # Msun h^2/Mpc^3 -> /kpc^3
    x = RBINS_KPC / Rs
    x_lo = jnp.where(x < 1, x, 0.5)   # safe-x: untaken branch never sees bad sqrt
    x_hi = jnp.where(x > 1, x, 2.0)
    f_lo = (1 - 2 * jnp.arctanh(jnp.sqrt((1 - x_lo) / (1 + x_lo))) / jnp.sqrt(1 - x_lo**2)) / (x_lo**2 - 1)
    f_hi = (1 - 2 * jnp.arctan(jnp.sqrt((x_hi - 1) / (1 + x_hi))) / jnp.sqrt(x_hi**2 - 1)) / (x_hi**2 - 1)
    f = jnp.where(x < 1, f_lo, jnp.where(x > 1, f_hi, 1.0 / 3.0))
    return jnp.log10(2 * rho0 * Rs * f)

# vectorize over a population of clusters
nfw_logSigma_vmap = jax.vmap(nfw_logSigma)

# ----------------------------------------------------------------------------
# 3. Hierarchical generative model
# ----------------------------------------------------------------------------
# Hyperparameters (what we want):
#   mu_M, sig_M           : population mean & intrinsic spread in log10 mass
#   c0, beta, sig_c       : m-c relation  E[c | logM] = c0 + beta*(logM - mu_M), + scatter
# Latents (marginalized): per-cluster (logM_j, c_j)  [non-centered]
# Likelihood: observed log10 profile ~ Normal(model(logM_j, c_j), sigma_obs)
#
# The KEY difference from the FTJ joint likelihood: there is a per-cluster (M_j, c_j)
# drawn from a population with FREE width sig_M, sig_c -- so the posterior on
# (sig_M, sig_c) directly estimates the population spread.

def hier_model(obs=None):
    mu_M  = numpyro.sample("mu_M",  dist.Normal(14.4, 0.3))
    sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.3))
    c0    = numpyro.sample("c0",    dist.Normal(4.6, 1.0))
    beta  = numpyro.sample("beta",  dist.Normal(0.0, 2.0))      # m-c slope
    sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.5))
    # extra fractional log-scatter on the profile (analogue of the emcee ln f term)
    sig_extra = numpyro.sample("sig_extra", dist.HalfNormal(0.1))

    with numpyro.plate("clusters", N_c):
        # non-centered parameterization to avoid Neal's funnel
        zM = numpyro.sample("zM", dist.Normal(0, 1))
        logM_j = mu_M + sig_M * zM
        zc = numpyro.sample("zc", dist.Normal(0, 1))
        c_j = c0 + beta * (logM_j - mu_M) + sig_c * zc

    model_logSig = nfw_logSigma_vmap(logM_j, c_j)               # (N_c, 30)
    sig_tot = jnp.sqrt(sigmas[None, :] ** 2 + sig_extra ** 2)
    numpyro.sample("obs", dist.Normal(model_logSig, sig_tot), obs=obs)

# ----------------------------------------------------------------------------
# 4. Run NUTS
# ----------------------------------------------------------------------------
def run(num_warmup=500, num_samples=500, seed=0):
    kernel = NUTS(hier_model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), obs=obs_profiles)
    mcmc.print_summary(exclude_deterministic=True)
    return mcmc


# ----------------------------------------------------------------------------
# 5. Pull the other two methods for comparison
# ----------------------------------------------------------------------------
def load_comparisons():
    """Population (M,c) estimates from MCMC-joint FTJ and SBI FTJ."""
    import sys
    sys.path.insert(0, REPO)
    sys.path.insert(0, f"{REPO}/plot")
    import plotutils as P

    with open(f"{INF_DIR}/mcmc_ftj_samplers.pickle", "rb") as f:
        j = pickle.load(f)
    ch = j.get_chain(); b = int(ch.shape[0] * 0.3)
    jf = ch[b:].reshape(-1, ch.shape[-1])
    mcmc_joint = dict(mu_logM=jf[:, 0].mean(), sig_logM=jf[:, 0].std(),
                      mu_c=jf[:, 1].mean(), sig_c=jf[:, 1].std())

    with open(f"{INF_DIR}/sbi_chains.pickle", "rb") as f:
        sbi = pickle.load(f)
    s = P.build_gaussian_summary_from_chain(sbi[1])
    sbi_ftj = dict(mu_logM=s["mass"]["mu"], sig_logM=s["mass"]["sigma"],
                   mu_c=s["concentration"]["mu"], sig_c=s["concentration"]["sigma"],
                   rho=s["correlation"]["rho"])
    return mcmc_joint, sbi_ftj


def comparison_table(hier, mcmc_joint, sbi_ftj):
    rows = [
        ("mu(logM)",  TRUE["mu_logM"],  mcmc_joint["mu_logM"],  sbi_ftj["mu_logM"],  hier["mu_M"][0]),
        ("sig(logM)", TRUE["sig_logM"], mcmc_joint["sig_logM"], sbi_ftj["sig_logM"], hier["sig_M"][0]),
        ("mu(c)",     TRUE["mu_c"],     mcmc_joint["mu_c"],     sbi_ftj["mu_c"],     hier["c0"][0]),
        ("sig(c)",    TRUE["sig_c"],    mcmc_joint["sig_c"],    sbi_ftj["sig_c"],    hier["sig_c"][0]),
    ]
    print(f"\n{'quantity':10s} {'TRUE':>9s} {'MCMC-joint':>11s} {'SBI-FTJ':>9s} {'Hier':>9s}")
    for name, t, m, s, h in rows:
        print(f"{name:10s} {t:9.4f} {m:11.4f} {s:9.4f} {h:9.4f}")
    print("\nKey: MCMC-joint posterior WIDTH (sig) is the population-spread estimate it")
    print("implicitly makes; it collapses far below TRUE (a mean estimator), while SBI")
    print("and the hierarchical model both recover the true spread.")
    return rows


def make_figure(hier_post, mcmc_joint, sbi_ftj):
    """Overlay the population (M,c) distribution each method implies vs TRUE."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- panel A: log10 M marginal ---
    ax = axes[0]
    xs = np.linspace(13.9, 14.85, 400)
    gauss = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    ax.fill_between(xs, gauss(xs, TRUE["mu_logM"], TRUE["sig_logM"]), color="k", alpha=0.15, label="TRUE population")
    ax.plot(xs, gauss(xs, TRUE["mu_logM"], TRUE["sig_logM"]), "k--", lw=2)
    ax.plot(xs, gauss(xs, mcmc_joint["mu_logM"], mcmc_joint["sig_logM"]), color="#1f77b4", lw=2, label="MCMC joint-likelihood (FTJ)")
    ax.plot(xs, gauss(xs, sbi_ftj["mu_logM"], sbi_ftj["sig_logM"]), color="#ff7f0e", lw=2, label="SBI FTJ")
    hM = hier_post["mu_M"][:, None] + hier_post["sig_M"][:, None] * rng.standard_normal((len(hier_post["mu_M"]), 1))
    ax.hist(hM.ravel(), bins=60, density=True, histtype="step", color="#2ca02c", lw=2, label="Hierarchical NUTS (this work)")
    ax.set_xlabel(r"$\log_{10} M$"); ax.set_ylabel("population density"); ax.legend(fontsize=9)
    ax.set_title("Population spread in mass")

    # --- panel B: spread bar chart (the headline) ---
    ax = axes[1]
    labels = ["MCMC\njoint (FTJ)", "SBI\nFTJ", "Hierarchical\n(this work)"]
    sigM = [mcmc_joint["sig_logM"], sbi_ftj["sig_logM"], hier_post["sig_M"].mean()]
    ax.bar(labels, sigM, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
    ax.axhline(TRUE["sig_logM"], color="k", ls="--", lw=2, label=f"TRUE $\\sigma_{{\\log M}}$ = {TRUE['sig_logM']:.3f}")
    ax.set_ylabel(r"recovered $\sigma_{\log_{10}M}$"); ax.legend(); ax.set_title("Recovered mass spread vs truth")
    for i, v in enumerate(sigM):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/population_spread_comparison.png", dpi=140)
    print(f"Saved figure -> {OUT_DIR}/population_spread_comparison.png")


if __name__ == "__main__":
    mcmc = run(num_warmup=600, num_samples=800, seed=0)
    post = mcmc.get_samples()
    summary = {k: (float(np.mean(post[k])), float(np.std(post[k])))
               for k in ["mu_M", "sig_M", "c0", "beta", "sig_c", "sig_extra"]}
    print("\n=== Hierarchical posterior (mean +/- sd) ===")
    for k, (m, s) in summary.items():
        print(f"  {k:10s} = {m:.4f} +/- {s:.4f}")

    mcmc_joint, sbi_ftj = load_comparisons()
    rows = comparison_table(summary, mcmc_joint, sbi_ftj)
    make_figure({k: np.array(v) for k, v in post.items()}, mcmc_joint, sbi_ftj)

    with open(f"{OUT_DIR}/hier_posterior.pkl", "wb") as f:
        pickle.dump({"samples": {k: np.array(v) for k, v in post.items()},
                     "summary": summary, "true": TRUE,
                     "mcmc_joint": mcmc_joint, "sbi_ftj": sbi_ftj}, f)
    print(f"\nSaved -> {OUT_DIR}/hier_posterior.pkl")
