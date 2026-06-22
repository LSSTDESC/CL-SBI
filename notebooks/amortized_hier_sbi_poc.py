"""
Option (c): Amortized hierarchical SBI proof-of-concept.

Idea: unify the two threads of the paper. Use SBI to amortize the *per-cluster* likelihood
(train once: q_phi(M,c | one noisy profile)), then plug those per-cluster posteriors into the SAME
hierarchical population model used for HMC. Because the SBI training prior is BoxUniform, the
per-cluster SBI posterior is proportional to the per-cluster likelihood, so it can be used directly
as the likelihood surrogate in the hierarchy (the constant uniform prior cancels in the population
inference up to an overall normalization).

Pipeline:
  1. Train a single-cluster SNPE q_phi(M,c | profile) on (M,c)->noisy-profile pairs drawn from a
     broad box prior. AMORTIZED: one-time cost.
  2. Evaluate q_phi on each of the N_c=376 baseline observed profiles -> 376 per-cluster posteriors.
     Summarize each by a Gaussian (mean, cov) for a fast differentiable surrogate.
  3. Hierarchical NumPyro model: per-cluster latent (M_j,c_j) ~ population(mu_M,sig_M,c0,beta,sig_c);
     per-cluster "likelihood" = N(theta_j | sbi_mean_j, sbi_cov_j). Sample hyperparameters w/ NUTS.
  4. Compare recovered population spread to TRUE, to direct-SBI (percentile-Gaussian), and to HMC.

This is a POC on the baseline experiment only.
"""
import os, pickle, time
import numpy as np
import torch
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"
sys.path.insert(0, REPO)
from weaklensclustersbi.inference import sbi_
from weaklensclustersbi.simulations import wlprofile

jax.config.update("jax_enable_x64", True)
OUT = f"{REPO}/notebooks/hierarchical_poc_outputs"
np.random.seed(0)
torch.manual_seed(0)

# ---- config (baseline) ----
PRIORS = {
    "min_log10mass": 13.5, "max_log10mass": 15.2,   # broad box for single-cluster training
    "min_concentration": 2.0, "max_concentration": 8.0,
}
Z = 0.275
RBINS = 10 ** np.arange(0, 3.0, 0.1)
N_TRAIN = 20000
PROFILE_NOISE_DEX = 0.3

# ---------------------------------------------------------------------------
# 1. Train single-cluster SNPE  q(M,c | one noisy profile)
# ---------------------------------------------------------------------------
def simulate_single(theta):
    """theta: (logM, c) -> noisy log10 surface-density profile (30,)."""
    logM, c = float(theta[0]), float(theta[1])
    prof = wlprofile.simulate_nfw(logM, c, rbins=RBINS, z=Z)
    logprof = np.log10(prof)
    logprof = logprof + np.random.normal(0, PROFILE_NOISE_DEX, size=logprof.shape)
    return logprof

def train_single_cluster_posterior():
    print(f"## Training single-cluster SNPE on {N_TRAIN} sims ...")
    t0 = time.time()
    lo = np.array([PRIORS["min_log10mass"], PRIORS["min_concentration"]])
    hi = np.array([PRIORS["max_log10mass"], PRIORS["max_concentration"]])
    theta = np.random.uniform(lo, hi, size=(N_TRAIN, 2))
    x = np.array([simulate_single(t) for t in theta])
    inferrer = sbi_.gen_inferrer(PRIORS, param_dim=2)   # BoxUniform prior, MDN
    th = torch.as_tensor(theta, dtype=torch.float32)
    xt = torch.as_tensor(x, dtype=torch.float32)
    de = inferrer.append_simulations(th, xt).train(training_batch_size=100, show_train_summary=False)
    posterior = inferrer.build_posterior(de)
    print(f"## trained in {time.time()-t0:.0f}s (one-time amortized cost)")
    return posterior

# ---------------------------------------------------------------------------
# 2. Per-cluster posteriors on the observed profiles -> Gaussian summaries
# ---------------------------------------------------------------------------
def per_cluster_gaussians(posterior, obs_profiles, n_samp=2000):
    print(f"## Evaluating per-cluster SBI posteriors for {len(obs_profiles)} clusters ...")
    means, covs = [], []
    for j, prof in enumerate(obs_profiles):
        x = torch.as_tensor(prof, dtype=torch.float32)
        s = posterior.sample((n_samp,), x=x, show_progress_bars=False).numpy()
        means.append(s.mean(0)); covs.append(np.cov(s.T))
    return np.array(means), np.array(covs)

# ---------------------------------------------------------------------------
# 3. Hierarchical model using the per-cluster SBI Gaussians as the likelihood
# ---------------------------------------------------------------------------
# Child18 M-c slope over our mass range (dc/dlog10M), computed from colossus; see notebook.
# Used as an informed-but-wide prior on beta to prevent the weakly-identified slope from running
# away and absorbing concentration scatter (Fix B).
CHILD18_BETA = -0.855

def run_hier(means, covs, informed_beta=True, beta_center=CHILD18_BETA, beta_width=0.5,
             sigc_prior_scale=0.3):
    """Hierarchical population inference using per-cluster SBI Gaussians as the likelihood.

    Fix B (informed_beta=True):
      - beta ~ Normal(beta_center, beta_width): informed by the paper's Child18 m-c relation but
        deliberately wide, to stop the runaway (unconstrained run gave beta=-2.1 vs Child18 -0.86).
      - sig_c ~ HalfNormal(sigc_prior_scale): tighter so the weakly-identified population scatter
        is not pulled up by a diffuse hyperprior.
    Set informed_beta=False to reproduce the original diffuse-prior behavior for comparison.
    """
    N_c = len(means)
    means_j = jnp.array(means); icovs_j = jnp.array(np.linalg.inv(covs))

    def model():
        mu_M = numpyro.sample("mu_M", dist.Normal(14.4, 0.4))
        sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.4))
        c0 = numpyro.sample("c0", dist.Normal(4.6, 1.5))
        if informed_beta:
            beta = numpyro.sample("beta", dist.Normal(beta_center, beta_width))
            sig_c = numpyro.sample("sig_c", dist.HalfNormal(sigc_prior_scale))
        else:
            beta = numpyro.sample("beta", dist.Normal(0.0, 2.0))
            sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.7))
        with numpyro.plate("clusters", N_c):
            zM = numpyro.sample("zM", dist.Normal(0, 1)); logM = mu_M + sig_M * zM
            zc = numpyro.sample("zc", dist.Normal(0, 1)); c = c0 + beta * (logM - mu_M) + sig_c * zc
            theta = jnp.stack([logM, c], axis=-1)                  # (N_c,2)
            # per-cluster SBI likelihood surrogate: N(theta | sbi_mean_j, sbi_cov_j)
            d = theta - means_j
            quad = jnp.einsum("ni,nij,nj->n", d, icovs_j, d)
            numpyro.factor("sbi_like", -0.5 * quad)
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median)
    mcmc = MCMC(kernel, num_warmup=600, num_samples=800, num_chains=1, progress_bar=False)
    t0 = time.time(); mcmc.run(jax.random.PRNGKey(0)); t = time.time() - t0
    p = mcmc.get_samples()
    return {k: (float(np.mean(p[k])), float(np.std(p[k]))) for k in ["mu_M","sig_M","c0","beta","sig_c"]}, t


if __name__ == "__main__":
    obs = np.load(f"{REPO}/outputs/observations/obs_z1_lambda5.376/drawn_nfw_profiles.npy")
    true_mc = np.load(f"{REPO}/outputs/observations/obs_z1_lambda5.376/drawn_mc_pairs.npy")

    posterior = train_single_cluster_posterior()
    means, covs = per_cluster_gaussians(posterior, obs)
    print(f"## per-cluster posterior: mean width sig(logM)={np.sqrt(covs[:,0,0]).mean():.3f}, "
          f"sig(c)={np.sqrt(covs[:,1,1]).mean():.3f}")

    # moment-based deconvolution sanity check: Var(per-cluster means) - mean(meas var)
    def decon(i):
        v = means[:, i].var() - covs[:, i, i].mean()
        return np.sqrt(v) if v > 0 else float("nan")
    print(f"\n## moment deconvolution: sig_logM={decon(0):.3f}, sig_c={decon(1):.3f} "
          f"(true {true_mc[:,0].std():.3f}, {true_mc[:,1].std():.3f})")

    summ_diffuse, t1 = run_hier(means, covs, informed_beta=False)
    summ_informed, t2 = run_hier(means, covs, informed_beta=True)

    def report(tag, summ, t):
        print(f"\n=== {tag} (sampling {t:.0f}s) ===")
        for k, (m, s) in summ.items():
            print(f"  {k:7s} = {m:.4f} +/- {s:.4f}")

    report("Diffuse prior (original)", summ_diffuse, t1)
    report("Informed-beta + tight sig_c (Fix B)", summ_informed, t2)
    print(f"\nTRUE: mu_logM={true_mc[:,0].mean():.4f} sig_logM={true_mc[:,0].std():.4f} "
          f"mu_c={true_mc[:,1].mean():.4f} sig_c={true_mc[:,1].std():.4f}")
    print(f"Child18 beta = {CHILD18_BETA}")
    pickle.dump({"summary_informed": summ_informed, "summary_diffuse": summ_diffuse,
                 "means": means, "covs": covs, "true_mc": true_mc,
                 "decon": (decon(0), decon(1))},
                open(f"{OUT}/amortized_hier_sbi.pkl", "wb"))
    print(f"saved -> {OUT}/amortized_hier_sbi.pkl")
