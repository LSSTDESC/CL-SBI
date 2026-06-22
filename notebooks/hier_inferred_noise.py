"""
Hierarchical HMC variant that INFERS the profile noise (per radial bin), rather than assuming the
known per-bin sigmas. Tests two questions:
  1. Does freeing the noise still recover the population mass spread sig_M?  (identifiability)
  2. Does the inferred per-bin noise match the true ~0.3 dex (flat) noise used to generate the data?

We free a per-bin log-noise vector sigma_noise[i] (30 values) with a weakly-informative prior, and
drop the previous fixed `sigmas` + scalar `sig_extra`. The radial SHAPE information is what lets the
model separate white per-bin measurement noise from coherent NFW-shape population scatter.
"""
import os, pickle, time
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary

import hierarchical_mcmc_poc as H

jax.config.update("jax_enable_x64", True)
OUT = H.OUT_DIR
REPO = H.REPO


def model_inferred_pernbin(obs, n_rbins):
    N_c = obs.shape[0]
    mu_M  = numpyro.sample("mu_M",  dist.Normal(14.4, 0.3))
    sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.3))
    c0    = numpyro.sample("c0",    dist.Normal(4.6, 1.0))
    beta  = numpyro.sample("beta",  dist.Normal(0.0, 2.0))
    sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.5))
    # INFER per-bin profile noise: 30 free values, weakly-informative half-normal (scale 0.5 dex
    # comfortably covers the true ~0.3). This replaces the fixed `sigmas` + scalar sig_extra.
    with numpyro.plate("rbins", n_rbins):
        sigma_noise = numpyro.sample("sigma_noise", dist.HalfNormal(0.5))
    with numpyro.plate("clusters", N_c):
        zM = numpyro.sample("zM", dist.Normal(0, 1)); logM_j = mu_M + sig_M * zM
        zc = numpyro.sample("zc", dist.Normal(0, 1)); c_j = c0 + beta * (logM_j - mu_M) + sig_c * zc
    model_logSig = H.nfw_logSigma_vmap(logM_j, c_j)            # (N_c, n_rbins)
    numpyro.sample("obs", dist.Normal(model_logSig, sigma_noise[None, :]), obs=obs)


def run(obs_id="obs_z1_lambda5", nwarm=1000, nsamp=1500, nchains=4, seed=0):
    obs = jnp.array(np.load(f"{REPO}/outputs/observations/{obs_id}.376/drawn_nfw_profiles.npy"))
    true_sig = np.load(f"{REPO}/outputs/observations/{obs_id}.376/sigmas.npy")
    true_mc = np.load(f"{REPO}/outputs/observations/{obs_id}.376/drawn_mc_pairs.npy")
    n_rbins = obs.shape[1]

    kernel = NUTS(lambda obs=None: model_inferred_pernbin(obs, n_rbins),
                  target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median)
    mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, num_chains=nchains,
                chain_method="sequential", progress_bar=False)
    t0 = time.time(); mcmc.run(jax.random.PRNGKey(seed), obs=obs)
    mcmc.get_samples()["mu_M"].block_until_ready(); t = time.time() - t0

    post = mcmc.get_samples()
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = nps_summary({k: np.array(grouped[k]) for k in ["mu_M", "sig_M", "c0", "beta", "sig_c"]}, prob=0.9)
    rhat = max(float(diag[k]["r_hat"]) for k in diag)

    sig_noise_post = np.array(post["sigma_noise"])  # (nsamp*nchains, n_rbins)
    summ = {k: (float(np.mean(post[k])), float(np.std(post[k]))) for k in ["mu_M","sig_M","c0","beta","sig_c"]}
    print(f"\n=== Inferred per-bin noise [{obs_id}]  ({t:.0f}s, rhat_max={rhat:.4f}) ===")
    for k, (m, s) in summ.items():
        print(f"  {k:7s} = {m:.4f} +/- {s:.4f}")
    print(f"  sig_M true = {true_mc[:,0].std():.4f} | sig_c true = {true_mc[:,1].std():.4f}")
    print(f"  inferred noise: mean over bins = {sig_noise_post.mean():.4f} "
          f"(true mean {true_sig.mean():.4f}); per-bin recovery RMS err = "
          f"{np.sqrt(np.mean((sig_noise_post.mean(0)-true_sig)**2)):.4f}")
    pickle.dump({"summary": summ, "sigma_noise": sig_noise_post, "true_sig": true_sig,
                 "true_mc": true_mc, "rhat_max": rhat, "t": t},
                open(f"{OUT}/hier_inferred_noise_{obs_id}.pkl", "wb"))
    return summ, sig_noise_post, true_sig


if __name__ == "__main__":
    run("obs_z1_lambda5")
