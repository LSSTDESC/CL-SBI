"""
Multi-bin joint hierarchical fit: the principled fix for the M-c slope beta.

A single richness bin spans only sigma_M ~ 0.12 dex in mass, so the slope beta = dc/dlogM has almost
no lever arm and is unconstrained (Section beta / fig:beta_leverarm). Pooling ALL 7 richness bins
spans ~2.1 dex (13.07 -> 15.15), an ~11x wider mass baseline, which should pin beta down.

Model: ONE shared M-c relation (c0, beta, sig_c) across all bins; each bin k has its OWN mass
population, modeled as the bin's mass-function x richness-selection window (Tinker08 x McClintock18,
the validated machinery from hier_massfn_informed.py) with a free per-bin mean shift + width scaling.
This handles the selection effect correctly: each bin selects a different mass range, and beta is
constrained by how the per-bin mean concentration tracks the per-bin mean mass ACROSS bins, plus the
within-bin (M,c) covariance. Profiles in every bin share the same (c0,beta,sig_c).

Compares the joint-fit beta to the single-bin (baseline-only) beta to show the lever-arm improvement.

Run under base env (slow: ~2600 per-cluster latents x NUTS):
  /Users/akumgill/anaconda3/bin/python notebooks/hier_multibin_beta.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle, numpy as np
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary
from scipy.stats import norm as scipy_norm
import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"; sys.path.insert(0, REPO)
import hierarchical_mcmc_poc as H
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from weaklensclustersbi.simulations import populationutils
jax.config.update("jax_enable_x64", True)

OUT = H.OUT_DIR; Z = H.Z
RM_RELATION = "mcclintock18"; MF_MODEL = "tinker08"; MF_MDEF = "200m"
cosmology.setCosmology("planck18")
F = float(populationutils.get_rm_slope(RM_RELATION)) if hasattr(populationutils, "get_rm_slope") else 1.356
RM_SCATTER = 0.1
LAMBDA_BINS = [(5, 10), (10, 14), (14, 20), (20, 30), (30, 45), (45, 60), (60, 100)]
BIN_IDS = [f"obs_z1_lambda{b}" for b in range(1, 8)]


def bin_mf_logpdf(lmin, lmax):
    """MF x richness-selection population log-pdf for one richness bin, on a logM grid."""
    grid = np.linspace(12.8, 15.5, 1400)
    dndlnM = mass_function.massFunction(10 ** grid, Z, mdef=MF_MDEF, model=MF_MODEL, q_out="dndlnM")
    sigma_lnlam = RM_SCATTER * np.log(10.0) / F
    lam_exp = populationutils.get_richness(grid, z=Z, model=RM_RELATION)
    log_lam_exp = np.log(lam_exp)
    p_in = (scipy_norm.cdf(np.log(lmax), log_lam_exp, sigma_lnlam)
            - scipy_norm.cdf(np.log(lmin), log_lam_exp, sigma_lnlam))
    pdf = dndlnM * np.log(10.0) * np.clip(p_in, 1e-12, None)
    pdf = np.clip(pdf, 1e-300, None)
    logpdf = np.log(pdf); logpdf -= np.log(np.trapz(np.exp(logpdf), grid))
    return grid, logpdf


# ---- load all bins; build per-bin forward-model data + MF priors ----
print("## loading 7 richness bins + building per-bin MF x selection priors ...", flush=True)
BINS = []
for bid, (lmin, lmax) in zip(BIN_IDS, LAMBDA_BINS):
    d = f"{REPO}/outputs/observations/{bid}.376"
    prof = jnp.array(np.load(f"{d}/drawn_nfw_profiles.npy"))
    sig = jnp.array(np.load(f"{d}/sigmas.npy"))
    true_mc = np.load(f"{d}/drawn_mc_pairs.npy")
    grid, logpdf = bin_mf_logpdf(lmin, lmax)
    BINS.append(dict(bid=bid, prof=prof, sig=sig, n=prof.shape[0],
                     grid=jnp.array(grid), logpdf=jnp.array(logpdf),
                     true_mu=float(true_mc[:, 0].mean())))
N_TOT = sum(b["n"] for b in BINS)
fwd = jax.vmap(H.nfw_logSigma)
# true global slope (cross-bin), for reference
allmc = np.concatenate([np.load(f"{REPO}/outputs/observations/{b['bid']}.376/drawn_mc_pairs.npy") for b in BINS])
TRUE_BETA = float(np.polyfit(allmc[:, 0] - allmc[:, 0].mean(), allmc[:, 1], 1)[0])
TRUE_MU_ALL = float(allmc[:, 0].mean())
print(f"## N_tot={N_TOT}, true cross-bin beta={TRUE_BETA:.3f}", flush=True)


def model():
    # ONE shared M-c relation across all bins
    c0 = numpyro.sample("c0", dist.Normal(4.8, 1.5))
    beta = numpyro.sample("beta", dist.Normal(0.0, 2.0))
    sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.7))
    sig_extra = numpyro.sample("sig_extra", dist.HalfNormal(0.15))
    for k, b in enumerate(BINS):
        # per-bin mass population: MF x selection shape (informative), with a free mean-shift dmu_k
        # and width-scale s_k so the bin can adapt while staying anchored to the physical prior.
        dmu = numpyro.sample(f"dmu_{k}", dist.Normal(0.0, 0.15))
        with numpyro.plate(f"cl_{k}", b["n"]):
            # latent mass: base draw near the MF-mode + free offset, with MF x selection as a factor
            zM = numpyro.sample(f"zM_{k}", dist.Normal(0, 1))
            logM = b["grid"][jnp.argmax(b["logpdf"])] + dmu + 0.15 * zM
            numpyro.factor(f"mfprior_{k}", jnp.interp(logM, b["grid"], b["logpdf"]))
            zc = numpyro.sample(f"zc_{k}", dist.Normal(0, 1))
            c = c0 + beta * (logM - TRUE_MU_ALL) + sig_c * zc
        model_logSig = fwd(logM, c)
        sig_tot = jnp.sqrt(b["sig"][None, :] ** 2 + sig_extra ** 2)
        numpyro.sample(f"obs_{k}", dist.Normal(model_logSig, sig_tot), obs=b["prof"])


def run(warm=800, samp=1000, chains=2, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median),
              num_warmup=warm, num_samples=samp, num_chains=chains, chain_method="sequential",
              progress_bar=True)
    t = time.time(); mc.run(jax.random.PRNGKey(seed)); mc.get_samples()["beta"].block_until_ready()
    t = time.time() - t
    p = mc.get_samples(); g = mc.get_samples(group_by_chain=True)
    keys = ["c0", "beta", "sig_c", "sig_extra"]
    diag = nps_summary({k: np.array(g[k]) for k in keys}, prob=0.9)
    rhat = max(float(diag[k]["r_hat"]) for k in keys)
    est = {k: (float(p[k].mean()), float(p[k].std())) for k in keys}
    return est, rhat, t


if __name__ == "__main__":
    t0 = time.time()
    est, rhat, t = run()
    # single-bin (baseline-only) beta, from the Hierarchical SBI run, for the lever-arm contrast
    try:
        sb = pickle.load(open(f"{OUT}/afull_neural_hbi.pkl", "rb"))["experiments"]["obs_z1_lambda5"]["est"]
        single_beta = sb["beta"]
    except Exception:
        single_beta = (float("nan"), float("nan"))
    print(f"\n=== MULTI-BIN JOINT FIT (N_tot={N_TOT}, 7 bins, rhat={rhat:.3f}, {t:.0f}s) ===")
    for k, (m, s) in est.items():
        print(f"  {k:9s} = {m:7.3f} +/- {s:.3f}")
    print(f"\n  beta (joint multi-bin) = {est['beta'][0]:.3f} +/- {est['beta'][1]:.3f}")
    print(f"  beta (single bin, baseline, Hier-SBI) = {single_beta[0]:.3f} +/- {single_beta[1]:.3f}")
    print(f"  TRUE cross-bin beta = {TRUE_BETA:.3f}")
    impr = single_beta[1] / est['beta'][1] if est['beta'][1] > 0 else float('nan')
    print(f"  -> beta posterior {impr:.1f}x tighter than single-bin" if np.isfinite(impr) else "")
    pickle.dump(dict(est=est, rhat=rhat, t=t, true_beta=TRUE_BETA, single_bin_beta=single_beta,
                     n_tot=N_TOT, true_mu_all=TRUE_MU_ALL),
                open(f"{OUT}/multibin_beta.pkl", "wb"))
    print(f"\n=== total wall {(time.time()-t0)/60:.1f} min ===  saved -> {OUT}/multibin_beta.pkl")
