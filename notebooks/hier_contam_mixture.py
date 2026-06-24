"""
Two-component mixture population for the richness-contamination experiments.

The richness-contamination stacks (low/high_richness_contam) contain interlopers that scattered UP
from lower-richness (lower-mass) bins, so the in-bin mass distribution has a low-mass tail. A single
Gaussian population is mis-specified there (it recovers sigma_M but biases the M-c relation -- see the
Hierarchical SBI limitations). Here we model the mass population as a 2-component Gaussian mixture:

    logM_j ~ (1-f) * N(mu_main, sig_main)  +  f * N(mu_contam, sig_contam),   mu_contam < mu_main

with f the contamination fraction (free). The concentration sector (c0, beta, sig_c) is shared, as in
the baseline hierarchical HMC. Compares the mixture fit to the single-Gaussian fit on the same data.

Implementation: marginalize the discrete component assignment analytically (logsumexp over the two
components' log-densities for each cluster's latent logM), so NUTS sees a smooth log-density.

Run under base env:
  /Users/akumgill/anaconda3/bin/python notebooks/hier_contam_mixture.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle, numpy as np
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary
import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"; sys.path.insert(0, REPO)
import hierarchical_mcmc_poc as H
jax.config.update("jax_enable_x64", True)
OUT = H.OUT_DIR
fwd = jax.vmap(H.nfw_logSigma)
EXPS = ["obs_z1_lambda5_low_richness_contam", "obs_z1_lambda5_high_richness_contam"]


def load(e):
    d = f"{REPO}/outputs/observations/{e}.376"
    return (jnp.array(np.load(f"{d}/drawn_nfw_profiles.npy")), jnp.array(np.load(f"{d}/sigmas.npy")),
            np.load(f"{d}/drawn_mc_pairs.npy"))


def make_models(prof, sig, N_c):
    def conc_like(logM):
        c0 = numpyro.sample("c0", dist.Normal(4.6, 1.0))
        beta = numpyro.sample("beta", dist.Normal(0.0, 2.0))
        sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.5))
        sig_extra = numpyro.sample("sig_extra", dist.HalfNormal(0.1))
        with numpyro.plate("cl_c", N_c):
            zc = numpyro.sample("zc", dist.Normal(0, 1))
            c = c0 + beta * (logM - jnp.mean(logM)) + sig_c * zc
        sig_tot = jnp.sqrt(sig[None, :] ** 2 + sig_extra ** 2)
        numpyro.sample("obs", dist.Normal(fwd(logM, c), sig_tot), obs=prof)

    def free_model():
        mu_M = numpyro.sample("mu_M", dist.Normal(14.3, 0.4)); sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.4))
        with numpyro.plate("cl_m", N_c):
            zM = numpyro.sample("zM", dist.Normal(0, 1)); logM = mu_M + sig_M * zM
        conc_like(logM)

    def mixture_model():
        # main + contaminant components; contaminant sits below the main mean by a free positive gap
        mu_main = numpyro.sample("mu_main", dist.Normal(14.4, 0.3))
        sig_main = numpyro.sample("sig_main", dist.HalfNormal(0.3))
        gap = numpyro.sample("gap", dist.HalfNormal(0.4))            # mu_contam = mu_main - gap (<0 enforced)
        sig_contam = numpyro.sample("sig_contam", dist.HalfNormal(0.4))
        f = numpyro.sample("f", dist.Beta(1.5, 5.0))                 # contamination fraction, prior favors small
        mu_contam = mu_main - gap
        # per-cluster latent logM with a 2-component mixture PRIOR (marginalized assignment)
        with numpyro.plate("cl_m", N_c):
            logM = numpyro.sample("logM", dist.Normal(14.3, 0.6))    # broad base; mixture enters as factor
            lp_main = jnp.log1p(-f) + dist.Normal(mu_main, sig_main).log_prob(logM)
            lp_cont = jnp.log(f) + dist.Normal(mu_contam, sig_contam).log_prob(logM)
            numpyro.factor("mix", jnp.logaddexp(lp_main, lp_cont))
        conc_like(logM)
        # report the effective population mean/spread of the mixture
        numpyro.deterministic("mu_eff", (1 - f) * mu_main + f * mu_contam)
        numpyro.deterministic("sig_eff", jnp.sqrt((1 - f) * (sig_main**2 + mu_main**2) + f * (sig_contam**2 + mu_contam**2)
                                                  - ((1 - f) * mu_main + f * mu_contam) ** 2))
    return free_model, mixture_model


def run(m, keys, warm=800, samp=1000, chains=2, seed=0):
    mc = MCMC(NUTS(m, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median),
              num_warmup=warm, num_samples=samp, num_chains=chains, chain_method="sequential", progress_bar=False)
    t = time.time(); mc.run(jax.random.PRNGKey(seed)); mc.get_samples()["beta"].block_until_ready(); t = time.time() - t
    p = mc.get_samples(); g = mc.get_samples(group_by_chain=True)
    diag = nps_summary({k: np.array(g[k]) for k in keys if k in g}, prob=0.9)
    rhat = max(float(diag[k]["r_hat"]) for k in diag)
    return {k: (float(p[k].mean()), float(p[k].std())) for k in keys if k in p}, rhat, t


if __name__ == "__main__":
    results = {}
    for e in EXPS:
        prof, sig, true_mc = load(e); N_c = prof.shape[0]
        lm, c = true_mc[:, 0], true_mc[:, 1]
        bt = float(np.polyfit(lm - lm.mean(), c, 1)[0])
        truth = dict(mu_M=float(lm.mean()), sig_M=float(lm.std()), beta=bt,
                     sig_c=float((c - (c.mean() + bt * (lm - lm.mean()))).std()))
        free_model, mixture_model = make_models(prof, sig, N_c)
        print(f"\n##### {e} (N_c={N_c}, true muM={truth['mu_M']:.3f} sigM={truth['sig_M']:.3f} beta={bt:.3f}) #####", flush=True)
        free_est, free_r, _ = run(free_model, ["mu_M", "sig_M", "c0", "beta", "sig_c"])
        mix_est, mix_r, _ = run(mixture_model, ["mu_eff", "sig_eff", "f", "mu_main", "gap", "c0", "beta", "sig_c"])
        results[e] = dict(truth=truth, free=free_est, mix=mix_est, free_rhat=free_r, mix_rhat=mix_r)
        print(f"  truth:   muM={truth['mu_M']:.3f} sigM={truth['sig_M']:.3f} beta={truth['beta']:.3f} sigc={truth['sig_c']:.3f}")
        print(f"  free:    muM={free_est['mu_M'][0]:.3f} sigM={free_est['sig_M'][0]:.3f} beta={free_est['beta'][0]:.3f} sigc={free_est['sig_c'][0]:.3f} (rhat={free_r:.2f})")
        print(f"  mixture: muEff={mix_est['mu_eff'][0]:.3f} sigEff={mix_est['sig_eff'][0]:.3f} beta={mix_est['beta'][0]:.3f} sigc={mix_est['sig_c'][0]:.3f} f={mix_est['f'][0]:.2f} (rhat={mix_r:.2f})")
    pickle.dump(results, open(f"{OUT}/contam_mixture.pkl", "wb"))
    print(f"\nsaved -> {OUT}/contam_mixture.pkl")
