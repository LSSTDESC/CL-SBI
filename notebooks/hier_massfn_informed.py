"""
ABLATION: mass-function-informed population vs free-Gaussian population in the
hierarchical HMC, on the HIGH-lambda-M-scatter experiment.

Motivation
----------
The Paper-2 hierarchical methods DELIBERATELY use a *free Gaussian* mass population
("infer the population, don't assume it"). The discussion only SPECULATES that a
mass-function (+ richness-selection) informed population family would (a) better
capture the genuinely non-Gaussian in-bin mass distribution -- which under high R-M
scatter becomes flat-topped / negative-kurtosis -- and (b) help pin down weakly
identified parameters such as the M-c slope beta. This script turns that speculation
into a real number on ONE experiment: obs_z1_lambda5_high_rm_scatter.376 (rm_scatter=0.7),
the case where the true in-bin mass distribution is most non-Gaussian and beta is
recovered worst.

Honest framing: the informed population REINTRODUCES exactly the modeling assumption
(Tinker08 halo mass function + McClintock18 R-M selection) that the hierarchical method
was designed to avoid. So this is a CONTRAST experiment about a tradeoff, not an upgrade.

What changes vs the free model
------------------------------
ONLY the mass population. The concentration block (c0, beta, sig_c) and the profile
likelihood (sig_extra) are byte-for-byte identical to the free-Gaussian baseline, so
any difference in the recovered beta / mass-shape is attributable solely to the mass
prior.

Chosen parameterization (most robust option for NUTS)
-----------------------------------------------------
We precompute, ONCE with colossus, the population log-pdf of log10 M implied by
   P(logM) propto [Tinker08 dn/dlnM](logM) * P(30 < lambda < 45 | logM)
on a dense logM grid, where the richness-selection term is the same log-normal-in-ln(lambda)
CDF window used by the Paper-I informed MCMC (mcmcutils.logprior), with
   sigma_lnlambda = rm_scatter * ln(10) / F      (F = McClintock18 slope = 1.356).
This grid log-pdf is wrapped as a differentiable JAX linear interpolant `mf_logpdf(logM)`.

The informed model then draws each per-cluster logM_j from a BROAD free Gaussian
envelope (free mu_M, sig_M -- same hyperpriors as the free model) and adds the
precomputed MF x selection log-pdf as an informative `numpyro.factor` on every logM_j.
Net per-cluster log-prior:
   log p(logM_j) = Normal(logM_j | mu_M, sig_M)            (free, broad envelope)
                 + mf_logpdf(logM_j)                        (physics, fixed shape)
This is robust in NUTS (no discrete choice / no importance weights), it lets the
non-Gaussian physical shape bend the per-cluster masses, and -- crucially -- the
RECOVERED population is read off as the posterior distribution of the latent logM_j
themselves (a deterministic site `logM_pop`), so it can be flat-topped even though the
envelope is Gaussian. We report both the envelope hyperparameters (mu_M, sig_M) and the
moments of the realised logM_j population (mean, std, excess kurtosis).

Run settings: 4 chains, 1000 warmup, 1500 samples; require r-hat < 1.05.
"""
import os
NCHAINS = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NCHAINS}"

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm, kurtosis as scipy_kurtosis

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary

jax.config.update("jax_enable_x64", True)

import hierarchical_mcmc_poc as H  # forward model (nfw_logSigma_vmap), config (Z, REPO, OUT_DIR)

import sys as _sys
_sys.path.insert(0, H.REPO)  # repo root for the weaklensclustersbi package import below

from colossus.cosmology import cosmology
from colossus.lss import mass_function
from weaklensclustersbi.simulations import populationutils

_NDEV = jax.local_device_count()
CHAIN_METHOD = "parallel" if _NDEV >= NCHAINS else "sequential"
print(f"JAX devices={_NDEV} -> running {NCHAINS} chains '{CHAIN_METHOD}'")

REPO = H.REPO
OUT_DIR = H.OUT_DIR
OBS_DIR = f"{REPO}/outputs/observations/obs_z1_lambda5_high_rm_scatter.376"

# ---- experiment config (from configs/observations/obs_z1_lambda5_high_rm_scatter.json) ----
Z = H.Z                       # 0.275, mid of (0.2, 0.35)
LAMBDA_MIN, LAMBDA_MAX = 30.0, 45.0
RM_RELATION = "mcclintock18"
RM_SCATTER = 0.7              # the high R-M scatter (baseline is 0.1)
MF_MODEL = "tinker08"
MF_MDEF = "200m"             # tinker08 native; matches population.draw_masses_in_richness_bin / mcmcutils
COSMO = "planck18"

NWARMUP, NSAMPLES = 1000, 1500

# ----------------------------------------------------------------------------
# Load the high-rm-scatter observation set (NOT the baseline H.obs_profiles)
# ----------------------------------------------------------------------------
obs_profiles = jnp.array(np.load(f"{OBS_DIR}/drawn_nfw_profiles.npy"))   # (376,30)
true_mc      = np.load(f"{OBS_DIR}/drawn_mc_pairs.npy")                  # (376,2) [log10M, c]
sigmas       = jnp.array(np.load(f"{OBS_DIR}/sigmas.npy"))               # (30,)
N_c = obs_profiles.shape[0]

# ---- TRUTH from the drawn population ----
_logM, _c = true_mc[:, 0], true_mc[:, 1]
TRUE = dict(
    mu_M=float(_logM.mean()),
    sig_M=float(_logM.std()),
    c0=float(_c.mean()),                       # population mean concentration (= c0 at logM=mu_M)
    sig_c=float(_c.std()),
    # beta = linear-fit slope of c vs (logM - mean), i.e. d c / d logM
    beta=float(np.polyfit(_logM, _c, 1)[0]),
    kurt_M=float(scipy_kurtosis(_logM, fisher=True)),   # excess kurtosis (Gaussian -> 0)
)
print("TRUE high_rm_scatter population:", {k: round(v, 4) for k, v in TRUE.items()})

# ----------------------------------------------------------------------------
# Precompute the mass-function x richness-selection population log-pdf on a grid.
# Mirrors mcmcutils.logprior (mass function term + log-normal richness window) and
# population.draw_masses_in_richness_bin (same Tinker08 weighting).
# ----------------------------------------------------------------------------
def build_mf_selection_logpdf():
    cosmology.setCosmology(COSMO)
    grid = np.linspace(13.3, 15.4, 1200)          # log10 M, wide enough to bracket the bin

    # 1. Tinker08 dn/dlnM  (proportional to dn/dlogM up to ln10 const -> drops in normalization)
    dndlnM = mass_function.massFunction(10 ** grid, Z, mdef=MF_MDEF, model=MF_MODEL, q_out="dndlnM")

    # 2. Richness selection P(lambda_min < lambda < lambda_max | logM), log-normal in ln(lambda).
    #    Same convention as mcmcutils.logprior: sigma_lnlambda = rm_scatter * ln10 / F.
    F = populationutils.get_rm_slope(RM_RELATION)
    sigma_lnlambda = RM_SCATTER * np.log(10.0) / F
    lam_exp = populationutils.get_richness(grid, z=Z, model=RM_RELATION)   # vectorized
    log_lam_exp = np.log(lam_exp)
    p_in_bin = (scipy_norm.cdf(np.log(LAMBDA_MAX), log_lam_exp, sigma_lnlambda)
                - scipy_norm.cdf(np.log(LAMBDA_MIN), log_lam_exp, sigma_lnlambda))

    pdf = dndlnM * p_in_bin
    pdf = np.clip(pdf, 1e-300, None)
    logpdf = np.log(pdf)
    # normalize on the grid (trapezoid) so the factor is a proper density (constant offset is
    # irrelevant to NUTS, but normalizing keeps the numbers interpretable)
    norm = np.trapz(np.exp(logpdf), grid)
    logpdf = logpdf - np.log(norm)
    return grid, logpdf, dict(sigma_lnlambda=float(sigma_lnlambda), F=float(F))


MF_GRID, MF_LOGPDF, MF_INFO = build_mf_selection_logpdf()
_MF_GRID_J = jnp.asarray(MF_GRID)
_MF_LOGPDF_J = jnp.asarray(MF_LOGPDF)

# moments of the *physical* (MF x selection) population, for reporting / the plot
_w = np.exp(MF_LOGPDF); _w = _w / _w.sum()
MF_MEAN = float(np.sum(_w * MF_GRID))
MF_STD = float(np.sqrt(np.sum(_w * (MF_GRID - MF_MEAN) ** 2)))
MF_KURT = float(np.sum(_w * ((MF_GRID - MF_MEAN) / MF_STD) ** 4) - 3.0)
print(f"MF x selection population: mean={MF_MEAN:.4f} std={MF_STD:.4f} excess_kurt={MF_KURT:.4f} "
      f"(sigma_lnlambda={MF_INFO['sigma_lnlambda']:.3f})")


def mf_logpdf_jax(logM):
    """Differentiable linear interpolation of the MF x selection log-pdf (flat extrapolation)."""
    return jnp.interp(logM, _MF_GRID_J, _MF_LOGPDF_J)


# ----------------------------------------------------------------------------
# Models. Concentration block + sig_extra + likelihood are IDENTICAL in both.
# Hyperpriors match run_hier_all_experiments.make_model (the existing baseline pkl),
# so the free model here reproduces the existing free-Gaussian fit.
# ----------------------------------------------------------------------------
def _conc_and_likelihood(logM_j, mu_M):
    c0    = numpyro.sample("c0",    dist.Normal(4.6, 1.5))
    beta  = numpyro.sample("beta",  dist.Normal(0.0, 2.0))
    sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.7))
    sig_extra = numpyro.sample("sig_extra", dist.HalfNormal(0.15))
    with numpyro.plate("clusters_c", N_c):
        zc = numpyro.sample("zc", dist.Normal(0, 1))
        c_j = c0 + beta * (logM_j - mu_M) + sig_c * zc
    model_logSig = H.nfw_logSigma_vmap(logM_j, c_j)
    sig_tot = jnp.sqrt(sigmas[None, :] ** 2 + sig_extra ** 2)
    numpyro.sample("obs", dist.Normal(model_logSig, sig_tot), obs=obs_profiles)


def free_model(obs=None):
    """Free-Gaussian mass population (the paper baseline)."""
    mu_M  = numpyro.sample("mu_M",  dist.Normal(14.4, 0.4))
    sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.4))
    with numpyro.plate("clusters_m", N_c):
        zM = numpyro.sample("zM", dist.Normal(0, 1))
        logM_j = mu_M + sig_M * zM
    numpyro.deterministic("logM_pop", logM_j)
    _conc_and_likelihood(logM_j, mu_M)


def informed_model(obs=None):
    """Mass-function x R-M-selection informed mass population.

    Per-cluster logM_j ~ broad free Gaussian envelope (free mu_M, sig_M) AND an
    informative MF x selection factor. The realised logM_j (deterministic logM_pop)
    is the recovered population and can be non-Gaussian.
    """
    mu_M  = numpyro.sample("mu_M",  dist.Normal(14.4, 0.4))
    sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.4))
    with numpyro.plate("clusters_m", N_c):
        # sample logM_j directly (centered) from the broad envelope; the MF factor
        # supplies the non-Gaussian shaping. Centered is fine here because the envelope
        # is deliberately broad and the MF factor is smooth.
        logM_j = numpyro.sample("logM_j", dist.Normal(mu_M, sig_M))
        # informative physics prior on each cluster mass
        numpyro.factor("mf_sel", mf_logpdf_jax(logM_j))
    numpyro.deterministic("logM_pop", logM_j)
    _conc_and_likelihood(logM_j, mu_M)


HYPER = ["mu_M", "sig_M", "c0", "beta", "sig_c", "sig_extra"]


def run_model(model, seed=0, label=""):
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median)
    mcmc = MCMC(kernel, num_warmup=NWARMUP, num_samples=NSAMPLES, num_chains=NCHAINS,
                chain_method=CHAIN_METHOD, progress_bar=False)
    print(f"\n[{label}] running {NCHAINS}x({NWARMUP}+{NSAMPLES}) ...", flush=True)
    mcmc.run(jax.random.PRNGKey(seed), obs=obs_profiles)
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = nps_summary({k: np.array(grouped[k]) for k in HYPER}, prob=0.9)
    rhat_max = max(float(diag[k]["r_hat"]) for k in HYPER)
    ess_min = min(float(diag[k]["n_eff"]) for k in HYPER)
    post = mcmc.get_samples()
    print(f"[{label}] rhat_max={rhat_max:.4f}  ess_min={ess_min:.0f}")
    return post, rhat_max, ess_min


def summarize(post):
    s = {k: (float(np.mean(post[k])), float(np.std(post[k]))) for k in HYPER}
    # realised population moments (from the latent per-cluster logM)
    logM_pop = np.array(post["logM_pop"])               # (n_draws, N_c)
    pop_mean = logM_pop.mean(axis=1)                    # per-draw population mean
    pop_std = logM_pop.std(axis=1)                      # per-draw population std
    pop_kurt = np.array([scipy_kurtosis(row, fisher=True) for row in logM_pop])
    s["pop_mean"] = (float(pop_mean.mean()), float(pop_mean.std()))
    s["pop_std"] = (float(pop_std.mean()), float(pop_std.std()))
    s["pop_kurt"] = (float(pop_kurt.mean()), float(pop_kurt.std()))
    return s


def main():
    free_post, free_rhat, free_ess = run_model(free_model, seed=0, label="FREE")
    inf_post, inf_rhat, inf_ess = run_model(informed_model, seed=1, label="INFORMED")

    free_s = summarize(free_post)
    inf_s = summarize(inf_post)

    # ---------------- comparison table ----------------
    def line(name, t, f, i):
        return f"{name:10s} {t:9.4f} | {f[0]:8.4f} +/- {f[1]:6.4f} | {i[0]:8.4f} +/- {i[1]:6.4f}"

    print("\n=== FREE vs INFORMED vs TRUTH (high_rm_scatter) ===")
    print(f"{'param':10s} {'TRUE':>9s} | {'FREE-Gaussian':>17s} | {'MF-INFORMED':>17s}")
    rows_keys = [("mu_M", "mu_M"), ("sig_M", "sig_M"), ("c0", "c0"),
                 ("beta", "beta"), ("sig_c", "sig_c")]
    for tk, pk in rows_keys:
        print(line(pk, TRUE[tk], free_s[pk], inf_s[pk]))
    # the non-Gaussian shape (excess kurtosis of the recovered mass population)
    print(line("popKurt", TRUE["kurt_M"], free_s["pop_kurt"], inf_s["pop_kurt"]))
    print(line("popStd", TRUE["sig_M"], free_s["pop_std"], inf_s["pop_std"]))

    # ---- did beta improve? ----
    beta_t = TRUE["beta"]
    free_beta_err = abs(free_s["beta"][0] - beta_t)
    inf_beta_err = abs(inf_s["beta"][0] - beta_t)
    free_beta_width = free_s["beta"][1]
    inf_beta_width = inf_s["beta"][1]
    print(f"\nbeta truth={beta_t:.4f}")
    print(f"  FREE     beta={free_s['beta'][0]:.4f}+/-{free_beta_width:.4f}  |bias|={free_beta_err:.4f}")
    print(f"  INFORMED beta={inf_s['beta'][0]:.4f}+/-{inf_beta_width:.4f}  |bias|={inf_beta_err:.4f}")
    beta_bias_better = inf_beta_err < free_beta_err
    beta_tighter = inf_beta_width < free_beta_width
    print(f"  -> INFORMED beta bias {'BETTER' if beta_bias_better else 'NOT better'}; "
          f"posterior {'TIGHTER' if beta_tighter else 'NOT tighter'} "
          f"({free_beta_width:.4f} -> {inf_beta_width:.4f}, "
          f"{100*(free_beta_width-inf_beta_width)/free_beta_width:+.1f}%)")

    # ---------------- figure ----------------
    make_figure(free_post, inf_post, free_s, inf_s)

    # ---------------- save ----------------
    out = dict(
        experiment="obs_z1_lambda5_high_rm_scatter.376",
        true=TRUE,
        free=dict(summary=free_s, rhat_max=free_rhat, ess_min=free_ess),
        informed=dict(summary=inf_s, rhat_max=inf_rhat, ess_min=inf_ess),
        mf_population=dict(mean=MF_MEAN, std=MF_STD, excess_kurt=MF_KURT, **MF_INFO),
        beta_verdict=dict(truth=beta_t,
                          free=dict(mean=free_s["beta"][0], sd=free_beta_width, abs_bias=free_beta_err),
                          informed=dict(mean=inf_s["beta"][0], sd=inf_beta_width, abs_bias=inf_beta_err),
                          bias_improved=bool(beta_bias_better), width_tighter=bool(beta_tighter)),
        run=dict(nchains=NCHAINS, nwarmup=NWARMUP, nsamples=NSAMPLES),
    )
    with open(f"{OUT_DIR}/massfn_informed.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved -> {OUT_DIR}/massfn_informed.pkl")
    return out


def make_figure(free_post, inf_post, free_s, inf_s):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- panel A: recovered mass-population marginal vs truth ---
    ax = axes[0]
    # truth: the actual drawn logM
    ax.hist(_logM, bins=30, density=True, histtype="stepfilled", color="k", alpha=0.18,
            label=f"TRUE drawn population (k={TRUE['kurt_M']:.2f})")
    # free model: pooled posterior logM_pop
    free_pop = np.array(free_post["logM_pop"]).ravel()
    inf_pop = np.array(inf_post["logM_pop"]).ravel()
    ax.hist(free_pop, bins=80, density=True, histtype="step", color="#1f77b4", lw=2,
            label=f"FREE-Gaussian (k={free_s['pop_kurt'][0]:.2f})")
    ax.hist(inf_pop, bins=80, density=True, histtype="step", color="#2ca02c", lw=2,
            label=f"MF-informed (k={inf_s['pop_kurt'][0]:.2f})")
    # the fixed physical MF x selection shape
    ax.plot(MF_GRID, np.exp(MF_LOGPDF), "r--", lw=1.6, alpha=0.8,
            label=f"MF x R-M selection (k={MF_KURT:.2f})")
    ax.set_xlim(13.6, 15.0)
    ax.set_xlabel(r"$\log_{10} M$"); ax.set_ylabel("population density")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Recovered in-bin mass distribution (high R-M scatter)")

    # --- panel B: beta posteriors vs truth ---
    ax = axes[1]
    ax.hist(np.array(free_post["beta"]), bins=60, density=True, histtype="step",
            color="#1f77b4", lw=2,
            label=f"FREE: {free_s['beta'][0]:.2f}$\\pm${free_s['beta'][1]:.2f}")
    ax.hist(np.array(inf_post["beta"]), bins=60, density=True, histtype="step",
            color="#2ca02c", lw=2,
            label=f"INFORMED: {inf_s['beta'][0]:.2f}$\\pm${inf_s['beta'][1]:.2f}")
    ax.axvline(TRUE["beta"], color="k", ls="--", lw=2, label=f"TRUE $\\beta$={TRUE['beta']:.2f}")
    ax.set_xlabel(r"$\beta$  (M-c slope, $dc/d\log_{10}M$)")
    ax.set_ylabel("posterior density"); ax.legend(fontsize=9)
    ax.set_title("M-c slope posterior")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/massfn_informed_compare.png", dpi=140)
    print(f"Saved figure -> {OUT_DIR}/massfn_informed_compare.png")


if __name__ == "__main__":
    main()
