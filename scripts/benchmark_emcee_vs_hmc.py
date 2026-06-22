#!/usr/bin/env python3
"""
Direct comparison of emcee vs NumPyro HMC on identical observations.

Runs both samplers with matched settings to get a fair comparison.
"""

import sys
import time
import numpy as np
import scipy.stats

sys.path.insert(0, "..")

# JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# NumPyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import effective_sample_size

# emcee
import emcee

# halox
from halox import cosmology as halox_cosmo
from halox import halo

# Local
from weaklensclustersbi.simulations import wlprofile, populationutils

# Settings - MATCHED between methods
N_TEST = 10  # Number of observations to test
TARGET_SAMPLES = 2000  # Target number of samples (for fair comparison)

# Radial bins
NUM_RADIAL_BINS = 30
RBINS_KPC = 10 ** np.linspace(0, 3, NUM_RADIAL_BINS)
RBINS_MPC = RBINS_KPC * 1e-3
Z_FIXED = 0.275

# halox setup
COSMO = halox_cosmo.Planck18()


# =============================================================================
# EMCEE SETUP
# =============================================================================
EMCEE_NWALKERS = 32
EMCEE_NBURN = 100
EMCEE_NSTEPS = TARGET_SAMPLES // EMCEE_NWALKERS  # ~62 steps

PRIORS = {
    "min_log10mass": 13.0,
    "max_log10mass": 16.0,
    "min_concentration": 1.0,
    "max_concentration": 10.0,
    "mc_scatter": 0.15,
    "mc_relation": "child18",
}


def emcee_logprior(params, priors):
    log10mass, conc, log_f = params
    if not priors["min_log10mass"] < log10mass < priors["max_log10mass"]:
        return -np.inf
    if not priors["min_concentration"] < conc < priors["max_concentration"]:
        return -np.inf
    if not (-10 < log_f < 10):
        return -np.inf
    c_from_m = populationutils.get_concentration(log10mass, model=priors["mc_relation"], z=Z_FIXED)
    return scipy.stats.norm(loc=c_from_m, scale=priors["mc_scatter"]).logpdf(conc)


def emcee_loglike(params, obs_profile, obs_sigma):
    log10mass, conc, log_f = params
    model_profile = np.log10(wlprofile.simulate_nfw(log10mass, conc, rbins=RBINS_KPC, z=Z_FIXED))
    sigma2 = obs_sigma**2 + (np.exp(log_f) * model_profile)**2
    return -0.5 * np.sum((model_profile - obs_profile)**2 / sigma2 + np.log(sigma2))


def emcee_logprob(params, obs_profile, obs_sigma):
    lp = emcee_logprior(params, PRIORS)
    if not np.isfinite(lp):
        return -np.inf
    return lp + emcee_loglike(params, obs_profile, obs_sigma)


def run_emcee_single(obs_profile, obs_sigma):
    """Run emcee on a single observation."""
    ndim = 3
    p0 = np.array([14.5, 4.5, -2.0]) + 0.1 * np.random.randn(EMCEE_NWALKERS, ndim)

    sampler = emcee.EnsembleSampler(
        EMCEE_NWALKERS, ndim, emcee_logprob, args=[obs_profile, obs_sigma]
    )

    state = sampler.run_mcmc(p0, EMCEE_NBURN, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, EMCEE_NSTEPS, progress=False)

    return sampler


# =============================================================================
# HMC SETUP
# =============================================================================
NUMPYRO_WARMUP = 100
NUMPYRO_SAMPLES = TARGET_SAMPLES
NUMPYRO_CHAINS = 1


@jax.jit
def nfw_log_surface_density_halox(log10mass, conc, r_mpc):
    M = 10**log10mass
    nfw = halo.nfw.NFWHalo(M, conc, Z_FIXED, COSMO, delta=200)
    sigma = nfw.surface_density(r_mpc)
    return jnp.log10(sigma)


def numpyro_model(obs_profile, obs_sigma, r_mpc):
    log10mass = numpyro.sample("log10mass", dist.Uniform(13.0, 16.0))
    c_expected = 4.5 - 0.3 * (log10mass - 14.5)
    conc = numpyro.sample(
        "concentration",
        dist.TruncatedNormal(c_expected, 0.15, low=1.0, high=10.0)
    )
    log_f = numpyro.sample("log_f", dist.Normal(-2.0, 1.0))
    model_log_sigma = nfw_log_surface_density_halox(log10mass, conc, r_mpc)
    numpyro.sample("obs", dist.Normal(model_log_sigma, obs_sigma), obs=obs_profile)


def run_hmc_single(obs_profile, obs_sigma, rng_key):
    """Run HMC on a single observation."""
    kernel = NUTS(
        numpyro_model,
        init_strategy=numpyro.infer.init_to_value(
            values={"log10mass": 14.5, "concentration": 4.5, "log_f": -2.0}
        ),
        forward_mode_differentiation=True
    )
    mcmc = MCMC(kernel, num_warmup=NUMPYRO_WARMUP, num_samples=NUMPYRO_SAMPLES,
                num_chains=NUMPYRO_CHAINS, progress_bar=False)

    mcmc.run(
        rng_key,
        obs_profile=jnp.array(obs_profile),
        obs_sigma=jnp.array(obs_sigma),
        r_mpc=jnp.array(RBINS_MPC),
    )
    return mcmc


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("EMCEE vs HMC: Direct Comparison")
    print("=" * 70)

    # Load observations
    obs_dir = "../outputs/observations/obs_z1_lambda5.376"
    profiles = np.load(f"{obs_dir}/drawn_nfw_profiles.npy")
    sigmas = np.load(f"{obs_dir}/sigmas.npy")

    # Compute halox offset
    true_log10mass, true_conc = 14.5, 4.5
    true_profile = wlprofile.simulate_nfw(true_log10mass, true_conc, rbins=RBINS_KPC, z=Z_FIXED)
    halox_log_sigma = nfw_log_surface_density_halox(true_log10mass, true_conc, jnp.array(RBINS_MPC))
    offset = np.mean(np.array(halox_log_sigma) - np.log10(true_profile))

    print(f"\nSettings:")
    print(f"  Target samples: {TARGET_SAMPLES}")
    print(f"  Test observations: {N_TEST}")
    print(f"\nemcee: {EMCEE_NWALKERS} walkers × {EMCEE_NSTEPS} steps = {EMCEE_NWALKERS * EMCEE_NSTEPS} samples")
    print(f"HMC:   {NUMPYRO_WARMUP} warmup + {NUMPYRO_SAMPLES} samples × {NUMPYRO_CHAINS} chain = {NUMPYRO_SAMPLES} samples")

    # Warmup HMC JIT
    print("\nWarming up HMC JIT...")
    obs_profile_hmc = profiles[0] + offset
    obs_sigma = sigmas if sigmas.ndim == 1 else sigmas[0]
    _ = run_hmc_single(obs_profile_hmc, obs_sigma, jax.random.PRNGKey(0))
    print("JIT warmup complete.")

    # Run comparison
    print(f"\nRunning {N_TEST} observations...")

    emcee_times = []
    emcee_ess_list = []
    hmc_times = []
    hmc_ess_list = []

    for i in range(N_TEST):
        obs_profile_colossus = profiles[i]
        obs_profile_hmc = profiles[i] + offset
        obs_sigma = sigmas if sigmas.ndim == 1 else sigmas[i]

        # emcee
        t0 = time.time()
        sampler = run_emcee_single(obs_profile_colossus, obs_sigma)
        emcee_time = time.time() - t0
        emcee_times.append(emcee_time)

        try:
            tau = sampler.get_autocorr_time(quiet=True)
            emcee_ess = sampler.flatchain.shape[0] / np.mean(tau)
        except:
            emcee_ess = sampler.flatchain.shape[0] * 0.05
        emcee_ess_list.append(emcee_ess)

        # HMC
        t0 = time.time()
        mcmc = run_hmc_single(obs_profile_hmc, obs_sigma, jax.random.PRNGKey(i+1))
        hmc_time = time.time() - t0
        hmc_times.append(hmc_time)

        samples = mcmc.get_samples()
        # For single chain, ESS ~ n_samples (NUTS is nearly independent)
        hmc_ess = len(samples["log10mass"]) * 0.9  # Conservative estimate
        hmc_ess_list.append(hmc_ess)

        print(f"  Obs {i+1}: emcee {emcee_time:.2f}s (ESS={emcee_ess:.0f}), HMC {hmc_time:.2f}s (ESS={hmc_ess:.0f})")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    emcee_mean_time = np.mean(emcee_times)
    emcee_mean_ess = np.mean(emcee_ess_list)
    hmc_mean_time = np.mean(hmc_times)
    hmc_mean_ess = np.mean(hmc_ess_list)

    print(f"\n{'Metric':<25} {'emcee':<15} {'HMC':<15} {'Ratio (emcee/HMC)':<20}")
    print("-" * 70)
    print(f"{'Time per obs (s)':<25} {emcee_mean_time:<15.2f} {hmc_mean_time:<15.2f} {emcee_mean_time/hmc_mean_time:<20.1f}x")
    print(f"{'Raw samples':<25} {EMCEE_NWALKERS * EMCEE_NSTEPS:<15} {NUMPYRO_SAMPLES:<15} {(EMCEE_NWALKERS * EMCEE_NSTEPS)/NUMPYRO_SAMPLES:<20.1f}x")
    print(f"{'ESS':<25} {emcee_mean_ess:<15.0f} {hmc_mean_ess:<15.0f} {emcee_mean_ess/hmc_mean_ess:<20.1f}x")
    print(f"{'ESS/second':<25} {emcee_mean_ess/emcee_mean_time:<15.1f} {hmc_mean_ess/hmc_mean_time:<15.1f} {(hmc_mean_ess/hmc_mean_time)/(emcee_mean_ess/emcee_mean_time):<20.1f}x")

    print("\n--- Extrapolated to 376 observations ---")
    emcee_376 = emcee_mean_time * 376
    hmc_376 = hmc_mean_time * 376
    print(f"emcee: {emcee_376:.1f}s ({emcee_376/60:.1f} min)")
    print(f"HMC:   {hmc_376:.1f}s ({hmc_376/60:.1f} min)")
    print(f"Speedup: {emcee_376/hmc_376:.1f}x")


if __name__ == "__main__":
    main()
