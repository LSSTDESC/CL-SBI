#!/usr/bin/env python3
"""
Quick HMC benchmark on baseline observations.

Runs NumPyro NUTS on a subset of obs_z1_lambda5 to estimate
total time for 376 clusters, for comparison with emcee pipeline.
"""

import sys
import time
import json
import numpy as np

sys.path.insert(0, "..")

# JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# NumPyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# halox
from halox import cosmology as halox_cosmo
from halox import halo

# Local
from weaklensclustersbi.simulations import wlprofile, populationutils

# Settings
N_SUBSET = 20  # Run on this many clusters, extrapolate to 376
NUM_WARMUP = 100  # Match pipeline burn-in
NUM_SAMPLES = 500  # Match pipeline production
NUM_CHAINS = 1  # Single chain per observation (like emcee FTJ)

# halox setup
COSMO = halox_cosmo.Planck18()
Z_FIXED = 0.275

# Radial bins (must match observations)
NUM_RADIAL_BINS = 30
RBINS_KPC = 10 ** np.linspace(0, 3, NUM_RADIAL_BINS)
RBINS_MPC = RBINS_KPC * 1e-3


@jax.jit
def nfw_log_surface_density_halox(log10mass, conc, r_mpc):
    """Compute log10(Σ) using halox NFW profile."""
    M = 10**log10mass
    nfw = halo.nfw.NFWHalo(M, conc, Z_FIXED, COSMO, delta=200)
    sigma = nfw.surface_density(r_mpc)
    return jnp.log10(sigma)


def numpyro_model(obs_profile, obs_sigma, r_mpc):
    """NumPyro model for NFW profile inference."""
    log10mass = numpyro.sample("log10mass", dist.Uniform(13.0, 16.0))

    # M-c relation prior (linear approximation to Child+18)
    c_expected = 4.5 - 0.3 * (log10mass - 14.5)
    conc = numpyro.sample(
        "concentration",
        dist.TruncatedNormal(c_expected, 0.15, low=1.0, high=10.0)
    )

    log_f = numpyro.sample("log_f", dist.Normal(-2.0, 1.0))

    model_log_sigma = nfw_log_surface_density_halox(log10mass, conc, r_mpc)

    numpyro.sample(
        "obs",
        dist.Normal(model_log_sigma, obs_sigma),
        obs=obs_profile
    )


def run_hmc_single(obs_profile, obs_sigma):
    """Run HMC on a single observation."""
    kernel = NUTS(
        numpyro_model,
        init_strategy=numpyro.infer.init_to_value(
            values={"log10mass": 14.5, "concentration": 4.5, "log_f": -2.0}
        ),
        forward_mode_differentiation=True
    )
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=NUM_CHAINS)

    rng_key = jax.random.PRNGKey(42)
    mcmc.run(
        rng_key,
        obs_profile=jnp.array(obs_profile),
        obs_sigma=jnp.array(obs_sigma),
        r_mpc=jnp.array(RBINS_MPC),
    )
    return mcmc


def main():
    print("=" * 70)
    print("HMC BENCHMARK: NumPyro NUTS on baseline observations")
    print("=" * 70)

    # Load observations
    obs_dir = "../outputs/observations/obs_z1_lambda5.376"
    print(f"\nLoading observations from {obs_dir}")

    profiles = np.load(f"{obs_dir}/drawn_nfw_profiles.npy")
    sigmas = np.load(f"{obs_dir}/sigmas.npy")
    true_params = np.load(f"{obs_dir}/drawn_mc_pairs.npy")

    n_total = profiles.shape[0]
    print(f"Profiles shape: {profiles.shape}")
    print(f"Sigmas shape: {sigmas.shape}")
    print(f"Total observations: {n_total}")
    print(f"Running HMC on subset: {N_SUBSET}")

    # Compute halox unit offset (same as notebook)
    true_log10mass = 14.5
    true_conc = populationutils.get_concentration(true_log10mass, model="child18", z=Z_FIXED)
    true_profile = wlprofile.simulate_nfw(true_log10mass, true_conc, rbins=RBINS_KPC, z=Z_FIXED)
    log_true_profile = np.log10(true_profile)

    halox_log_sigma_true = nfw_log_surface_density_halox(
        true_log10mass, true_conc, jnp.array(RBINS_MPC)
    )
    offset = np.mean(np.array(halox_log_sigma_true) - log_true_profile)
    print(f"halox unit offset: {offset:.3f}")

    # Settings summary
    print(f"\nHMC settings (matching pipeline):")
    print(f"  Warmup: {NUM_WARMUP}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Chains: {NUM_CHAINS}")

    # Warmup JIT (first run is slow)
    print("\nWarming up JIT compilation...")
    # Profiles are already in log10 space
    obs_profile = profiles[0] + offset
    obs_sigma = sigmas[0] if sigmas.ndim > 1 else np.full(NUM_RADIAL_BINS, sigmas[0])
    _ = run_hmc_single(obs_profile, obs_sigma)
    print("JIT warmup complete.")

    # Run on subset
    print(f"\nRunning HMC on {N_SUBSET} observations...")
    times = []

    for i in range(N_SUBSET):
        obs_profile = profiles[i] + offset
        obs_sigma = sigmas[i] if sigmas.ndim > 1 else np.full(NUM_RADIAL_BINS, sigmas[0])

        t0 = time.time()
        mcmc = run_hmc_single(obs_profile, obs_sigma)
        elapsed = time.time() - t0
        times.append(elapsed)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_SUBSET} done, last: {elapsed:.2f}s, mean: {np.mean(times):.2f}s")

    # Results
    mean_time = np.mean(times)
    std_time = np.std(times)
    total_376 = mean_time * 376

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nPer-observation HMC time: {mean_time:.2f} ± {std_time:.2f} s")
    print(f"Extrapolated time for 376 obs: {total_376:.1f}s ({total_376/60:.1f} min)")

    # Compare with pipeline
    print("\n--- Comparison with pipeline (from pipeline_speed.csv) ---")
    mcmc_ftj_time = 350  # Approximate from recent runs
    sbi_time = 1.4

    print(f"MCMC FTJ (emcee): ~{mcmc_ftj_time}s for 376 obs")
    print(f"HMC (NumPyro):    ~{total_376:.0f}s for 376 obs (extrapolated)")
    print(f"SBI:              ~{sbi_time}s for 376 obs")

    print(f"\nSpeedups:")
    print(f"  emcee → HMC: {mcmc_ftj_time/total_376:.1f}x")
    print(f"  HMC → SBI:   {total_376/sbi_time:.0f}x")
    print(f"  emcee → SBI: {mcmc_ftj_time/sbi_time:.0f}x")


if __name__ == "__main__":
    main()
