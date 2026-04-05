"""
Toy comparison: Surface Density (Σ) vs Excess Surface Density (ΔΣ) as data vectors.

This script investigates whether using Σ vs ΔΣ affects inference quality.
ΔΣ(R) = Σ̄(<R) - Σ(R) is the standard weak lensing observable.

Author: Claude (analysis script for paper review)
"""

import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
import emcee
from scipy import stats

# Set cosmology
cosmology.setCosmology('planck18')

# Configuration
N_OBS = 10
TRUE_LOG10MASS = 14.3
TRUE_CONCENTRATION = 4.8
REDSHIFT = 0.3
NOISE_DEX = 0.3
RBINS = 10 ** np.linspace(np.log10(100), np.log10(3000), 15)  # kpc/h

np.random.seed(42)


def generate_profile(log10mass, concentration, kind='surfaceDensity'):
    """Generate NFW profile using either Σ or ΔΣ."""
    nfw = profile_nfw.NFWProfile(
        M=10**log10mass, c=concentration, z=REDSHIFT, mdef='vir'
    )
    if kind == 'surfaceDensity':
        return nfw.surfaceDensity(RBINS)
    elif kind == 'deltaSigma':
        return nfw.deltaSigma(RBINS)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def add_noise(profile, dex=NOISE_DEX):
    """Add log-normal noise to profile."""
    noise = np.random.normal(0, dex, len(profile))
    return profile * 10**noise


def generate_observations(n_obs, kind='surfaceDensity'):
    """Generate n_obs noisy observations."""
    true_profile = generate_profile(TRUE_LOG10MASS, TRUE_CONCENTRATION, kind)
    observations = np.array([add_noise(true_profile) for _ in range(n_obs)])
    return true_profile, observations


def log_likelihood(params, observed_profile, sigma_err, kind):
    """Gaussian log-likelihood in log-space."""
    log10mass, concentration = params
    if concentration <= 0 or log10mass < 12 or log10mass > 16:
        return -np.inf

    try:
        model = generate_profile(log10mass, concentration, kind)
        # Compare in log-space
        log_obs = np.log10(observed_profile)
        log_model = np.log10(model)
        chi2 = np.sum((log_obs - log_model)**2 / sigma_err**2)
        return -0.5 * chi2
    except:
        return -np.inf


def log_prior(params):
    """Flat prior on mass and concentration."""
    log10mass, concentration = params
    if 12 < log10mass < 16 and 1 < concentration < 10:
        return 0.0
    return -np.inf


def log_probability(params, observed_profile, sigma_err, kind):
    """Combined log-probability."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, observed_profile, sigma_err, kind)


def run_mcmc(observed_profile, sigma_err, kind, n_walkers=32, n_steps=500):
    """Run MCMC inference."""
    ndim = 2
    # Start near truth with some scatter
    p0 = np.array([TRUE_LOG10MASS, TRUE_CONCENTRATION]) + 0.1 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability,
        args=(observed_profile, sigma_err, kind)
    )

    # Burn-in
    state = sampler.run_mcmc(p0, 100, progress=False)
    sampler.reset()

    # Production
    sampler.run_mcmc(state, n_steps, progress=False)

    return sampler.flatchain


def main():
    print("=" * 60)
    print("Σ vs ΔΣ Data Vector Comparison")
    print("=" * 60)
    print(f"\nTrue parameters: log10M = {TRUE_LOG10MASS}, c = {TRUE_CONCENTRATION}")
    print(f"Number of observations: {N_OBS}")
    print(f"Noise level: {NOISE_DEX} dex")

    # Generate observations for both data vector types
    results = {}

    for kind in ['surfaceDensity', 'deltaSigma']:
        print(f"\n{'='*40}")
        print(f"Testing: {kind}")
        print('='*40)

        true_profile, observations = generate_observations(N_OBS, kind)

        # Stack observations (median)
        stacked = np.median(observations, axis=0)
        sigma_err = NOISE_DEX / np.sqrt(N_OBS)  # Simplified error estimate

        print(f"\nProfile shape comparison:")
        print(f"  True profile range: [{true_profile.min():.2e}, {true_profile.max():.2e}]")
        print(f"  Stacked profile range: [{stacked.min():.2e}, {stacked.max():.2e}]")
        print(f"  Dynamic range: {true_profile.max()/true_profile.min():.1f}x")

        # Run MCMC
        print(f"\nRunning MCMC inference...")
        chain = run_mcmc(stacked, sigma_err, kind)

        # Results
        mass_samples = chain[:, 0]
        conc_samples = chain[:, 1]

        mass_median = np.median(mass_samples)
        mass_std = np.std(mass_samples)
        conc_median = np.median(conc_samples)
        conc_std = np.std(conc_samples)

        mass_bias = mass_median - TRUE_LOG10MASS
        conc_bias = conc_median - TRUE_CONCENTRATION

        print(f"\nResults:")
        print(f"  log10M: {mass_median:.3f} ± {mass_std:.3f} (truth: {TRUE_LOG10MASS}, bias: {mass_bias:+.3f})")
        print(f"  c:      {conc_median:.2f} ± {conc_std:.2f} (truth: {TRUE_CONCENTRATION}, bias: {conc_bias:+.2f})")

        # Check if truth is within 1σ
        mass_in_1sigma = abs(mass_bias) < mass_std
        conc_in_1sigma = abs(conc_bias) < conc_std
        print(f"  Truth within 1σ: mass={mass_in_1sigma}, conc={conc_in_1sigma}")

        results[kind] = {
            'chain': chain,
            'true_profile': true_profile,
            'stacked': stacked,
            'observations': observations,
            'mass_median': mass_median,
            'mass_std': mass_std,
            'conc_median': conc_median,
            'conc_std': conc_std,
            'mass_bias': mass_bias,
            'conc_bias': conc_bias,
        }

    # Compare profile shapes
    print("\n" + "=" * 60)
    print("KEY DIFFERENCE: Profile Shape")
    print("=" * 60)

    sigma_profile = generate_profile(TRUE_LOG10MASS, TRUE_CONCENTRATION, 'surfaceDensity')
    delta_profile = generate_profile(TRUE_LOG10MASS, TRUE_CONCENTRATION, 'deltaSigma')

    print(f"\nAt R = {RBINS[0]:.0f} kpc/h (inner):")
    print(f"  Σ = {sigma_profile[0]:.2e} M_sun/kpc^2")
    print(f"  ΔΣ = {delta_profile[0]:.2e} M_sun/kpc^2")
    print(f"  Ratio Σ/ΔΣ = {sigma_profile[0]/delta_profile[0]:.2f}")

    print(f"\nAt R = {RBINS[-1]:.0f} kpc/h (outer):")
    print(f"  Σ = {sigma_profile[-1]:.2e} M_sun/kpc^2")
    print(f"  ΔΣ = {delta_profile[-1]:.2e} M_sun/kpc^2")
    print(f"  Ratio Σ/ΔΣ = {sigma_profile[-1]/delta_profile[-1]:.2f}")

    print(f"\nSlope comparison (d log profile / d log R):")
    sigma_slope = np.diff(np.log10(sigma_profile)) / np.diff(np.log10(RBINS))
    delta_slope = np.diff(np.log10(delta_profile)) / np.diff(np.log10(RBINS))
    print(f"  Σ slope (median): {np.median(sigma_slope):.2f}")
    print(f"  ΔΣ slope (median): {np.median(delta_slope):.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Σ (surfaceDensity)':<20} {'ΔΣ (deltaSigma)':<20}")
    print("-" * 70)
    print(f"{'Mass bias':<30} {results['surfaceDensity']['mass_bias']:+.4f}{'':<14} {results['deltaSigma']['mass_bias']:+.4f}")
    print(f"{'Mass uncertainty':<30} {results['surfaceDensity']['mass_std']:.4f}{'':<15} {results['deltaSigma']['mass_std']:.4f}")
    print(f"{'Conc bias':<30} {results['surfaceDensity']['conc_bias']:+.3f}{'':<15} {results['deltaSigma']['conc_bias']:+.3f}")
    print(f"{'Conc uncertainty':<30} {results['surfaceDensity']['conc_std']:.3f}{'':<16} {results['deltaSigma']['conc_std']:.3f}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Profile comparison
    ax1 = axes[0, 0]
    ax1.loglog(RBINS, sigma_profile, 'b-', lw=2, label='Σ (surface density)')
    ax1.loglog(RBINS, delta_profile, 'r-', lw=2, label='ΔΣ (excess surface density)')
    ax1.set_xlabel('R [kpc/h]')
    ax1.set_ylabel('Density [M_sun/kpc^2]')
    ax1.legend()
    ax1.set_title('True Profiles: Σ vs ΔΣ')

    ax2 = axes[0, 1]
    for i, obs in enumerate(results['surfaceDensity']['observations']):
        ax2.loglog(RBINS, obs, 'b-', alpha=0.3, lw=0.5)
    ax2.loglog(RBINS, results['surfaceDensity']['stacked'], 'b-', lw=2, label='Stacked')
    ax2.loglog(RBINS, results['surfaceDensity']['true_profile'], 'k--', lw=2, label='Truth')
    ax2.set_xlabel('R [kpc/h]')
    ax2.set_ylabel('Σ [M_sun/kpc^2]')
    ax2.legend()
    ax2.set_title(f'Σ: {N_OBS} Observations')

    ax3 = axes[0, 2]
    for i, obs in enumerate(results['deltaSigma']['observations']):
        ax3.loglog(RBINS, obs, 'r-', alpha=0.3, lw=0.5)
    ax3.loglog(RBINS, results['deltaSigma']['stacked'], 'r-', lw=2, label='Stacked')
    ax3.loglog(RBINS, results['deltaSigma']['true_profile'], 'k--', lw=2, label='Truth')
    ax3.set_xlabel('R [kpc/h]')
    ax3.set_ylabel('ΔΣ [M_sun/kpc^2]')
    ax3.legend()
    ax3.set_title(f'ΔΣ: {N_OBS} Observations')

    # Row 2: Posterior comparison
    ax4 = axes[1, 0]
    ax4.hist(results['surfaceDensity']['chain'][:, 0], bins=50, alpha=0.5, label='Σ', color='blue', density=True)
    ax4.hist(results['deltaSigma']['chain'][:, 0], bins=50, alpha=0.5, label='ΔΣ', color='red', density=True)
    ax4.axvline(TRUE_LOG10MASS, color='k', ls='--', lw=2, label='Truth')
    ax4.set_xlabel('log10(M)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.set_title('Mass Posterior')

    ax5 = axes[1, 1]
    ax5.hist(results['surfaceDensity']['chain'][:, 1], bins=50, alpha=0.5, label='Σ', color='blue', density=True)
    ax5.hist(results['deltaSigma']['chain'][:, 1], bins=50, alpha=0.5, label='ΔΣ', color='red', density=True)
    ax5.axvline(TRUE_CONCENTRATION, color='k', ls='--', lw=2, label='Truth')
    ax5.set_xlabel('Concentration')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.set_title('Concentration Posterior')

    ax6 = axes[1, 2]
    ax6.scatter(results['surfaceDensity']['chain'][::10, 0], results['surfaceDensity']['chain'][::10, 1],
                alpha=0.3, s=1, c='blue', label='Σ')
    ax6.scatter(results['deltaSigma']['chain'][::10, 0], results['deltaSigma']['chain'][::10, 1],
                alpha=0.3, s=1, c='red', label='ΔΣ')
    ax6.scatter([TRUE_LOG10MASS], [TRUE_CONCENTRATION], c='k', s=100, marker='*', label='Truth', zorder=10)
    ax6.set_xlabel('log10(M)')
    ax6.set_ylabel('Concentration')
    ax6.legend()
    ax6.set_title('Joint Posterior')

    plt.tight_layout()
    plt.savefig('/Users/akumgill/Documents/GitHub/CL-SBI/notebooks/sigma_vs_deltasigma_comparison.png', dpi=150)
    print(f"\nPlot saved to: notebooks/sigma_vs_deltasigma_comparison.png")

    # Final verdict
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    mass_diff = abs(results['surfaceDensity']['mass_bias']) - abs(results['deltaSigma']['mass_bias'])
    conc_diff = abs(results['surfaceDensity']['conc_bias']) - abs(results['deltaSigma']['conc_bias'])

    if abs(mass_diff) < 0.01 and abs(conc_diff) < 0.1:
        print("\nBoth Σ and ΔΣ give comparable results for this toy example.")
        print("The choice may matter more for:")
        print("  - Non-NFW profiles (ΔΣ is more directly observable)")
        print("  - Real data with shear calibration (ΔΣ = Σ_crit * γ_t)")
        print("  - Profiles extending to R > R_vir")
    elif mass_diff < 0:
        print(f"\nΣ shows LESS mass bias ({results['surfaceDensity']['mass_bias']:+.4f} vs {results['deltaSigma']['mass_bias']:+.4f})")
    else:
        print(f"\nΔΣ shows LESS mass bias ({results['deltaSigma']['mass_bias']:+.4f} vs {results['surfaceDensity']['mass_bias']:+.4f})")


if __name__ == '__main__':
    main()
