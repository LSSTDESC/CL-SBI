"""
Multiple realizations to test robustness of Σ vs ΔΣ comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
import emcee

cosmology.setCosmology('planck18')

N_OBS = 10
TRUE_LOG10MASS = 14.3
TRUE_CONCENTRATION = 4.8
REDSHIFT = 0.3
NOISE_DEX = 0.3
RBINS = 10 ** np.linspace(np.log10(100), np.log10(3000), 15)
N_REALIZATIONS = 20


def generate_profile(log10mass, concentration, kind='surfaceDensity'):
    nfw = profile_nfw.NFWProfile(M=10**log10mass, c=concentration, z=REDSHIFT, mdef='vir')
    if kind == 'surfaceDensity':
        return nfw.surfaceDensity(RBINS)
    elif kind == 'deltaSigma':
        return nfw.deltaSigma(RBINS)


def add_noise(profile, dex=NOISE_DEX):
    noise = np.random.normal(0, dex, len(profile))
    return profile * 10**noise


def log_probability(params, observed_profile, sigma_err, kind):
    log10mass, concentration = params
    if not (12 < log10mass < 16 and 1 < concentration < 10):
        return -np.inf
    try:
        model = generate_profile(log10mass, concentration, kind)
        log_obs = np.log10(observed_profile)
        log_model = np.log10(model)
        chi2 = np.sum((log_obs - log_model)**2 / sigma_err**2)
        return -0.5 * chi2
    except:
        return -np.inf


def run_single_realization(seed, kind):
    np.random.seed(seed)

    true_profile = generate_profile(TRUE_LOG10MASS, TRUE_CONCENTRATION, kind)
    observations = np.array([add_noise(true_profile) for _ in range(N_OBS)])
    stacked = np.median(observations, axis=0)
    sigma_err = NOISE_DEX / np.sqrt(N_OBS)

    # Quick MCMC
    ndim, n_walkers, n_steps = 2, 24, 300
    p0 = np.array([TRUE_LOG10MASS, TRUE_CONCENTRATION]) + 0.1 * np.random.randn(n_walkers, ndim)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(stacked, sigma_err, kind))
    state = sampler.run_mcmc(p0, 50, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=False)

    chain = sampler.flatchain
    mass_est = np.median(chain[:, 0])
    conc_est = np.median(chain[:, 1])

    return mass_est - TRUE_LOG10MASS, conc_est - TRUE_CONCENTRATION


def main():
    print(f"Running {N_REALIZATIONS} realizations for each data vector type...\n")

    results = {'surfaceDensity': {'mass_bias': [], 'conc_bias': []},
               'deltaSigma': {'mass_bias': [], 'conc_bias': []}}

    for kind in ['surfaceDensity', 'deltaSigma']:
        print(f"Processing {kind}...")
        for i in range(N_REALIZATIONS):
            mass_bias, conc_bias = run_single_realization(1000 + i, kind)
            results[kind]['mass_bias'].append(mass_bias)
            results[kind]['conc_bias'].append(conc_bias)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{N_REALIZATIONS} done")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY OVER", N_REALIZATIONS, "REALIZATIONS")
    print("=" * 70)

    for kind in ['surfaceDensity', 'deltaSigma']:
        mb = np.array(results[kind]['mass_bias'])
        cb = np.array(results[kind]['conc_bias'])
        print(f"\n{kind}:")
        print(f"  Mass bias:  mean = {np.mean(mb):+.4f}, std = {np.std(mb):.4f}, |mean|/std = {abs(np.mean(mb))/np.std(mb):.2f}")
        print(f"  Conc bias:  mean = {np.mean(cb):+.3f}, std = {np.std(cb):.3f}, |mean|/std = {abs(np.mean(cb))/np.std(cb):.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.hist(results['surfaceDensity']['mass_bias'], bins=10, alpha=0.6, label='Σ', color='blue')
    ax1.hist(results['deltaSigma']['mass_bias'], bins=10, alpha=0.6, label='ΔΣ', color='red')
    ax1.axvline(0, color='k', ls='--', lw=2)
    ax1.axvline(np.mean(results['surfaceDensity']['mass_bias']), color='blue', ls='-', lw=2)
    ax1.axvline(np.mean(results['deltaSigma']['mass_bias']), color='red', ls='-', lw=2)
    ax1.set_xlabel('Mass Bias (log10M_inferred - log10M_true)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.set_title(f'Mass Bias Distribution (N={N_REALIZATIONS})')

    ax2 = axes[1]
    ax2.hist(results['surfaceDensity']['conc_bias'], bins=10, alpha=0.6, label='Σ', color='blue')
    ax2.hist(results['deltaSigma']['conc_bias'], bins=10, alpha=0.6, label='ΔΣ', color='red')
    ax2.axvline(0, color='k', ls='--', lw=2)
    ax2.axvline(np.mean(results['surfaceDensity']['conc_bias']), color='blue', ls='-', lw=2)
    ax2.axvline(np.mean(results['deltaSigma']['conc_bias']), color='red', ls='-', lw=2)
    ax2.set_xlabel('Concentration Bias (c_inferred - c_true)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.set_title(f'Concentration Bias Distribution (N={N_REALIZATIONS})')

    plt.tight_layout()
    plt.savefig('/Users/akumgill/Documents/GitHub/CL-SBI/notebooks/sigma_vs_deltasigma_multirun.png', dpi=150)
    print(f"\nPlot saved to: notebooks/sigma_vs_deltasigma_multirun.png")

    # Statistical test
    from scipy.stats import ttest_ind
    t_mass, p_mass = ttest_ind(results['surfaceDensity']['mass_bias'], results['deltaSigma']['mass_bias'])
    t_conc, p_conc = ttest_ind(results['surfaceDensity']['conc_bias'], results['deltaSigma']['conc_bias'])

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON (t-test)")
    print("=" * 70)
    print(f"Mass bias difference:  t={t_mass:.2f}, p={p_mass:.4f} {'*' if p_mass < 0.05 else ''}")
    print(f"Conc bias difference:  t={t_conc:.2f}, p={p_conc:.4f} {'*' if p_conc < 0.05 else ''}")

    print("\n" + "=" * 70)
    print("CONCLUSION FOR PAPER")
    print("=" * 70)
    sigma_mass_bias = np.mean(results['surfaceDensity']['mass_bias'])
    delta_mass_bias = np.mean(results['deltaSigma']['mass_bias'])
    sigma_conc_bias = np.mean(results['surfaceDensity']['conc_bias'])
    delta_conc_bias = np.mean(results['deltaSigma']['conc_bias'])

    if p_mass > 0.05 and p_conc > 0.05:
        print("\nNo statistically significant difference between Σ and ΔΣ.")
        print("The paper's use of surfaceDensity() is acceptable for this NFW inference task.")
    else:
        if p_mass < 0.05:
            better = "Σ" if abs(sigma_mass_bias) < abs(delta_mass_bias) else "ΔΣ"
            print(f"\nMass: {better} shows significantly less bias (p={p_mass:.4f})")
        if p_conc < 0.05:
            better = "Σ" if abs(sigma_conc_bias) < abs(delta_conc_bias) else "ΔΣ"
            print(f"Concentration: {better} shows significantly less bias (p={p_conc:.4f})")


if __name__ == '__main__':
    main()
