"""
Plotting utilities for CL-SBI.
Compatible with ChainConsumer 1.x API.
"""

import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from chainconsumer import Chain, ChainConsumer, Truth, PlotConfig
import os
from colossus.cosmology import cosmology
from weaklensclustersbi.simulations import wlprofile
import seaborn as sns

script_dir = os.path.dirname(__file__)
plt.style.use(os.path.join(script_dir, "mplstyle.txt"))

param_labels = ["log10mass", "concentration"]
chain_labels = ["join_then_fit", "fit_then_join"]
wide_param_ranges = ((12, 17), (2, 9))
narrow_param_ranges = ((13.5, 15), (4, 6.2))

# High-contrast color palette for JTF vs FTJ
JTF_COLOR = "#1f77b4"  # Standard blue
FTJ_COLOR = "#d62728"  # Red


# =============================================================================
# ChainConsumer 1.x Helper Functions
# =============================================================================

def _array_to_chain(arr: np.ndarray, name: str, columns: list = None, color: str = None) -> Chain:
    """Convert numpy array to ChainConsumer 1.x Chain object."""
    if columns is None:
        columns = param_labels[:arr.shape[1]]
    arr = arr[:, :len(columns)]
    df = pd.DataFrame(arr, columns=columns)
    if color is not None:
        return Chain(samples=df, name=name, color=color)
    return Chain(samples=df, name=name)


def _add_truth(cc: ChainConsumer, truth_values: np.ndarray) -> None:
    """Add truth marker to ChainConsumer."""
    if truth_values is not None and len(truth_values) >= 2:
        truth = Truth(location={
            'log10mass': float(truth_values[0]),
            'concentration': float(truth_values[1])
        })
        cc.add_truth(truth)


# =============================================================================
# Utility Functions (unchanged from original)
# =============================================================================

def compute_dynamic_extents(
    chains, truth=None, padding_factor=0.3, use_percentiles=True
):
    """Compute plot extents dynamically based on chain data."""
    all_samples = []
    for chain in chains:
        if chain is not None and len(chain) > 0:
            samples = np.asarray(chain)
            if samples.ndim == 1:
                continue
            all_samples.append(samples[:, :2])

    if not all_samples:
        return narrow_param_ranges

    combined = np.vstack(all_samples)

    if use_percentiles:
        mass_min, mass_max = np.percentile(combined[:, 0], [1, 99])
        conc_min, conc_max = np.percentile(combined[:, 1], [1, 99])
    else:
        mass_min, mass_max = combined[:, 0].min(), combined[:, 0].max()
        conc_min, conc_max = combined[:, 1].min(), combined[:, 1].max()

    if truth is not None:
        if isinstance(truth, (list, tuple)) and len(truth) > 0:
            for t in truth:
                if t is not None and len(t) >= 2:
                    mass_min = min(mass_min, t[0])
                    mass_max = max(mass_max, t[0])
                    conc_min = min(conc_min, t[1])
                    conc_max = max(conc_max, t[1])
        elif hasattr(truth, "__len__") and len(truth) >= 2:
            mass_min = min(mass_min, truth[0])
            mass_max = max(mass_max, truth[0])
            conc_min = min(conc_min, truth[1])
            conc_max = max(conc_max, truth[1])

    mass_range = mass_max - mass_min
    conc_range = conc_max - conc_min
    mass_range = max(mass_range, 0.5)
    conc_range = max(conc_range, 0.5)

    mass_min -= padding_factor * mass_range
    mass_max += padding_factor * mass_range
    conc_min -= padding_factor * conc_range
    conc_max += padding_factor * conc_range

    return ((mass_min, mass_max), (conc_min, conc_max))


QUARTILE_SIGMA = 0.6744897501960817
Z_95 = 1.6448536269514722
MIN_SIGMA = 1e-3
PERCENTILE_LEVEL_SETS = {
    3: [25, 50, 75],
    7: [5, 16, 25, 50, 75, 84, 95],
}


def logprob_to_numpy(logprob):
    if logprob is None:
        return None
    if hasattr(logprob, "detach"):
        tensor = logprob.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return tensor.numpy()
    return np.asarray(logprob)


def get_percentile_levels(num_levels):
    return PERCENTILE_LEVEL_SETS.get(num_levels)


def median_index(levels):
    if 50 in levels:
        return levels.index(50)
    return len(levels) // 2


def split_percentile_chain(chain):
    if chain.shape[1] <= 2:
        return None
    has_correlation = chain.shape[1] % 2 == 1
    num_levels = (chain.shape[1] - int(has_correlation)) // 2
    levels = get_percentile_levels(num_levels)
    if levels is None:
        return None
    mass_percentiles = chain[:, :num_levels]
    conc_percentiles = chain[:, num_levels : 2 * num_levels]
    correlation = None
    if has_correlation:
        correlation = chain[:, -1]
    return levels, mass_percentiles, conc_percentiles, correlation


def compute_quantile_summary(samples, levels):
    return {
        level: float(np.median(samples[:, idx])) for idx, level in enumerate(levels)
    }


def estimate_sigma_from_quantiles(quantiles):
    candidates = []
    if 84 in quantiles and 16 in quantiles:
        candidates.append((quantiles[84] - quantiles[16]) / 2.0)
    if 75 in quantiles and 25 in quantiles:
        candidates.append((quantiles[75] - quantiles[25]) / (2.0 * QUARTILE_SIGMA))
    if 95 in quantiles and 5 in quantiles:
        candidates.append((quantiles[95] - quantiles[5]) / (2.0 * Z_95))
    candidates = [c for c in candidates if c > 0]
    if not candidates:
        return MIN_SIGMA
    return max(candidates)


def build_gaussian_summary_from_chain(chain):
    chain = np.asarray(chain)
    split = split_percentile_chain(chain)
    if not split:
        return None

    levels, mass_percentiles, conc_percentiles, correlations = split
    med_idx = median_index(levels)
    mass_quantiles = compute_quantile_summary(mass_percentiles, levels)
    conc_quantiles = compute_quantile_summary(conc_percentiles, levels)
    mu_mass = mass_quantiles.get(50, float(np.median(mass_percentiles[:, med_idx])))
    mu_conc = conc_quantiles.get(50, float(np.median(conc_percentiles[:, med_idx])))
    sigma_mass = max(estimate_sigma_from_quantiles(mass_quantiles), MIN_SIGMA)
    sigma_conc = max(estimate_sigma_from_quantiles(conc_quantiles), MIN_SIGMA)
    if correlations is not None:
        rho = float(np.median(np.clip(correlations, -0.99, 0.99)))
    else:
        rho = 0.0

    return {
        "levels": levels,
        "median_index": med_idx,
        "mass_percentiles": mass_percentiles,
        "conc_percentiles": conc_percentiles,
        "correlation_samples": correlations,
        "mass": {"quantiles": mass_quantiles, "mu": mu_mass, "sigma": sigma_mass},
        "concentration": {"quantiles": conc_quantiles, "mu": mu_conc, "sigma": sigma_conc},
        "correlation": {"rho": rho},
    }


def sample_gaussian_nfw_profiles(summary, z, n_samples=200, rng=None, observable="surface_density"):
    if summary is None:
        return None

    rng = np.random.default_rng() if rng is None else rng
    mass_mu = summary["mass"]["mu"]
    conc_mu = summary["concentration"]["mu"]
    mass_sigma = summary["mass"]["sigma"]
    conc_sigma = summary["concentration"]["sigma"]

    rho = summary.get("correlation", {}).get("rho", 0.0)
    rho = float(np.clip(rho, -0.99, 0.99))
    cov = np.array([
        [mass_sigma**2, rho * mass_sigma * conc_sigma],
        [rho * mass_sigma * conc_sigma, conc_sigma**2],
    ])
    samples = rng.multivariate_normal(mean=[mass_mu, conc_mu], cov=cov, size=n_samples)
    mass_samples, conc_samples = samples[:, 0], samples[:, 1]

    profiles = [
        wlprofile.simulate_nfw(float(m), float(c), z=z, kind=observable)
        for m, c in zip(mass_samples, conc_samples)
    ]
    return np.asarray(profiles)


def timestamp():
    import time
    return time.strftime("%Y-%m-%d-%H-%M-%S")


# =============================================================================
# ChainConsumer Plotting Functions (updated for 1.x API)
# =============================================================================

def plot_chainconsumer(chains, out_path, infer_type, true_param=[], mc_pairs=None):
    """Plot chains using ChainConsumer 1.x API."""
    chains = [np.array(chain, copy=True) for chain in chains]

    gaussian_summary = None
    for idx, chain in enumerate(chains):
        summary = build_gaussian_summary_from_chain(chain)
        if not summary:
            continue
        med_idx = summary["median_index"]
        chains[idx] = np.column_stack((
            summary["mass_percentiles"][:, med_idx],
            summary["conc_percentiles"][:, med_idx],
        ))
        if idx == 1:
            gaussian_summary = summary

    if gaussian_summary is not None:
        mass_summary = gaussian_summary["mass"]
        conc_summary = gaussian_summary["concentration"]
        mu_mass = mass_summary["mu"]
        mu_conc = conc_summary["mu"]
        sigma_mass = max(mass_summary["sigma"], MIN_SIGMA)
        sigma_conc = max(conc_summary["sigma"], MIN_SIGMA)
        rho = float(np.clip(gaussian_summary.get("correlation", {}).get("rho", 0.0), -0.99, 0.99))
        cov = np.array([
            [sigma_mass**2, rho * sigma_mass * sigma_conc],
            [rho * sigma_mass * sigma_conc, sigma_conc**2],
        ])

    # Extract truth values
    if len(true_param) >= 3:
        p50 = np.array(true_param[0])
        p25 = np.array(true_param[1])
        p75 = np.array(true_param[2])
    else:
        p50 = np.array((np.median(mc_pairs.T[0]), np.median(mc_pairs.T[1])))
        p25 = p50
        p75 = p50

    # Remove error column from MCMC chains
    if infer_type == "mcmc":
        if chains[0].shape[1] > 2:
            chains[0] = chains[0][:, :2]
        if chains[1].shape[1] > 2:
            chains[1] = chains[1][:, :2]

    cc = ChainConsumer()
    cc.add_chain(_array_to_chain(chains[0], chain_labels[0], color=JTF_COLOR))

    if gaussian_summary is not None:
        rng = np.random.default_rng()
        gauss_chain = rng.multivariate_normal(mean=[mu_mass, mu_conc], cov=cov, size=int(1e7))
        cc.add_chain(_array_to_chain(gauss_chain, chain_labels[1], color=FTJ_COLOR))
    else:
        cc.add_chain(_array_to_chain(chains[1], chain_labels[1], color=FTJ_COLOR))

    _add_truth(cc, p50)

    # Compute dynamic extents
    truth_points = [p25, p50, p75]
    if mc_pairs is not None:
        obs_2sigma_low = np.array([np.percentile(mc_pairs[:, 0], 2.5), np.percentile(mc_pairs[:, 1], 2.5)])
        obs_2sigma_high = np.array([np.percentile(mc_pairs[:, 0], 97.5), np.percentile(mc_pairs[:, 1], 97.5)])
        truth_points.extend([obs_2sigma_low, obs_2sigma_high])
    dynamic_extents = compute_dynamic_extents(chains, truth=truth_points, padding_factor=1)

    # AG (MF feedback): use the SAME axis convention as the aggregate summary figures (Figs 8/10/11),
    # which display the full population spread cleanly: center each axis on the true population mean
    # with a half-width of 4.5x the widest 1-sigma shown (true population OR either chain). This is
    # (i) identical between the MCMC and SBI panels (it depends only on the shared true population and
    # is generous enough that both narrow chains fit), and (ii) wide enough that the full true
    # population is never clipped -- e.g. the high M-c scatter concentration spread (~2.8-6.8).
    if mc_pairs is not None:
        new_extents = []
        for d in range(2):
            mu_t = float(np.median(mc_pairs[:, d]))
            sig_true = float(np.std(mc_pairs[:, d]))
            sig_chain = max((float(np.std(np.asarray(ch)[:, d])) for ch in chains
                             if np.asarray(ch).ndim == 2 and np.asarray(ch).shape[1] > d), default=0.0)
            half = 4.5 * max(sig_true, sig_chain)
            # AG: the window must also COVER every plotted posterior, not just the truth --
            # under strong misspecification (e.g. Prada/Ludlow M-c) a posterior can sit many
            # sigma from the truth median and would otherwise be clipped out of the panel
            # entirely. Span from the lowest to the highest center among truth and chain
            # medians, each padded by the same 4.5-sigma half-width.
            centers = [mu_t] + [float(np.median(np.asarray(ch)[:, d])) for ch in chains
                                if np.asarray(ch).ndim == 2 and np.asarray(ch).shape[1] > d]
            new_extents.append((min(centers) - half, max(centers) + half))
        dynamic_extents = tuple(new_extents)

    cc.set_plot_config(PlotConfig(
        extents={'log10mass': dynamic_extents[0], 'concentration': dynamic_extents[1]},
        # AG (MF feedback): display LaTeX axis labels (consistent with the rest of the draft);
        # internal column key stays 'log10mass'.
        labels={'log10mass': r'$\log_{10} M$', 'concentration': 'Concentration'},
    ))

    fig = cc.plotter.plot(columns=param_labels, figsize=(8, 8))

    # Add label
    fig.text(0.85, 0.85, infer_type.upper(), fontsize=24, weight="bold", ha="center", va="center")

    # KDE overlay
    if mc_pairs is not None:
        sns.kdeplot(x=mc_pairs[:, 0], y=mc_pairs[:, 1], color="black", bw_adjust=1.3,
                    levels=[1 - 0.95, 1 - 0.6827], ax=fig.axes[2], linestyles=["-", "--"])
        sns.kdeplot(x=mc_pairs[:, 0], color="black", ax=fig.axes[0])
        sns.kdeplot(x=mc_pairs[:, 1], color="black", ax=fig.axes[3])

    # Legend - ChainConsumer 1.x puts legend on axes[1] (upper right histogram)
    ax = fig.axes[2]
    cc_leg = fig.axes[1].get_legend()
    if cc_leg:
        cc_leg.set_bbox_to_anchor((0.98, 0.5))
        cc_leg.set_loc("center right")

    legend_handles = [Line2D([], [], color="black", lw=2, ls="-"), Line2D([], [], color="black", lw=2, ls="--")]
    legend_labels = ["True M–C (95%)", "True M–C (68%)"]
    ax.legend(legend_handles, legend_labels, frameon=False, loc="lower left",
              bbox_to_anchor=(0.02, 0.02), handlelength=2.6)

    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.png"))
    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.pdf"))
    plt.close()


def plot_chainconsumer_combined(mcmc_chains, sbi_chains, out_path, true_param=[], mc_pairs=None):
    """Plot combined MCMC and SBI chains."""
    cc = ChainConsumer()

    sbi_chains = [np.array(chain, copy=True) for chain in sbi_chains]
    for idx, chain in enumerate(sbi_chains):
        summary = build_gaussian_summary_from_chain(chain)
        if not summary:
            continue
        med_idx = summary["median_index"]
        sbi_chains[idx] = np.column_stack((
            summary["mass_percentiles"][:, med_idx],
            summary["conc_percentiles"][:, med_idx],
        ))

    # Remove error column
    if np.shape(mcmc_chains[0])[1] > len(param_labels):
        mcmc_chains[0] = np.delete(mcmc_chains[0], 2, 1)
        mcmc_chains[1] = np.delete(mcmc_chains[1], 2, 1)

    cc.add_chain(_array_to_chain(mcmc_chains[0], "mcmc_jtf"))
    cc.add_chain(_array_to_chain(sbi_chains[0], "sbi_jtf"))
    cc.add_chain(_array_to_chain(mcmc_chains[1], "mcmc_ftj"))
    cc.add_chain(_array_to_chain(sbi_chains[1], "sbi_ftj"))

    p25 = np.array(true_param[1])
    p50 = np.array(true_param[0])
    p75 = np.array(true_param[2])
    _add_truth(cc, p50)

    # Compute extents
    truth_mass, truth_conc = p50[0], p50[1]
    default_mass_range, default_conc_range = narrow_param_ranges

    obs_outside_default = False
    if mc_pairs is not None:
        obs_mass_min, obs_mass_max = mc_pairs[:, 0].min(), mc_pairs[:, 0].max()
        obs_conc_min, obs_conc_max = mc_pairs[:, 1].min(), mc_pairs[:, 1].max()
        if (obs_mass_min < default_mass_range[0] or obs_mass_max > default_mass_range[1] or
            obs_conc_min < default_conc_range[0] or obs_conc_max > default_conc_range[1]):
            obs_outside_default = True

    if (default_mass_range[0] <= truth_mass <= default_mass_range[1] and
        default_conc_range[0] <= truth_conc <= default_conc_range[1] and not obs_outside_default):
        dynamic_extents = narrow_param_ranges
    else:
        jtf_chains = [mcmc_chains[0], sbi_chains[0]]
        truth_points = [p25, p50, p75]
        if mc_pairs is not None:
            obs_2sigma_low = np.array([np.percentile(mc_pairs[:, 0], 2.5), np.percentile(mc_pairs[:, 1], 2.5)])
            obs_2sigma_high = np.array([np.percentile(mc_pairs[:, 0], 97.5), np.percentile(mc_pairs[:, 1], 97.5)])
            truth_points.extend([obs_2sigma_low, obs_2sigma_high])
        dynamic_extents = compute_dynamic_extents(jtf_chains, truth=truth_points, padding_factor=1)

    cc.set_plot_config(PlotConfig(
        extents={'log10mass': dynamic_extents[0], 'concentration': dynamic_extents[1]}
    ))
    fig = cc.plotter.plot(columns=param_labels, figsize=(8, 8))

    mass_range, conc_range = dynamic_extents

    def is_close_range(r1, r2, tol=0.1):
        return abs(r1[0] - r2[0]) < tol and abs(r1[1] - r2[1]) < tol

    for ax in fig.get_axes():
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if is_close_range(xlim, mass_range) and ylim[1] < 5:
            ax.axvline(p25[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axvline(p75[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
        elif is_close_range(ylim, conc_range) and xlim[1] < 6:
            ax.axhline(p25[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p75[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
        elif is_close_range(xlim, mass_range) and is_close_range(ylim, conc_range):
            ax.axvline(p25[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axvline(p75[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p25[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p75[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)

    for ax in fig.axes:
        ax.grid(False)
    plt.gcf().subplots_adjust(bottom=0.12, left=0.1)

    plt.savefig(os.path.join(out_path, "mcmc_sbi_cc.png"))
    plt.savefig(os.path.join(out_path, "mcmc_sbi_cc.pdf"))
    plt.close()


def plot_chainconsumer_ftj_comparison(
    mcmc_ftj_joint_chain, mcmc_ftj_population_samples, out_path, true_param, mc_pairs,
    population_params=None, mcmc_ftj_naive_samples=None,
):
    """Plot FTJ method comparison."""
    if mcmc_ftj_joint_chain.shape[1] > 2:
        mcmc_ftj_joint_chain = mcmc_ftj_joint_chain[:, :2]
    if mcmc_ftj_population_samples.shape[1] > 2:
        mcmc_ftj_population_samples = mcmc_ftj_population_samples[:, :2]

    p50 = np.array((np.median(mc_pairs.T[0]), np.median(mc_pairs.T[1])))
    p25 = np.array(true_param[1])
    p75 = np.array(true_param[2])

    cc = ChainConsumer()
    cc.add_chain(_array_to_chain(mcmc_ftj_joint_chain, "FTJ Joint Likelihood"))
    cc.add_chain(_array_to_chain(mcmc_ftj_population_samples, "FTJ Two-Stage"))
    if mcmc_ftj_naive_samples is not None:
        cc.add_chain(_array_to_chain(mcmc_ftj_naive_samples, "FTJ Naive Stacking"))

    _add_truth(cc, p50)

    all_chains = [mcmc_ftj_joint_chain, mcmc_ftj_population_samples]
    if mcmc_ftj_naive_samples is not None:
        all_chains.append(mcmc_ftj_naive_samples)
    dynamic_extents = compute_dynamic_extents(all_chains, truth=[p25, p50, p75], padding_factor=1)

    cc.set_plot_config(PlotConfig(
        extents={'log10mass': dynamic_extents[0], 'concentration': dynamic_extents[1]}
    ))
    fig = cc.plotter.plot(columns=param_labels, figsize=(8, 8))

    fig.text(0.85, 0.85, "MCMC FTJ", fontsize=24, weight="bold", ha="center", va="center")

    # KDE overlay
    sns.kdeplot(x=mc_pairs[:, 0], y=mc_pairs[:, 1], color="black",
                levels=[1 - 0.95, 1 - 0.6827], ax=fig.axes[2], linestyles=["-", "--"])
    sns.kdeplot(x=mc_pairs[:, 0], color="black", ax=fig.axes[0])
    sns.kdeplot(x=mc_pairs[:, 1], color="black", ax=fig.axes[3])

    ax = fig.axes[2]
    # ChainConsumer 1.x puts legend on axes[1] (upper right histogram), not axes[2]
    cc_leg = fig.axes[1].get_legend()
    if cc_leg:
        cc_leg.set_bbox_to_anchor((0.98, 0.5))
        cc_leg.set_loc("center right")

    # Add True M-C legend to main contour plot (lower left)
    legend_handles = [Line2D([], [], color="black", lw=2, ls="-"), Line2D([], [], color="black", lw=2, ls="--")]
    legend_labels = ["True M–C (95%)", "True M–C (68%)"]
    ax.legend(legend_handles, legend_labels, frameon=False, loc="lower left",
              bbox_to_anchor=(0.02, 0.02), handlelength=2.6)

    plt.savefig(os.path.join(out_path, "mcmc_ftj_comparison.png"))
    plt.savefig(os.path.join(out_path, "mcmc_ftj_comparison.pdf"))
    plt.close()


def plot_chainconsumer_mcmc_all_methods(
    mcmc_jtf_chain, mcmc_ftj_joint_chain, mcmc_ftj_population_samples,
    out_path, true_param, mc_pairs, mcmc_ftj_naive_samples=None,
):
    """Plot all MCMC methods comparison."""
    if mcmc_jtf_chain.shape[1] > 2:
        mcmc_jtf_chain = mcmc_jtf_chain[:, :2]
    if mcmc_ftj_joint_chain.shape[1] > 2:
        mcmc_ftj_joint_chain = mcmc_ftj_joint_chain[:, :2]
    if mcmc_ftj_population_samples.shape[1] > 2:
        mcmc_ftj_population_samples = mcmc_ftj_population_samples[:, :2]

    p50 = np.array((np.median(mc_pairs.T[0]), np.median(mc_pairs.T[1])))
    p25 = np.array(true_param[1])
    p75 = np.array(true_param[2])

    cc = ChainConsumer()
    cc.add_chain(_array_to_chain(mcmc_jtf_chain, "JTF"))
    cc.add_chain(_array_to_chain(mcmc_ftj_joint_chain, "FTJ Joint Likelihood"))
    cc.add_chain(_array_to_chain(mcmc_ftj_population_samples, "FTJ Two-Stage"))
    if mcmc_ftj_naive_samples is not None:
        cc.add_chain(_array_to_chain(mcmc_ftj_naive_samples, "FTJ Naive Stacking"))

    _add_truth(cc, p50)

    all_chains = [mcmc_jtf_chain, mcmc_ftj_joint_chain, mcmc_ftj_population_samples]
    if mcmc_ftj_naive_samples is not None:
        all_chains.append(mcmc_ftj_naive_samples)
    dynamic_extents = compute_dynamic_extents(all_chains, truth=[p25, p50, p75], padding_factor=1)

    cc.set_plot_config(PlotConfig(
        extents={'log10mass': dynamic_extents[0], 'concentration': dynamic_extents[1]}
    ))
    fig = cc.plotter.plot(columns=param_labels, figsize=(8, 8))

    fig.text(0.85, 0.85, "MCMC", fontsize=24, weight="bold", ha="center", va="center")

    # KDE overlay
    sns.kdeplot(x=mc_pairs[:, 0], y=mc_pairs[:, 1], color="black",
                levels=[1 - 0.95, 1 - 0.6827], ax=fig.axes[2], linestyles=["-", "--"])
    sns.kdeplot(x=mc_pairs[:, 0], color="black", ax=fig.axes[0])
    sns.kdeplot(x=mc_pairs[:, 1], color="black", ax=fig.axes[3])

    ax = fig.axes[2]
    # ChainConsumer 1.x puts legend on axes[1] (upper right histogram)
    cc_leg = fig.axes[1].get_legend()
    if cc_leg:
        cc_leg.set_bbox_to_anchor((0.98, 0.5))
        cc_leg.set_loc("center right")

    legend_handles = [Line2D([], [], color="black", lw=2, ls="-"), Line2D([], [], color="black", lw=2, ls="--")]
    legend_labels = ["True M–C (95%)", "True M–C (68%)"]
    ax.legend(legend_handles, legend_labels, frameon=False, loc="lower left",
              bbox_to_anchor=(0.02, 0.02), handlelength=2.6)

    plt.savefig(os.path.join(out_path, "mcmc_all_methods.png"))
    plt.savefig(os.path.join(out_path, "mcmc_all_methods.pdf"))
    plt.close()


# =============================================================================
# Diagnostics Plotting (unchanged, no ChainConsumer)
# =============================================================================

def plot_walkers(sampler, out_path, prefix=""):
    ndim = 2
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["log10mass", "concentration"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.savefig(os.path.join(out_path, f"{prefix}mcmc_walkers.png"))
    plt.savefig(os.path.join(out_path, f"{prefix}mcmc_walkers.pdf"))
    plt.close()


def plot_cc_diagnostic(chains, out_path, infer_type, true_param_median=[]):
    """Diagnostic plot with multiple chains."""
    cc = ChainConsumer()
    for i, chain in enumerate(chains):
        if infer_type == "mcmc" and chain.shape[1] > 2:
            chain = chain[:, :2]
        cc.add_chain(_array_to_chain(chain, f"chain_{i}"))

    if len(true_param_median) >= 2:
        _add_truth(cc, np.array(true_param_median))

    fig = cc.plotter.plot(columns=param_labels, figsize=(8, 8))
    for ax in fig.axes:
        ax.grid(False)

    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.png"))
    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.pdf"))
    plt.close()


def plot_mc_pairs(mc_pairs, out_path):
    plt.scatter(mc_pairs[:, 0], mc_pairs[:, 1], s=50)
    plt.scatter(np.median(mc_pairs[:, 0]), np.median(mc_pairs[:, 1]),
                s=100, marker="^", label="median m-c pair")
    plt.xlabel("log$_{10}$M [M$_\odot$]", fontsize="xx-large")
    plt.ylabel("Concentration", fontsize="xx-large")
    plt.title("Drawn mc_pairs")
    plt.legend()
    plt.savefig(os.path.join(out_path, "drawn_mc_pairs.png"))
    plt.savefig(os.path.join(out_path, "drawn_mc_pairs.pdf"))
    plt.close()


# =============================================================================
# NFW Profile Inference Helpers
# =============================================================================

def inferred_mc_from_chains(chains, method="joint_median", log_probs=None, sbi_posterior=None):
    split = split_percentile_chain(chains)

    if split:
        levels, mass_percentiles, conc_percentiles, _ = split
        med_idx = median_index(levels)

        if method == "joint_median":
            inferred_log10mass = np.median(mass_percentiles[:, med_idx])
            inferred_concentration = np.median(conc_percentiles[:, med_idx])
            return (inferred_log10mass, inferred_concentration)
        elif method == "map" and log_probs is not None:
            idx = np.argmax(log_probs)
            return (mass_percentiles[idx, med_idx], conc_percentiles[idx, med_idx])
    else:
        mass_samples = chains[:, 0]
        conc_samples = chains[:, 1]

        if method == "joint_median":
            inferred_log10mass = np.median(mass_samples)
            inferred_concentration = np.median(conc_samples)
            return (inferred_log10mass, inferred_concentration)
        elif method == "map" and log_probs is not None:
            idx = np.argmax(log_probs)
            return (chains[idx, 0], chains[idx, 1])

    raise ValueError("Invalid method. Choose either 'joint_median' or 'map' with log_probs.")


def inferred_nfw_from_chains(chains, z, method="joint_median", log_probs=None, observable="surface_density"):
    inferred_mc_pair = inferred_mc_from_chains(chains, method, log_probs)
    return wlprofile.simulate_nfw(inferred_mc_pair[0], inferred_mc_pair[1], z=z, kind=observable)


def sampled_nfw_profiles(chains, z, n_samples=200, log_probs=None, observable="surface_density"):
    if log_probs is not None:
        weights = np.exp(log_probs - np.max(log_probs))
        weights /= weights.sum()
        idxs = np.random.choice(len(chains), size=n_samples, replace=True, p=weights)
    else:
        idxs = np.random.choice(len(chains), size=n_samples, replace=False)
    mc_samples = chains[idxs]

    split = split_percentile_chain(chains)
    if split:
        levels, mass_percentiles, conc_percentiles, _ = split
        med_idx = median_index(levels)
        mc_samples = np.column_stack((mass_percentiles[idxs, med_idx], conc_percentiles[idxs, med_idx]))
    elif mc_samples.shape[1] > 2:
        mc_samples = mc_samples[:, :2]

    profiles = []
    for log10m, c in mc_samples:
        prof = wlprofile.simulate_nfw(log10m, c, z=z, kind=observable)
        profiles.append(prof)
    return np.array(profiles)


def plot_posterior_band(chains, z, radii, ax=None, color="C0", label=None, observable="surface_density"):
    profiles = sampled_nfw_profiles(chains, z, observable=observable)
    median = np.median(profiles, axis=0)
    lo = np.percentile(profiles, 16, axis=0)
    hi = np.percentile(profiles, 84, axis=0)

    if ax is None:
        fig, ax = plt.subplots()

    ax.fill_between(radii, lo, hi, color=color, alpha=0.3)
    ax.plot(radii, median, color=color, label=label)
    return ax


# =============================================================================
# NFW Profile Plotting (complex, keeping original structure)
# =============================================================================

# Axis labels adapt to the observable so DeltaSigma plots are labeled correctly.
_PROFILE_YLABEL = {
    "surface_density": r"$\Sigma$ [$M_\odot h / kpc^2$]",
    "delta_sigma": r"$\Delta\Sigma$ [$M_\odot h / kpc^2$]",
}
_FRACDIFF_YLABEL = {
    "surface_density": r"$\frac{\Sigma_{model} - \Sigma_{median}}{\Sigma_{median}}$",
    "delta_sigma": r"$\frac{\Delta\Sigma_{model} - \Delta\Sigma_{median}}{\Delta\Sigma_{median}}$",
}


def plot_mcmc_nfw_profiles(
    nfw_profiles, sigmas, out_path, num_radial_bins, min_richness, max_richness, z,
    mcmc_chains=None, mcmc_ftj_samplers=None, mcmc_jtf_sampler=None, true_param_median=None,
    observable="surface_density",
):
    plot_nfw_profiles(
        nfw_profiles, sigmas, out_path, num_radial_bins, min_richness, max_richness, z,
        is_noisy=True, mcmc_chains=mcmc_chains, mcmc_ftj_samplers=mcmc_ftj_samplers,
        mcmc_jtf_sampler=mcmc_jtf_sampler, true_param_median=true_param_median,
        observable=observable,
    )


def plot_sbi_nfw_profiles(
    nfw_profiles, sigmas, out_path, num_radial_bins, min_richness, max_richness, z,
    sbi_chains, sbi_ftj_mc, sbi_jtf_mc, sbi_ftj_logprob, sbi_jtf_logprob, true_param_median,
    observable="surface_density",
):
    plot_nfw_profiles(
        nfw_profiles, sigmas, out_path, num_radial_bins, min_richness, max_richness, z,
        is_noisy=True, sbi_chains=sbi_chains, sbi_ftj_mc=sbi_ftj_mc, sbi_jtf_mc=sbi_jtf_mc,
        sbi_ftj_logprob=sbi_ftj_logprob, sbi_jtf_logprob=sbi_jtf_logprob,
        true_param_median=true_param_median, observable=observable,
    )


def plot_nfw_profiles(
    nfw_profiles, sigmas, out_path, num_radial_bins, min_richness, max_richness, z, is_noisy,
    mcmc_chains=None, sbi_chains=None, mcmc_ftj_samplers=None, mcmc_jtf_sampler=None,
    sbi_ftj_mc=None, sbi_jtf_mc=None, sbi_ftj_logprob=None, sbi_jtf_logprob=None,
    true_param_median=None, observable="surface_density",
):
    """Plot NFW profiles with posterior predictions."""
    cosmo = cosmology.setCosmology("planck18")
    nfw_profiles = 10**nfw_profiles

    rbins = 10 ** np.arange(0, num_radial_bins / 10, 0.1)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 10),
                                    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})
    ax1.loglog()
    ax1.set_xscale("log")
    ax2.set_yscale("linear")

    ax2.set_xlabel("radius [kpc/h]", fontsize="xx-large")
    ax1.set_ylabel(_PROFILE_YLABEL[observable], fontsize="xx-large")
    ax1.set_ylim(1e7, 1e10)
    ax1.set_xlim(min(rbins), max(rbins))

    ax1.plot(rbins, np.median(nfw_profiles, axis=0), "k", linewidth=2.0,
             label=f"Median Drawn NFW, $\lambda \in$ [{min_richness}, {max_richness}]")

    if is_noisy:
        y_med = np.median(nfw_profiles, axis=0)
        p_lo_68, p_hi_68 = np.percentile(nfw_profiles, [16, 84], axis=0)
        p_lo_95, p_hi_95 = np.percentile(nfw_profiles, [2.5, 97.5], axis=0)
        p_lo_997, p_hi_997 = np.percentile(nfw_profiles, [0.15, 99.85], axis=0)

        dx = np.diff(rbins)
        dx = np.r_[dx[0], dx]
        w68, w95, w997 = 0.45 * dx, 0.30 * dx, 0.18 * dx

        ax1.bar(rbins, p_hi_997 - p_lo_997, bottom=p_lo_997, width=w997,
                align="center", color="k", alpha=0.20, linewidth=0, zorder=1)
        ax1.bar(rbins, p_hi_95 - p_lo_95, bottom=p_lo_95, width=w95,
                align="center", color="k", alpha=0.30, linewidth=0, zorder=2)
        ax1.bar(rbins, p_hi_68 - p_lo_68, bottom=p_lo_68, width=w68,
                align="center", color="k", alpha=0.45, linewidth=0, zorder=3)
        ax1.plot(rbins, y_med, color="k", lw=2.5, zorder=4)

    gaussian_summary_ftj = None
    if sbi_chains:
        sbi_ftj_chains = sbi_chains[1]
        gaussian_summary_ftj = build_gaussian_summary_from_chain(sbi_ftj_chains)

    ideal_nfw = None
    if true_param_median is not None:
        ideal_nfw = wlprofile.simulate_nfw(float(true_param_median[0]), float(true_param_median[1]), z=z, kind=observable)

    gaussian_profiles = None
    if gaussian_summary_ftj is not None:
        gaussian_profiles = sample_gaussian_nfw_profiles(gaussian_summary_ftj, z, n_samples=len(nfw_profiles), observable=observable)
        if gaussian_profiles is not None and gaussian_profiles.size:
            gaussian_lo = np.percentile(gaussian_profiles, 16, axis=0)
            gaussian_hi = np.percentile(gaussian_profiles, 84, axis=0)
            ax1.fill_between(rbins, gaussian_lo, gaussian_hi, alpha=0.25, color=FTJ_COLOR,
                            label="SBI fit-then-join 68% CI")

    # MCMC plotting
    if mcmc_chains:
        mcmc_jtf_nfw = inferred_nfw_from_chains(mcmc_chains[0], z, log_probs=mcmc_jtf_sampler.get_log_prob(flat=True), observable=observable)
        mcmc_ftj_nfw = inferred_nfw_from_chains(mcmc_chains[1], z, log_probs=mcmc_ftj_samplers.get_log_prob(flat=True), observable=observable)

        ax1.plot(rbins, mcmc_jtf_nfw, label="MCMC join-then-fit", color=JTF_COLOR)
        ax1.plot(rbins, mcmc_ftj_nfw, linestyle="dashed", label="MCMC fit-then-join", color=FTJ_COLOR)

        mcmc_jtf_nfw_range = sampled_nfw_profiles(mcmc_chains[0], z, n_samples=len(nfw_profiles), observable=observable,
                                                   log_probs=mcmc_jtf_sampler.get_log_prob(flat=True))
        mcmc_jtf_lo = np.percentile(mcmc_jtf_nfw_range, 16, axis=0)
        mcmc_jtf_hi = np.percentile(mcmc_jtf_nfw_range, 84, axis=0)
        ax1.fill_between(rbins, mcmc_jtf_lo, mcmc_jtf_hi, alpha=0.3, color=JTF_COLOR,
                        label="MCMC join-then-fit 68% CI")

        mcmc_ftj_nfw_range = sampled_nfw_profiles(mcmc_chains[1], z, n_samples=len(nfw_profiles), observable=observable,
                                                   log_probs=mcmc_ftj_samplers.get_log_prob(flat=True))
        mcmc_ftj_lo = np.percentile(mcmc_ftj_nfw_range, 16, axis=0)
        mcmc_ftj_hi = np.percentile(mcmc_ftj_nfw_range, 84, axis=0)
        ax1.fill_between(rbins, mcmc_ftj_lo, mcmc_ftj_hi, alpha=0.3, color=FTJ_COLOR,
                        label="MCMC fit-then-join 68% CI")

        ax1.set_ylim(
            min(np.min(mcmc_jtf_nfw), np.min(mcmc_ftj_nfw), np.min(np.median(nfw_profiles, axis=0))) * 0.8,
            max(np.max(mcmc_jtf_nfw), np.max(mcmc_ftj_nfw), np.max(np.median(nfw_profiles, axis=0))) * 1.2,
        )

        if ideal_nfw is not None:
            mcmc_jtf_nfw_range_diff = mcmc_jtf_nfw_range / ideal_nfw - 1
            mcmc_jtf_diff_lo = np.percentile(mcmc_jtf_nfw_range_diff, 16, axis=0)
            mcmc_jtf_diff_hi = np.percentile(mcmc_jtf_nfw_range_diff, 84, axis=0)
            mcmc_ftj_nfw_range_diff = mcmc_ftj_nfw_range / ideal_nfw - 1
            mcmc_ftj_diff_lo = np.percentile(mcmc_ftj_nfw_range_diff, 16, axis=0)
            mcmc_ftj_diff_hi = np.percentile(mcmc_ftj_nfw_range_diff, 84, axis=0)

            ax2.set_ylabel(_FRACDIFF_YLABEL[observable], fontsize="xx-large")  # AG (MF feedback): larger ylabel
            ax2.axhline(0, color="gray", linestyle="dotted", alpha=0.5)

            frac_diff_lo_68 = (p_lo_68 / ideal_nfw) - 1
            frac_diff_hi_68 = (p_hi_68 / ideal_nfw) - 1
            ax2.bar(rbins, frac_diff_hi_68 - frac_diff_lo_68, bottom=frac_diff_lo_68, width=w68,
                    align="center", color="k", alpha=0.45, linewidth=0, zorder=3)

            ax2.plot(rbins, (np.median(nfw_profiles, axis=0) / ideal_nfw) - 1,
                    color="gray", alpha=0.5, linestyle="-.", label="Median Observed NFW")
            ax2.plot(rbins, (mcmc_jtf_nfw / ideal_nfw) - 1, color=JTF_COLOR)
            ax2.plot(rbins, (mcmc_ftj_nfw / ideal_nfw) - 1, color=FTJ_COLOR, linestyle="--")

            if gaussian_profiles is not None and gaussian_profiles.size:
                gaussian_diff = gaussian_profiles / ideal_nfw[None, :] - 1
                gaussian_diff_lo = np.percentile(gaussian_diff, 16, axis=0)
                gaussian_diff_hi = np.percentile(gaussian_diff, 84, axis=0)
                ax2.fill_between(rbins, gaussian_diff_lo, gaussian_diff_hi, alpha=0.25, color=FTJ_COLOR)

            ax2.fill_between(rbins, mcmc_jtf_diff_lo, mcmc_jtf_diff_hi, alpha=0.3, color=JTF_COLOR)
            ax2.fill_between(rbins, mcmc_ftj_diff_lo, mcmc_ftj_diff_hi, alpha=0.3, color=FTJ_COLOR)

        ax1.legend(fontsize="large")
        ax2.legend(fontsize="medium", loc="upper right")
        ax1.text(0.98, 0.98, "MCMC", transform=ax1.transAxes, fontsize="xx-large",
                fontweight="bold", ha="right", va="top")

        plt.savefig(os.path.join(out_path, "mcmc_drawn_nfw_profiles.png"))
        plt.savefig(os.path.join(out_path, "mcmc_drawn_nfw_profiles.pdf"))

    # SBI plotting
    if sbi_chains:
        sbi_jtf_chains = sbi_chains[0]
        sbi_ftj_chains = sbi_chains[1]

        if sbi_jtf_mc is not None:
            sbi_jtf_nfw = wlprofile.simulate_nfw(float(sbi_jtf_mc[0]), float(sbi_jtf_mc[1]), z=z, kind=observable)
            jtf_logprob_np = None
        else:
            jtf_logprob_np = logprob_to_numpy(sbi_jtf_logprob)
            sbi_jtf_nfw = inferred_nfw_from_chains(sbi_jtf_chains, z, log_probs=jtf_logprob_np, observable=observable)

        ax1.plot(rbins, sbi_jtf_nfw, label="SBI join-then-fit", color=JTF_COLOR)
        sbi_ftj_nfw = inferred_nfw_from_chains(sbi_ftj_chains, z, log_probs=logprob_to_numpy(sbi_ftj_logprob), observable=observable)
        ax1.plot(rbins, sbi_ftj_nfw, linestyle="dashed", label="SBI fit-then-join", color=FTJ_COLOR)

        sbi_jtf_nfw_range = sampled_nfw_profiles(sbi_jtf_chains, z, n_samples=len(nfw_profiles), log_probs=jtf_logprob_np, observable=observable)
        sbi_jtf_lo = np.percentile(sbi_jtf_nfw_range, 16, axis=0)
        sbi_jtf_hi = np.percentile(sbi_jtf_nfw_range, 84, axis=0)
        ax1.fill_between(rbins, sbi_jtf_lo, sbi_jtf_hi, alpha=0.3, color=JTF_COLOR,
                        label="SBI join-then-fit 68% CI")

        sbi_ftj_nfw_range = sampled_nfw_profiles(sbi_chains[1], z, n_samples=200, observable=observable,
                                                  log_probs=logprob_to_numpy(sbi_ftj_logprob))
        sbi_ftj_lo = np.percentile(sbi_ftj_nfw_range, 16, axis=0)
        sbi_ftj_hi = np.percentile(sbi_ftj_nfw_range, 84, axis=0)

        ax1.set_ylim(
            min(np.min(sbi_jtf_nfw), np.min(sbi_ftj_nfw), np.min(np.median(nfw_profiles, axis=0))) * 0.8,
            max(np.max(sbi_jtf_nfw), np.max(sbi_ftj_nfw), np.max(np.median(nfw_profiles, axis=0))) * 1.2,
        )

        if ideal_nfw is not None:
            sbi_jtf_nfw_range_diff = sbi_jtf_nfw_range / ideal_nfw - 1
            sbi_jtf_diff_lo = np.percentile(sbi_jtf_nfw_range_diff, 16, axis=0)
            sbi_jtf_diff_hi = np.percentile(sbi_jtf_nfw_range_diff, 84, axis=0)

            ax2.set_ylabel(_FRACDIFF_YLABEL[observable], fontsize="xx-large")  # AG (MF feedback): larger ylabel
            ax2.axhline(0, color="gray", linestyle="dotted", alpha=0.5)

            frac_diff_lo_68 = (p_lo_68 / ideal_nfw) - 1
            frac_diff_hi_68 = (p_hi_68 / ideal_nfw) - 1
            ax2.bar(rbins, frac_diff_hi_68 - frac_diff_lo_68, bottom=frac_diff_lo_68, width=w68,
                    align="center", color="k", alpha=0.45, linewidth=0, zorder=3)

            ax2.plot(rbins, (np.median(nfw_profiles, axis=0) / ideal_nfw) - 1,
                    color="gray", alpha=0.5, linestyle="-.", label="Median Observed NFW")
            ax2.plot(rbins, (sbi_jtf_nfw / ideal_nfw) - 1, color=JTF_COLOR)
            ax2.plot(rbins, (sbi_ftj_nfw / ideal_nfw) - 1, color=FTJ_COLOR, linestyle="--")

            if gaussian_profiles is not None and gaussian_profiles.size:
                gaussian_diff = gaussian_profiles / ideal_nfw[None, :] - 1
                gaussian_diff_lo = np.percentile(gaussian_diff, 16, axis=0)
                gaussian_diff_hi = np.percentile(gaussian_diff, 84, axis=0)
                ax2.fill_between(rbins, gaussian_diff_lo, gaussian_diff_hi, alpha=0.25, color=FTJ_COLOR)

            ax2.fill_between(rbins, sbi_jtf_diff_lo, sbi_jtf_diff_hi, alpha=0.3, color=JTF_COLOR)

        ax1.legend(fontsize="large")
        ax2.legend(fontsize="medium", loc="upper right")
        ax1.text(0.98, 0.98, "SBI", transform=ax1.transAxes, fontsize="xx-large",
                fontweight="bold", ha="right", va="top")

        plt.savefig(os.path.join(out_path, "sbi_drawn_nfw_profiles.png"))
        plt.savefig(os.path.join(out_path, "sbi_drawn_nfw_profiles.pdf"))


def plot_frac_diff(
    nfw_profiles, sigmas, out_path, num_radial_bins, min_richness, max_richness, z,
    mcmc_chains, mcmc_jtf_sampler, mcmc_ftj_samplers, sbi_chains, true_param_median,
    sbi_ftj_logprob, sbi_jtf_logprob, observable="surface_density",
):
    """Plot fractional difference from ideal NFW."""
    plt.figure(figsize=(8, 5))
    plt.xscale("log")
    plt.xlabel("radius [kpc/h]", fontsize="xx-large")
    plt.title(f"Fractional Diff with NFW from Median Drawn M-C $\lambda \in$ [{min_richness}, {max_richness}]",
              fontsize="x-large")
    plt.ylabel(_FRACDIFF_YLABEL[observable], fontsize="xx-large")  # AG (MF feedback): larger ylabel
    nfw_profiles = 10**nfw_profiles

    rbins = 10 ** np.arange(0, num_radial_bins / 10, 0.1)
    ideal_nfw = wlprofile.simulate_nfw(float(true_param_median[0]), float(true_param_median[1]), z=z, kind=observable)

    sbi_jtf_chains = sbi_chains[0]
    sbi_ftj_chains = sbi_chains[1]
    sbi_jtf_nfw = inferred_nfw_from_chains(sbi_jtf_chains, z, log_probs=sbi_jtf_logprob.numpy(), observable=observable)
    sbi_ftj_nfw = inferred_nfw_from_chains(sbi_ftj_chains, z, log_probs=sbi_ftj_logprob.numpy(), observable=observable)
    mcmc_jtf_nfw = inferred_nfw_from_chains(mcmc_chains[0], z, log_probs=mcmc_jtf_sampler.get_log_prob(flat=True), observable=observable)
    mcmc_ftj_nfw = inferred_nfw_from_chains(mcmc_chains[1], z, log_probs=mcmc_ftj_samplers.get_log_prob(flat=True), observable=observable)

    plt.axhline(0, color="gray", linestyle="dotted", alpha=0.5)
    plt.plot(rbins, (sbi_ftj_nfw / ideal_nfw) - 1, color="blue", linestyle="--", label="SBI fit-then-join")
    plt.plot(rbins, (sbi_jtf_nfw / ideal_nfw) - 1, color="blue", label="SBI join-then-fit")
    plt.plot(rbins, (mcmc_ftj_nfw / ideal_nfw) - 1, color="green", linestyle="--", label="MCMC fit-then-join")
    plt.plot(rbins, (mcmc_jtf_nfw / ideal_nfw) - 1, color="green", label="MCMC join-then-fit")
    plt.plot(rbins, (np.median(nfw_profiles, axis=0) / ideal_nfw) - 1,
            color="gray", alpha=0.5, linestyle="-.", label="Median Observed NFW")

    plt.legend(fontsize="large")
    plt.xlim(min(rbins), max(rbins))
    plt.savefig(os.path.join(out_path, "frac_diff.png"))
    plt.savefig(os.path.join(out_path, "frac_diff.pdf"))
