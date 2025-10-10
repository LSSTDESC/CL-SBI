import pygtc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from chainconsumer import ChainConsumer
import os
from colossus.cosmology import cosmology
from context import wlprofile
import seaborn as sns

script_dir = os.path.dirname(__file__)
# out_path =
# plt.style.use("mplstyle.txt")
plt.style.use(os.path.join(script_dir, "mplstyle.txt"))

param_labels = ["log10mass", "concentration"]
chain_labels = ["join_then_fit", "fit_then_join"]
wide_param_ranges = ((12, 17), (2, 9))
narrow_param_ranges = ((13.5, 15), (3.5, 5.5))

QUARTILE_SIGMA = 0.6744897501960817  # z-score at 75th percentile for a normal
Z_95 = 1.6448536269514722  # z-score at 95th percentile for a normal
MIN_SIGMA = 1e-3
PERCENTILE_LEVEL_SETS = {
    3: [25, 50, 75],
    7: [5, 16, 25, 50, 75, 84, 95],
}


def get_percentile_levels(num_levels):
    return PERCENTILE_LEVEL_SETS.get(num_levels)


def median_index(levels):
    if 50 in levels:
        return levels.index(50)
    return len(levels) // 2


def split_percentile_chain(chain):
    if chain.shape[1] <= 2 or chain.shape[1] % 2 != 0:
        return None
    num_levels = chain.shape[1] // 2
    levels = get_percentile_levels(num_levels)
    if levels is None:
        raise AssertionError(
            f"Unsupported number of percentile levels in chain: {chain.shape[1]}"
        )
    mass_percentiles = chain[:, :num_levels]
    conc_percentiles = chain[:, num_levels:]
    return levels, mass_percentiles, conc_percentiles


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

    levels, mass_percentiles, conc_percentiles = split
    med_idx = median_index(levels)
    mass_quantiles = compute_quantile_summary(mass_percentiles, levels)
    conc_quantiles = compute_quantile_summary(conc_percentiles, levels)
    mu_mass = mass_quantiles.get(50, float(np.median(mass_percentiles[:, med_idx])))
    mu_conc = conc_quantiles.get(50, float(np.median(conc_percentiles[:, med_idx])))
    sigma_mass = max(estimate_sigma_from_quantiles(mass_quantiles), MIN_SIGMA)
    sigma_conc = max(estimate_sigma_from_quantiles(conc_quantiles), MIN_SIGMA)

    return {
        "levels": levels,
        "median_index": med_idx,
        "mass_percentiles": mass_percentiles,
        "conc_percentiles": conc_percentiles,
        "mass": {
            "quantiles": mass_quantiles,
            "mu": mu_mass,
            "sigma": sigma_mass,
        },
        "concentration": {
            "quantiles": conc_quantiles,
            "mu": mu_conc,
            "sigma": sigma_conc,
        },
    }


def sample_gaussian_nfw_profiles(summary, z, n_samples=200, rng=None):
    if summary is None:
        return None

    rng = np.random.default_rng() if rng is None else rng
    mass_mu = summary["mass"]["mu"]
    conc_mu = summary["concentration"]["mu"]
    mass_sigma = summary["mass"]["sigma"]
    conc_sigma = summary["concentration"]["sigma"]

    mass_samples = rng.normal(mass_mu, mass_sigma, size=n_samples)
    conc_samples = rng.normal(conc_mu, conc_sigma, size=n_samples)

    profiles = [
        wlprofile.simulate_nfw(float(m), float(c), z=z)
        for m, c in zip(mass_samples, conc_samples)
    ]
    return np.asarray(profiles)


def timestamp():
    import time

    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    return timestr


def plot_pygtc(chains, out_path, infer_type, true_param_median=()):
    # posterFont = {'family': 'Arial', 'size': 18}

    GTC = pygtc.plotGTC(
        chains=chains,
        chainLabels=chain_labels,
        paramNames=param_labels,
        figureSize=8.0,
        # paramRanges=wide_param_ranges,
        sigmaContourLevels=True,
        plotDensity=True,
        truths=true_param_median,
        # customLabelFont=posterFont,
        # customTickFont=posterFont,
        # customLegendFont=posterFont,
        nContourLevels=2,
    )

    GTC.savefig(os.path.join(out_path, f"{infer_type}_gtc.png"))
    GTC.savefig(os.path.join(out_path, f"{infer_type}_gtc.pdf"))
    plt.close(GTC)


def plot_chainconsumer(chains, out_path, infer_type, true_param=[], mc_pairs=None):
    chains = [np.array(chain, copy=True) for chain in chains]

    gaussian_summary = None
    for idx, chain in enumerate(chains):
        summary = build_gaussian_summary_from_chain(chain)
        if not summary:
            continue

        med_idx = summary["median_index"]
        chains[idx] = np.column_stack(
            (
                summary["mass_percentiles"][:, med_idx],
                summary["conc_percentiles"][:, med_idx],
            )
        )

        if gaussian_summary is None:
            gaussian_summary = {
                "mass": summary["mass"],
                "concentration": summary["concentration"],
            }

    p50 = np.array((np.median(mc_pairs.T[0]), np.median(mc_pairs.T[1])))
    p25 = np.array(
        (
            np.percentile(mc_pairs.T[0], 25),
            np.percentile(mc_pairs.T[1], 25),
        )
    )
    p75 = np.array(
        (
            np.percentile(mc_pairs.T[0], 75),
            np.percentile(mc_pairs.T[1], 75),
        )
    )

    # In MCMC we also fit for the error. We don't need to plot that so pruning that param from the data
    # TODO: do we need this?
    if infer_type == "mcmc":
        chains[0] = np.delete(chains[0], 2, 1)
        chains[1] = np.delete(chains[1], 2, 1)

    cc = ChainConsumer()

    cc.add_chain(chains[0], parameters=param_labels, name=chain_labels[0])
    cc.add_chain(chains[1], parameters=param_labels, name=chain_labels[1])
    # cc.add_chain(Chain(samples=chains[0], name=chain_labels[0]))
    # cc.add_chain(Chain(samples=chains[1], name=chain_labels[1]))
    cc.configure(
        statistics="mean",
        summary=True,
        label_font_size=20,
        tick_font_size=16,
        usetex=False,
        serif=False,
        sigmas=[2, 3],
        # sigmas=[1, 2],
        shade_alpha=0.3,
    )

    # Manually plot the truth values
    # p25 = np.array(true_param[1])  # 25th percentile
    # p50 = np.array(true_param[0])  # median
    # p75 = np.array(true_param[2])  # 75th percentile

    # Set the median as the truth value
    fig = cc.plotter.plot(
        truth=p50,
        parameters=param_labels,
        extents=list(narrow_param_ranges),
        # extents=list(wide_param_ranges),
        figsize=(8, 8),
    )

    # These are your actual data extents
    mass_range = wide_param_ranges[0]
    conc_range = wide_param_ranges[1]

    # TODO: there has to be a cleaner solution than this to plotting 25 and 75 percentiles
    def is_close_range(r1, r2, tol=0.1):
        return abs(r1[0] - r2[0]) < tol and abs(r1[1] - r2[1]) < tol

    for ax in fig.get_axes():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Top histogram: log10mass only
        if is_close_range(xlim, mass_range) and ylim[1] < 5:
            ax.axvline(p25[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axvline(p75[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # Right histogram: concentration only
        elif is_close_range(ylim, conc_range) and xlim[1] < 6:
            ax.axhline(p25[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p75[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # Bottom-left 2D plot: both mass and concentration
        elif is_close_range(xlim, mass_range) and is_close_range(ylim, conc_range):
            ax.axvline(p25[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axvline(p75[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p25[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p75[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Add a large text label in the top-right empty space
    fig.text(
        0.85,
        0.85,
        infer_type.upper(),
        fontsize=24,
        weight="bold",
        ha="center",
        va="center",
    )

    # Main plot
    sns.kdeplot(
        x=mc_pairs[:, 0],
        y=mc_pairs[:, 1],
        color="black",
        # levels that map to 2 sigmas included and 3 sigmas included
        levels=[1 - 0.997, 1 - 0.95],
        ax=fig.axes[2],
        linestyles=["-", "--"],
        # legend=True,
    )
    # Mass plot
    sns.kdeplot(
        x=mc_pairs[:, 0],
        color="black",
        ax=fig.axes[0],
    )
    # Conc plot
    sns.kdeplot(
        y=mc_pairs[:, 1],
        color="black",
        ax=fig.axes[3],
    )

    # hack to show the legend with correct line styles
    true95 = Line2D([0], [0], color="black", lw=2, ls="-")
    true997 = Line2D([0], [0], color="black", lw=2, ls="--")
    # h, l = fig.axes[2].get_legend_handles_labels()
    # fig.axes[2].legend(
    #     h + [true95, true997], l + ["True M-C (99.7%)", "True M-C (95%)"]
    # )
    # plt.legend()
    ax = fig.axes[2]  # Main plot
    # keep ChainConsumer's legend
    cc_leg = ax.get_legend()

    # add a second legend for the true KDE
    legend_handles = [
        Line2D([], [], color="black", lw=2, ls="-"),
        Line2D([], [], color="black", lw=2, ls="--"),
    ]
    legend_labels = ["True M–C (99.7%)", "True M–C (95%)"]

    if gaussian_summary is not None:
        legend_handles.extend(
            [
                Line2D([], [], color="purple", lw=1.5, ls=":"),
                Line2D([], [], color="purple", lw=1.5, ls="--"),
            ]
        )
        legend_labels.extend(["SBI Pop Gaussian (1σ)", "SBI Pop Gaussian (2σ)"])

    leg2 = ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        handlelength=2.6,
    )
    ax.add_artist(cc_leg)  # re-add CC legend so both show

    if gaussian_summary is not None:
        mass_summary = gaussian_summary["mass"]
        conc_summary = gaussian_summary["concentration"]
        mu_mass = mass_summary["mu"]
        mu_conc = conc_summary["mu"]
        sigma_mass = max(mass_summary["sigma"], MIN_SIGMA)
        sigma_conc = max(conc_summary["sigma"], MIN_SIGMA)

        ellipse_levels = [(1.0, ":"), (2.0, "--")]
        for scale, linestyle in ellipse_levels:
            ellipse = Ellipse(
                (mu_mass, mu_conc),
                width=2 * scale * sigma_mass,
                height=2 * scale * sigma_conc,
                edgecolor="purple",
                linestyle=linestyle,
                linewidth=1.5,
                fill=False,
                alpha=0.7,
            )
            ax.add_patch(ellipse)

        # Overlay 1D Gaussian marginals on the histograms
        xs = np.linspace(mu_mass - 4 * sigma_mass, mu_mass + 4 * sigma_mass, 200)
        mass_pdf = np.exp(-0.5 * ((xs - mu_mass) / sigma_mass) ** 2) / (
            sigma_mass * np.sqrt(2 * np.pi)
        )
        ax_mass = fig.axes[0]
        ax_mass.plot(xs, mass_pdf, color="purple", linestyle=":", linewidth=1.5)
        # for offset, style in ((sigma_mass, ":"), (2 * sigma_mass, "--")):
        #     ax_mass.axvline(
        #         mu_mass - offset,
        #         color="purple",
        #         linestyle=style,
        #         linewidth=1.2,
        #         alpha=0.7,
        #     )
        #     ax_mass.axvline(
        #         mu_mass + offset,
        #         color="purple",
        #         linestyle=style,
        #         linewidth=1.2,
        #         alpha=0.7,
        #     )

        ys = np.linspace(mu_conc - 4 * sigma_conc, mu_conc + 4 * sigma_conc, 200)
        conc_pdf = np.exp(-0.5 * ((ys - mu_conc) / sigma_conc) ** 2) / (
            sigma_conc * np.sqrt(2 * np.pi)
        )
        ax_conc = fig.axes[3]
        ax_conc.plot(conc_pdf, ys, color="purple", linestyle=":", linewidth=1.5)

    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.png"))
    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.pdf"))
    plt.close()


def plot_chainconsumer_combined(mcmc_chains, sbi_chains, out_path, true_param=[]):
    cc = ChainConsumer()

    sbi_chains = [np.array(chain, copy=True) for chain in sbi_chains]
    for idx, chain in enumerate(sbi_chains):
        summary = build_gaussian_summary_from_chain(chain)
        if not summary:
            continue
        med_idx = summary["median_index"]
        sbi_chains[idx] = np.column_stack(
            (
                summary["mass_percentiles"][:, med_idx],
                summary["conc_percentiles"][:, med_idx],
            )
        )

    # In MCMC we also fit for the error. We don't need to plot that so pruning that param from the data
    # TODO: clean this up
    if np.shape(mcmc_chains[0])[1] > len(param_labels):
        mcmc_chains[0] = np.delete(mcmc_chains[0], 2, 1)
        mcmc_chains[1] = np.delete(mcmc_chains[1], 2, 1)

    cc.add_chain(mcmc_chains[0], parameters=param_labels, name="mcmc_jtf")
    cc.add_chain(sbi_chains[0], parameters=param_labels, name="sbi_jtf")
    cc.add_chain(mcmc_chains[1], parameters=param_labels, name="mcmc_ftj")
    cc.add_chain(sbi_chains[1], parameters=param_labels, name="sbi_ftj")
    # cc.add_chain(Chain(samples=mcmc_chains[0], name='mcmc_jtf'))
    # cc.add_chain(Chain(samples=sbi_chains[0], name='sbi_jtf'))
    # cc.add_chain(Chain(samples=mcmc_chains[1], name='mcmc_ftj'))
    # cc.add_chain(Chain(samples=sbi_chains[1], name='sbi_ftj'))

    # cc.add_chain(chains[1], parameters=param_labels, name=chain_labels[1])
    cc.configure(
        statistics="mean",
        summary=True,
        label_font_size=20,
        tick_font_size=16,
        usetex=False,
        serif=False,
        # sigmas=[2, 3],
        sigmas=[1],
        shade_alpha=0.3,
    )
    # Manually plot the truth values
    p25 = np.array(true_param[1])  # 25th percentile
    p50 = np.array(true_param[0])  # median
    p75 = np.array(true_param[2])  # 75th percentile

    fig = cc.plotter.plot(
        truth=p50,
        parameters=param_labels,
        extents=list(narrow_param_ranges),
        # extents=list(wide_param_ranges),
        figsize=(8, 8),
    )

    # These are your actual data extents
    mass_range = wide_param_ranges[0]
    conc_range = wide_param_ranges[1]

    # TODO: there has to be a cleaner solution than this to plotting 25 and 75 percentiles
    def is_close_range(r1, r2, tol=0.1):
        return abs(r1[0] - r2[0]) < tol and abs(r1[1] - r2[1]) < tol

    for ax in fig.get_axes():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Top histogram: log10mass only
        if is_close_range(xlim, mass_range) and ylim[1] < 5:
            ax.axvline(p25[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axvline(p75[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # Right histogram: concentration only
        elif is_close_range(ylim, conc_range) and xlim[1] < 6:
            ax.axhline(p25[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p75[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # Bottom-left 2D plot: both mass and concentration
        elif is_close_range(xlim, mass_range) and is_close_range(ylim, conc_range):
            ax.axvline(p25[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axvline(p75[0], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p25[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(p75[1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_list = fig.axes
    plt.gcf().subplots_adjust(bottom=0.12)
    plt.gcf().subplots_adjust(left=0.1)
    for ax in ax_list:
        ax.grid(False)

    plt.savefig(os.path.join(out_path, f"mcmc_sbi_cc.png"))
    plt.savefig(os.path.join(out_path, f"mcmc_sbi_cc.pdf"))
    plt.close()


### DIAGNOSTICS PLOTTING BELOW ###


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


# Overplotting multiple chains (for each observation)
def plot_cc_diagnostic(chains, out_path, infer_type, true_param_median=[]):
    cc = ChainConsumer()
    i = 0
    for chain in chains:
        # In MCMC we also fit for the error. We don't need to plot that so pruning that param from the data
        # TODO: clean this up
        if infer_type == "mcmc":  # np.shape(chain)[1] > len(param_labels):
            chain = np.delete(chain, 2, 1)

        cc.add_chain(chain, parameters=param_labels)
        i += 1
    cc.configure(
        statistics="max",
        summary=True,
        # label_font_size=20,
        tick_font_size=16,
        usetex=False,
        serif=False,
        # sigmas=[2, 3],
        sigmas=[1],
    )
    fig = cc.plotter.plot(
        truth=true_param_median,
        parameters=param_labels,
        # extents=list(wide_param_ranges),
        figsize=(8, 8),
    )
    ax_list = fig.axes
    for ax in ax_list:
        ax.grid(False)

    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.png"))
    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.pdf"))
    plt.close()


def plot_mc_pairs(mc_pairs, out_path):
    plt.scatter(mc_pairs[:, 0], mc_pairs[:, 1], s=50)
    plt.scatter(
        np.median(mc_pairs[:, 0]),
        np.median(mc_pairs[:, 1]),
        s=100,
        marker="^",
        label="median m-c pair",
    )
    plt.xlabel("log$_{10}$M [M$_\odot$]", fontsize="xx-large")
    plt.ylabel("Concentration", fontsize="xx-large")
    plt.title("Drawn mc_pairs")
    plt.legend()
    plt.savefig(os.path.join(out_path, f"drawn_mc_pairs.png"))
    plt.savefig(os.path.join(out_path, f"drawn_mc_pairs.pdf"))
    plt.close()


def plot_mcmc_nfw_profiles(
    nfw_profiles,
    sigmas,
    out_path,
    num_radial_bins,
    min_richness,
    max_richness,
    z,
    mcmc_chains=None,
    mcmc_ftj_samplers=None,
    mcmc_jtf_sampler=None,
    true_param_median=None,
):
    plot_nfw_profiles(
        nfw_profiles,
        sigmas,
        out_path,
        num_radial_bins,
        min_richness,
        max_richness,
        z,
        is_noisy=True,
        mcmc_chains=mcmc_chains,
        mcmc_ftj_samplers=mcmc_ftj_samplers,
        mcmc_jtf_sampler=mcmc_jtf_sampler,
        true_param_median=true_param_median,
    )


def plot_sbi_nfw_profiles(
    nfw_profiles,
    sigmas,
    out_path,
    num_radial_bins,
    min_richness,
    max_richness,
    z,
    sbi_chains,
    sbi_ftj_mc,
    sbi_jtf_mc,
    sbi_ftj_logprob,
    sbi_jtf_logprob,
    true_param_median,
):
    plot_nfw_profiles(
        nfw_profiles,
        sigmas,
        out_path,
        num_radial_bins,
        min_richness,
        max_richness,
        z,
        is_noisy=True,
        sbi_chains=sbi_chains,
        sbi_ftj_mc=sbi_ftj_mc,
        sbi_jtf_mc=sbi_jtf_mc,
        sbi_ftj_logprob=sbi_ftj_logprob,
        sbi_jtf_logprob=sbi_jtf_logprob,
        true_param_median=true_param_median,
    )


def plot_nfw_profiles(
    nfw_profiles,
    sigmas,
    out_path,
    num_radial_bins,
    min_richness,
    max_richness,
    z,
    is_noisy,
    mcmc_chains=None,
    sbi_chains=None,
    mcmc_ftj_samplers=None,
    mcmc_jtf_sampler=None,
    sbi_ftj_mc=None,
    sbi_jtf_mc=None,
    sbi_ftj_logprob=None,
    sbi_jtf_logprob=None,
    true_param_median=None,
):
    cosmo = cosmology.setCosmology("planck18")

    nfw_profiles = 10**nfw_profiles  # convert back from log space

    rbins = 10 ** np.arange(0, num_radial_bins / 10, 0.1)
    fig, (ax1, ax2) = plt.subplots(
        2,
        sharex=True,
        figsize=(8, 10),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )
    ax1.loglog()
    ax1.set_xscale("log")
    ax2.set_yscale("linear")

    # plt.figure(figsize=(8, 8))
    ax2.set_xlabel("radius [kpc/h]", fontsize="xx-large")
    ax1.set_ylabel("$ \Delta \Sigma$ [$M_\odot h / kpc^2$]", fontsize="xx-large")
    ax1.set_ylim(1e7, 1e10)
    ax1.set_xlim(min(rbins), max(rbins))

    for nfw_profile in nfw_profiles:
        ax1.plot(rbins, nfw_profile, "-", alpha=0.1, zorder=0, color="gray")
    ax1.plot(
        rbins,
        np.median(nfw_profiles, axis=0),
        "k",
        linewidth=2.0,
        label=f"Median Drawn NFW, $\lambda \in$ [{min_richness}, {max_richness}]",
    )
    if is_noisy:
        upper_error = np.exp(
            np.log(np.median(nfw_profiles, axis=0)) + sigmas
        ) - np.median(nfw_profiles, axis=0)
        lower_error = np.median(nfw_profiles, axis=0) - np.exp(
            np.log(np.median(nfw_profiles, axis=0)) - sigmas
        )
        yerr = [lower_error, upper_error]

        ax1.errorbar(
            rbins,
            np.median(nfw_profiles, axis=0),
            yerr=yerr,
            fmt="k",
            capsize=3.0,
            linewidth=4,
            elinewidth=1,
            zorder=1,
        )

    gaussian_summary = None
    if sbi_chains:
        gaussian_summary = build_gaussian_summary_from_chain(sbi_chains[0])

    ideal_nfw = None
    if true_param_median is not None:
        ideal_nfw = wlprofile.simulate_nfw(
            float(true_param_median[0]), float(true_param_median[1]), z=z
        )

    gaussian_profiles = None
    if gaussian_summary is not None:
        gaussian_profiles = sample_gaussian_nfw_profiles(
            gaussian_summary, z, n_samples=200
        )
        if gaussian_profiles is not None and gaussian_profiles.size:
            gaussian_lo = np.percentile(gaussian_profiles, 16, axis=0)
            gaussian_hi = np.percentile(gaussian_profiles, 84, axis=0)
            ax1.fill_between(
                rbins,
                gaussian_lo,
                gaussian_hi,
                alpha=0.25,
                color="purple",
                label="SBI Pop Gaussian 68% CI",
            )

    if mcmc_chains:
        mcmc_jtf_nfw = inferred_nfw_from_chains(
            mcmc_chains[0],
            z,
            method="map",
            log_probs=mcmc_jtf_sampler.get_log_prob(flat=True),
        )
        mcmc_ftj_nfw = inferred_nfw_from_chains(
            mcmc_chains[1],
            z,
            method="map",
            log_probs=mcmc_ftj_samplers.get_log_prob(flat=True),
        )

        ax1.plot(
            rbins,
            mcmc_jtf_nfw,
            label=f"MCMC join-then-fit",
            color="blue",
        )
        ax1.plot(
            rbins,
            mcmc_ftj_nfw,
            linestyle="dashed",
            label=f"MCMC fit-then-join",
            color="green",
        )

        # Plot the range of drawn NFW profiles from MCMC chains
        mcmc_jtf_nfw_range = sampled_nfw_profiles(
            mcmc_chains[0],
            z,
            n_samples=200,
            log_probs=mcmc_jtf_sampler.get_log_prob(flat=True),
        )
        mcmc_jtf_lo = np.percentile(mcmc_jtf_nfw_range, 16, axis=0)
        mcmc_jtf_hi = np.percentile(mcmc_jtf_nfw_range, 84, axis=0)

        ax1.fill_between(
            rbins,
            mcmc_jtf_lo,
            mcmc_jtf_hi,
            alpha=0.3,
            color="blue",
            label="MCMC join-then-fit 68\% CI",
        )

        mcmc_ftj_nfw_range = sampled_nfw_profiles(
            mcmc_chains[1],
            z,
            n_samples=200,
            log_probs=mcmc_ftj_samplers.get_log_prob(flat=True),
        )
        mcmc_ftj_lo = np.percentile(mcmc_ftj_nfw_range, 16, axis=0)
        mcmc_ftj_hi = np.percentile(mcmc_ftj_nfw_range, 84, axis=0)
        ax1.fill_between(
            rbins,
            mcmc_ftj_lo,
            mcmc_ftj_hi,
            alpha=0.3,
            color="green",
            label="MCMC join-then-fit 68\% CI",
        )
        ax1.set_ylim(
            min(
                np.min(mcmc_jtf_nfw),
                np.min(mcmc_ftj_nfw),
                np.min(np.median(nfw_profiles, axis=0)),
                # np.min(lo),  # include band
            )
            * 0.8,
            max(
                np.max(mcmc_jtf_nfw),
                np.max(mcmc_ftj_nfw),
                np.max(np.median(nfw_profiles, axis=0)),
                # np.max(hi),  # include band
            )
            * 1.2,
        )
        # Rescale the axes
        # ax = ax1.gca()
        # ax.relim()
        # ax.autoscale_view()

        if ideal_nfw is not None:
            mcmc_jtf_nfw_range_diff = mcmc_jtf_nfw_range / ideal_nfw - 1
            mcmc_jtf_diff_lo = np.percentile(mcmc_jtf_nfw_range_diff, 16, axis=0)
            mcmc_jtf_diff_hi = np.percentile(mcmc_jtf_nfw_range_diff, 84, axis=0)

            mcmc_ftj_nfw_range_diff = mcmc_ftj_nfw_range / ideal_nfw - 1
            mcmc_ftj_diff_lo = np.percentile(mcmc_ftj_nfw_range_diff, 16, axis=0)
            mcmc_ftj_diff_hi = np.percentile(mcmc_ftj_nfw_range_diff, 84, axis=0)

        if ideal_nfw is not None:
            # ax2.xlabel("radius [kpc/h]", fontsize="xx-large")
            # ax2.title(
            #     f"Fractional Diff with NFW from Median Drawn M-C $\lambda \in$ [{min_richness}, {max_richness}]",
            #     fontsize="x-large",
            # )
            ax2.set_ylabel(
                r"$\frac{\Delta \Sigma_{model} - \Delta \Sigma_{median}}{\Delta \Sigma_{median}}$",
            )
            ax2.axhline(
                0, color="gray", linestyle="dotted", alpha=0.5
            )  # , label='Median NFW Profile')
            ax2.plot(
                rbins,
                # (mcmc_jtf_nfw / np.median(nfw_profiles, axis=0)) - 1,
                (np.median(nfw_profiles, axis=0) / ideal_nfw) - 1,
                color="gray",
                alpha=0.5,
                linestyle="-.",
                label="Median Observed NFW",
            )
            ax2.plot(
                rbins,
                # (mcmc_jtf_nfw / np.median(nfw_profiles, axis=0)) - 1,
                (mcmc_jtf_nfw / ideal_nfw) - 1,
                color="blue",
                # linestyle="-.",
                label="MCMC join-then-fit",
            )
            ax2.plot(
                rbins,
                # (mcmc_ftj_nfw / np.median(nfw_profiles, axis=0)) - 1,
                (mcmc_ftj_nfw / ideal_nfw) - 1,
                color="green",
                linestyle="--",
                # linewidth=3,
                label="MCMC fit-then-join",
            )
            if gaussian_profiles is not None and gaussian_profiles.size:
                gaussian_diff = gaussian_profiles / ideal_nfw[None, :] - 1
                gaussian_diff_lo = np.percentile(gaussian_diff, 16, axis=0)
                gaussian_diff_hi = np.percentile(gaussian_diff, 84, axis=0)
                ax2.fill_between(
                    rbins,
                    gaussian_diff_lo,
                    gaussian_diff_hi,
                    alpha=0.25,
                    color="purple",
                    label="SBI Pop Gaussian 68% CI",
                )
            ax2.fill_between(
                rbins,
                mcmc_jtf_diff_lo,
                mcmc_jtf_diff_hi,
                alpha=0.3,
                color="blue",
                label="MCMC join-then-fit 68\% CI",
            )
            ax2.fill_between(
                rbins,
                mcmc_ftj_diff_lo,
                mcmc_ftj_diff_hi,
                alpha=0.3,
                color="green",
                label="MCMC join-then-fit 68\% CI",
            )

        ax1.legend(fontsize="large")
        ax2.legend(fontsize="large")

        if is_noisy:
            plt.savefig(os.path.join(out_path, f"mcmc_drawn_nfw_profiles.png"))
            plt.savefig(os.path.join(out_path, f"mcmc_drawn_nfw_profiles.pdf"))
        else:
            plt.savefig(os.path.join(out_path, f"noiseless_drawn_nfw_profiles.png"))
            plt.savefig(os.path.join(out_path, f"noiseless_drawn_nfw_profiles.pdf"))

    if sbi_chains:
        sbi_jtf_nfw = inferred_nfw_from_chains(
            sbi_chains[0],
            z,
            method="map",
            log_probs=sbi_jtf_logprob.numpy(),
        )
        ax1.plot(
            rbins,
            sbi_jtf_nfw,
            label=f"SBI join-then-fit",
            color="blue",
        )
        sbi_ftj_nfw = inferred_nfw_from_chains(
            sbi_chains[1],
            z,
            method="map",
            log_probs=sbi_ftj_logprob.numpy(),
        )
        ax1.plot(
            rbins,
            sbi_ftj_nfw,
            linestyle="dashed",
            label=f"SBI fit-then-join",
            color="green",
        )
        sbi_jtf_nfw_range = sampled_nfw_profiles(
            sbi_chains[0],
            z,
            n_samples=200,
            log_probs=sbi_jtf_logprob.numpy(),
        )
        sbi_jtf_lo = np.percentile(sbi_jtf_nfw_range, 16, axis=0)
        sbi_jtf_hi = np.percentile(sbi_jtf_nfw_range, 84, axis=0)

        ax1.fill_between(
            rbins,
            sbi_jtf_lo,
            sbi_jtf_hi,
            alpha=0.3,
            color="blue",
            label="SBI fit-then-join 68\% CI",
        )

        sbi_ftj_nfw_range = sampled_nfw_profiles(
            sbi_chains[1],
            z,
            n_samples=200,
            log_probs=sbi_ftj_logprob.numpy(),
        )
        sbi_ftj_lo = np.percentile(sbi_ftj_nfw_range, 16, axis=0)
        sbi_ftj_hi = np.percentile(sbi_ftj_nfw_range, 84, axis=0)
        ax1.fill_between(
            rbins,
            sbi_ftj_lo,
            sbi_ftj_hi,
            alpha=0.3,
            color="green",
            label="SBI join-then-fit 68\% CI",
        )
        ax1.set_ylim(
            min(
                np.min(sbi_jtf_nfw),
                np.min(sbi_ftj_nfw),
                np.min(np.median(nfw_profiles, axis=0)),
                # np.min(lo),  # include band
            )
            * 0.8,
            max(
                np.max(sbi_jtf_nfw),
                np.max(sbi_ftj_nfw),
                np.max(np.median(nfw_profiles, axis=0)),
                # np.max(hi),  # include band
            )
            * 1.2,
        )

        if ideal_nfw is not None:
            sbi_jtf_nfw_range_diff = sbi_jtf_nfw_range / ideal_nfw - 1
            sbi_jtf_diff_lo = np.percentile(sbi_jtf_nfw_range_diff, 16, axis=0)
            sbi_jtf_diff_hi = np.percentile(sbi_jtf_nfw_range_diff, 84, axis=0)

            sbi_ftj_nfw_range_diff = sbi_ftj_nfw_range / ideal_nfw - 1
            sbi_ftj_diff_lo = np.percentile(sbi_ftj_nfw_range_diff, 16, axis=0)
            sbi_ftj_diff_hi = np.percentile(sbi_ftj_nfw_range_diff, 84, axis=0)

            ax2.set_ylabel(
                r"$\frac{\Delta \Sigma_{model} - \Delta \Sigma_{median}}{\Delta \Sigma_{median}}$",
            )
            ax2.axhline(
                0, color="gray", linestyle="dotted", alpha=0.5
            )  # , label='Median NFW Profile')
            ax2.plot(
                rbins,
                (np.median(nfw_profiles, axis=0) / ideal_nfw) - 1,
                color="gray",
                alpha=0.5,
                linestyle="-.",
                label="Median Observed NFW",
            )
            ax2.plot(
                rbins,
                (sbi_jtf_nfw / ideal_nfw) - 1,
                color="blue",
                label="SBI join-then-fit",
            )
            ax2.plot(
                rbins,
                (sbi_ftj_nfw / ideal_nfw) - 1,
                color="green",
                linestyle="--",
                label="SBI fit-then-join",
            )
            if gaussian_profiles is not None and gaussian_profiles.size:
                gaussian_diff = gaussian_profiles / ideal_nfw[None, :] - 1
                gaussian_diff_lo = np.percentile(gaussian_diff, 16, axis=0)
                gaussian_diff_hi = np.percentile(gaussian_diff, 84, axis=0)
                ax2.fill_between(
                    rbins,
                    gaussian_diff_lo,
                    gaussian_diff_hi,
                    alpha=0.25,
                    color="purple",
                    label="SBI Pop Gaussian 68% CI",
                )
            ax2.fill_between(
                rbins,
                sbi_jtf_diff_lo,
                sbi_jtf_diff_hi,
                alpha=0.3,
                color="blue",
                label="SBI join-then-fit 68\% CI",
            )
            ax2.fill_between(
                rbins,
                sbi_ftj_diff_lo,
                sbi_ftj_diff_hi,
                alpha=0.3,
                color="green",
                label="SBI join-then-fit 68\% CI",
            )

        ax1.legend(fontsize="large")
        ax2.legend(fontsize="large")

        if is_noisy:
            plt.savefig(os.path.join(out_path, f"sbi_drawn_nfw_profiles.png"))
            plt.savefig(os.path.join(out_path, f"sbi_drawn_nfw_profiles.pdf"))
        else:
            plt.savefig(os.path.join(out_path, f"noiseless_drawn_nfw_profiles.png"))
            plt.savefig(os.path.join(out_path, f"noiseless_drawn_nfw_profiles.pdf"))


def plot_frac_diff(
    nfw_profiles,
    sigmas,
    out_path,
    num_radial_bins,
    min_richness,
    max_richness,
    z,
    mcmc_chains,
    mcmc_jtf_sampler,
    mcmc_ftj_samplers,
    sbi_chains,
    # sbi_ftj_mc,
    # sbi_jtf_mc,
    true_param_median,
    sbi_ftj_logprob,
    sbi_jtf_logprob,
):
    plt.figure(figsize=(8, 5))
    plt.xscale("log")
    plt.xlabel("radius [kpc/h]", fontsize="xx-large")
    plt.title(
        f"Fractional Diff with NFW from Median Drawn M-C $\lambda \in$ [{min_richness}, {max_richness}]",
        fontsize="x-large",
    )
    plt.ylabel(
        r"$\frac{\Delta \Sigma_{model} - \Delta \Sigma_{median}}{\Delta \Sigma_{median}}$",
    )
    nfw_profiles = 10**nfw_profiles

    rbins = 10 ** np.arange(0, num_radial_bins / 10, 0.1)
    ideal_nfw = wlprofile.simulate_nfw(
        float(true_param_median[0]), float(true_param_median[1]), z=z
    )
    # sbi_ftj_nfw = wlprofile.simulate_nfw(
    #     float(sbi_ftj_mc[0]), float(sbi_ftj_mc[1]), z=z
    # )
    # sbi_jtf_nfw = wlprofile.simulate_nfw(
    #     float(sbi_jtf_mc[0]), float(sbi_jtf_mc[1]), z=z
    # )
    sbi_jtf_nfw = inferred_nfw_from_chains(
        sbi_chains[0],
        z,
        method="map",
        log_probs=sbi_jtf_logprob.numpy(),
    )

    sbi_ftj_nfw = inferred_nfw_from_chains(
        sbi_chains[1],
        z,
        method="map",
        log_probs=sbi_ftj_logprob.numpy(),
    )

    mcmc_jtf_nfw = inferred_nfw_from_chains(
        mcmc_chains[0],
        z,
        method="map",
        log_probs=mcmc_jtf_sampler.get_log_prob(flat=True),
    )
    mcmc_ftj_nfw = inferred_nfw_from_chains(
        mcmc_chains[1],
        z,
        method="map",
        log_probs=mcmc_ftj_samplers.get_log_prob(flat=True),
    )

    plt.axhline(
        0, color="gray", linestyle="dotted", alpha=0.5
    )  # , label='Median NFW Profile')
    plt.plot(
        rbins,
        # (sbi_ftj_nfw / np.median(nfw_profiles, axis=0)) - 1,
        (sbi_ftj_nfw / ideal_nfw) - 1,
        color="blue",
        linestyle="--",
        # linewidth=3,
        label="SBI fit-then-join",
    )
    plt.plot(
        rbins,
        # (sbi_jtf_nfw / np.median(nfw_profiles, axis=0)) - 1,
        (sbi_jtf_nfw / ideal_nfw) - 1,
        color="blue",
        # linestyle="-.",
        label="SBI join-then-fit",
    )
    plt.plot(
        rbins,
        # (mcmc_ftj_nfw / np.median(nfw_profiles, axis=0)) - 1,
        (mcmc_ftj_nfw / ideal_nfw) - 1,
        color="green",
        linestyle="--",
        # linewidth=3,
        label="MCMC fit-then-join",
    )
    plt.plot(
        rbins,
        # (mcmc_jtf_nfw / np.median(nfw_profiles, axis=0)) - 1,
        (mcmc_jtf_nfw / ideal_nfw) - 1,
        color="green",
        # linestyle="-.",
        label="MCMC join-then-fit",
    )
    plt.plot(
        rbins,
        # (mcmc_jtf_nfw / np.median(nfw_profiles, axis=0)) - 1,
        (np.median(nfw_profiles, axis=0) / ideal_nfw) - 1,
        color="gray",
        alpha=0.5,
        linestyle="-.",
        label="Median Observed NFW",
    )

    plt.legend(fontsize="large")
    plt.xlim(min(rbins), max(rbins))
    plt.savefig(os.path.join(out_path, f"frac_diff.png"))
    plt.savefig(os.path.join(out_path, f"frac_diff.pdf"))
    # plt.close()


# From chains, let's see what the inferred mc pair is to compare with drawn mc pairs.
def inferred_mc_from_chains(
    chains, method="joint_median", log_probs=None, sbi_posterior=None
):
    split = split_percentile_chain(chains)

    if split:
        levels, mass_percentiles, conc_percentiles = split
        med_idx = median_index(levels)

        if method == "joint_median":
            inferred_log10mass = np.median(mass_percentiles[:, med_idx])
            inferred_concentration = np.median(conc_percentiles[:, med_idx])
            return (inferred_log10mass, inferred_concentration)
        elif method == "map" and log_probs is not None:
            idx = np.argmax(log_probs)
            return (
                mass_percentiles[idx, med_idx],
                conc_percentiles[idx, med_idx],
            )
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

    raise ValueError(
        "Invalid method. Choose either 'joint_median' or 'map' with log_probs."
    )


# From chains, generate an NFW from the inferred m-c pair to compare with drawn NFWs.
def inferred_nfw_from_chains(chains, z, method="joint_median", log_probs=None):
    inferred_mc_pair = inferred_mc_from_chains(chains, method, log_probs)
    return wlprofile.simulate_nfw(inferred_mc_pair[0], inferred_mc_pair[1], z=z)


# Draw a bunch of m-c pairs from the chains and generate NFW profiles
def sampled_nfw_profiles(chains, z, n_samples=200, log_probs=None):
    # Randomly choose indices from chains
    if log_probs is not None:
        # convert to normalized probabilities
        weights = np.exp(log_probs - np.max(log_probs))
        weights /= weights.sum()
        idxs = np.random.choice(len(chains), size=n_samples, replace=True, p=weights)
    else:
        idxs = np.random.choice(len(chains), size=n_samples, replace=False)
    mc_samples = chains[idxs]

    split = split_percentile_chain(chains)
    if split:
        levels, mass_percentiles, conc_percentiles = split
        med_idx = median_index(levels)
        mc_samples = np.column_stack(
            (mass_percentiles[idxs, med_idx], conc_percentiles[idxs, med_idx])
        )
    elif mc_samples.shape[1] > 2:
        mc_samples = mc_samples[:, :2]

    profiles = []
    for log10m, c in mc_samples:
        prof = wlprofile.simulate_nfw(log10m, c, z=z)  # shape: (n_radii,)
        profiles.append(prof)

    return np.array(profiles)  # shape: (n_samples, n_radii)


# Plot the range of drawn m-c pairs
def plot_posterior_band(chains, z, radii, ax=None, color="C0", label=None):
    profiles = sampled_nfw_profiles(chains, z)
    median = np.median(profiles, axis=0)
    lo = np.percentile(profiles, 16, axis=0)
    hi = np.percentile(profiles, 84, axis=0)

    if ax is None:
        fig, ax = plt.subplots()

    # fill between 16th–84th
    ax.fill_between(radii, lo, hi, color=color, alpha=0.3)
    ax.plot(radii, median, color=color, label=label)
    return ax


def plot_ppc(
    drawn_nfw_profiles,
    median_obs_mc_pair,
    posterior,
    out_path,
    num_radial_bins=30,
    z=0,
):
    # SBI is weird so they define the posterior before we actually have observations, so this is, in effect, the prior.
    # Prior here is just the output from the simulations without the observations applied.
    # That makes this next line look a little gross though.
    prior = posterior

    x_o = np.median(drawn_nfw_profiles, axis=0)
    posterior.set_default_x(x_o)

    # 1 - Sample m-c pairs from the posterior and prior
    posterior_samples = posterior.sample((5_000,))
    prior_samples = prior.sample((5_000,))

    # 2 - Generate NFW profiles for those m-c pairs
    rbins = 10 ** np.arange(0, num_radial_bins / 10, 0.1)
    nfw_samples = np.array(
        [
            wlprofile.simulate_nfw(log10mass, concentration, rbins, z)
            for log10mass, concentration in posterior_samples.numpy()
        ]
    )
    prior_nfw_samples = np.array(
        [
            wlprofile.simulate_nfw(log10mass, concentration, rbins, z)
            for log10mass, concentration in prior_samples.numpy()
        ]
    )

    # 3 - Compare with the median observed NFW profile
    plt.loglog()
    plt.xlabel("radius [kpc/h]", fontsize="xx-large")
    plt.ylabel("$\Delta\Sigma$ [$M_\odot h / kpc^2$]", fontsize="xx-large")

    # Setting upper and lower bounds for 1,2,3 sigma
    sig_1_high = np.percentile(nfw_samples, 68.3, axis=0)
    sig_1_low = np.percentile(nfw_samples, 50 - (68.3 - 50), axis=0)
    sig_2_high = np.percentile(nfw_samples, 95.5, axis=0)
    sig_2_low = np.percentile(nfw_samples, 50 - (95.5 - 50), axis=0)
    sig_3_high = np.percentile(nfw_samples, 99.7, axis=0)
    sig_3_low = np.percentile(nfw_samples, 50 - (99.7 - 50), axis=0)

    prior_sig_1_high = np.percentile(prior_nfw_samples, 68.3, axis=0)
    prior_sig_1_low = np.percentile(prior_nfw_samples, 50 - (68.3 - 50), axis=0)
    prior_sig_2_high = np.percentile(prior_nfw_samples, 95.5, axis=0)
    prior_sig_2_low = np.percentile(prior_nfw_samples, 50 - (95.5 - 50), axis=0)
    prior_sig_3_high = np.percentile(prior_nfw_samples, 99.7, axis=0)
    prior_sig_3_low = np.percentile(prior_nfw_samples, 50 - (99.7 - 50), axis=0)

    plt.fill_between(
        rbins, sig_3_low, sig_3_high, color="red", alpha=0.1, label="3$\sigma$"
    )
    plt.fill_between(
        rbins, sig_2_low, sig_2_high, color="yellow", alpha=0.4, label="2$\sigma$"
    )
    plt.fill_between(
        rbins, sig_1_low, sig_1_high, color="green", alpha=0.7, label="1$\sigma$"
    )

    median_obs = np.median(drawn_nfw_profiles, axis=0)[
        num_radial_bins : 2 * num_radial_bins
    ]
    plt.plot(rbins, median_obs, "-", color="blue", label="Median Observed NFW")
    le = []
    ue = []
    for nfw_profile in drawn_nfw_profiles:
        le.append(
            nfw_profile[num_radial_bins : 2 * num_radial_bins]
            - nfw_profile[0:num_radial_bins]
        )
        ue.append(
            nfw_profile[2 * num_radial_bins : 3 * num_radial_bins]
            - nfw_profile[num_radial_bins : 2 * num_radial_bins]
        )
    yerr = [(np.median(le, axis=0)), (np.median(ue, axis=0))]
    plt.errorbar(
        rbins,
        median_obs,
        yerr=yerr,
        fmt=".k",
        capsize=3.0,
    )
    plt.ylim(1e7, 1e10)

    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0]
    plt.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize="large",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    if not os.path.exists(f"{out_path}/PPC"):
        os.makedirs(f"{out_path}/PPC")
    plt.savefig(f"{out_path}/PPC/nfw.pdf", bbox_inches="tight")
    plt.close()

    # Doing the same thing for the prior
    plt.loglog()
    plt.xlabel("radius [kpc/h]", fontsize="xx-large")
    plt.ylabel("$\Delta\Sigma$ [$M_\odot h / kpc^2$]", fontsize="xx-large")
    plt.fill_between(
        rbins,
        prior_sig_3_low,
        prior_sig_3_high,
        color="red",
        alpha=0.1,
        label="3$\sigma$",
    )
    plt.fill_between(
        rbins,
        prior_sig_2_low,
        prior_sig_2_high,
        color="yellow",
        alpha=0.4,
        label="2$\sigma$",
    )
    plt.fill_between(
        rbins,
        prior_sig_1_low,
        prior_sig_1_high,
        color="green",
        alpha=0.7,
        label="1$\sigma$",
    )

    median_obs = np.median(drawn_nfw_profiles, axis=0)[
        num_radial_bins : 2 * num_radial_bins
    ]
    upper_error = np.median(drawn_nfw_profiles, axis=0)[
        2 * num_radial_bins : 3 * num_radial_bins
    ]
    lower_error = np.median(drawn_nfw_profiles, axis=0)[0:num_radial_bins]
    plt.plot(rbins, median_obs, "-", color="blue", label="Median Observed NFW")
    plt.errorbar(
        rbins,
        median_obs,
        yerr=yerr,
        fmt=".k",
        capsize=3.0,
    )
    plt.ylim(1e7, 1e10)

    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0]
    plt.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize="large",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.savefig(f"{out_path}/PPC/prior_nfw.pdf", bbox_inches="tight")
    plt.close()

    # 4 - For a given radial bin, compare the delta sigma values from observation vs from the generated values
    nfw_samples_norm = nfw_samples - np.median(nfw_samples, axis=0)
    sigmas_diff = []
    upper_sigmas_diff = []
    lower_sigmas_diff = []
    for rbin in range(num_radial_bins):
        # plt.figure()
        sns.kdeplot(nfw_samples[:, rbin], label="NFW Profiles Sampled from Posterior")
        plt.xlabel("$\Delta\Sigma$ [$M_\odot h / kpc^2$]")
        plt.ylabel("Frequency")
        plt.axvline(median_obs[rbin], color="k", linestyle="--", label="Observation")
        plt.axvspan(
            lower_error[rbin],
            upper_error[rbin],
            alpha=0.3,
            color="gray",
            label="Observation Uncertainty",
        )
        plt.legend(fontsize="large", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"Radial Bin: {rbin}")
        plt.savefig(f"{out_path}/PPC/rbin_{rbin}.pdf", bbox_inches="tight")
        plt.close()
        sigmas_diff.append(
            (median_obs[rbin] - np.median(nfw_samples[:, rbin]))
            / np.std(nfw_samples[:, rbin])
        )
        upper_sigmas_diff.append(
            (upper_error[rbin] - np.median(nfw_samples[:, rbin]))
            / np.std(nfw_samples[:, rbin])
        )
        lower_sigmas_diff.append(
            (lower_error[rbin] - np.median(nfw_samples[:, rbin]))
            / np.std(nfw_samples[:, rbin])
        )

    for rbin in range(num_radial_bins):
        upper_sigmas_diff[rbin] = upper_sigmas_diff[rbin] - sigmas_diff[rbin]
        lower_sigmas_diff[rbin] = sigmas_diff[rbin] - lower_sigmas_diff[rbin]

    # 5 - Plot the inference error (# sigmas) per radial bin
    plt.errorbar(
        rbins,
        sigmas_diff,
        yerr=[lower_sigmas_diff, upper_sigmas_diff],
        fmt="blue",
        ecolor="k",
        capsize=3.0,
    )
    # plt.plot(rbins, sigmas_diff)
    plt.xscale("log")
    plt.axhline(np.median(sigmas_diff), color="gray", linestyle="--")
    plt.xlabel("radius [kpc/h]", fontsize="xx-large")
    plt.ylabel(r"Inference error: $\sigma$s from observation")
    plt.savefig(f"{out_path}/PPC/error_per_radial_bin.pdf", bbox_inches="tight")
