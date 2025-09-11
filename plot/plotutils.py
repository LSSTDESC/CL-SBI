import pygtc
import numpy as np
import matplotlib.pyplot as plt
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
narrow_param_ranges = ((14.2, 15.2), (4.2, 6.2))


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


def plot_chainconsumer(chains, out_path, infer_type, true_param=[]):
    # In MCMC we also fit for the error. We don't need to plot that so pruning that param from the data
    # TODO: clean this up
    if np.shape(chains[0])[1] > len(param_labels):
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
        # sigmas=[2, 3],
        sigmas=[1, 2],
        shade_alpha=0.3,
    )

    # Manually plot the truth values
    p25 = np.array(true_param[1])  # 25th percentile
    p50 = np.array(true_param[0])  # median
    p75 = np.array(true_param[2])  # 75th percentile

    # Set the median as the truth value
    fig = cc.plotter.plot(
        truth=p50,
        parameters=param_labels,
        extents=list(wide_param_ranges),
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

    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.png"))
    plt.savefig(os.path.join(out_path, f"{infer_type}_cc.pdf"))
    plt.close()


def plot_chainconsumer_combined(mcmc_chains, sbi_chains, out_path, true_param=[]):
    cc = ChainConsumer()

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
        extents=list(wide_param_ranges),
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


def plot_chainconsumer_agg(chain, out_path, true_params=None):
    """
    Generate two contour plots from an aggregate SBI posterior:
    1) All individual cluster posteriors overlaid in one plot
    2) One combined posterior from pooling all samples

    Args:
        chain: ndarray of shape (n_samples, 2 * N_obs)
        out_path: directory path to save plots
        true_params: optional list of N_obs (m, c) tuples for ground truth
    """
    chain = np.asarray(chain)
    nsamp, ndim = chain.shape
    if ndim % 2:
        raise ValueError(f"Expected even dims (2*N_obs), got {ndim}")
    N_obs = ndim // 2

    # labels for each (m,c)
    unit_labels = ["log10 M", "c"]

    truth_flat = None
    if true_params is not None:
        truth_flat = np.array([val for pair in true_params for val in pair])

    # 1) Individual overlay plot
    cc_ind = ChainConsumer()
    for i in range(N_obs):
        sub = chain[:, 2 * i : 2 * i + 2]
        cc_ind.add_chain(sub, parameters=unit_labels, name=f"cluster_{i}")
    cc_ind.configure(statistics="mean", summary=False, sigmas=[1], shade_alpha=0.2)
    fig1 = cc_ind.plotter.plot(
        truth=truth_flat,
        figsize=(6, 6),
        parameters=unit_labels,
        extents=list(wide_param_ranges),
    )
    os.makedirs(out_path, exist_ok=True)
    fig1.savefig(os.path.join(out_path, "sbi_agg_individual.png"))
    fig1.savefig(os.path.join(out_path, "sbi_agg_individual.pdf"))
    plt.close(fig1)

    # 2) Combined-chain plot
    pooled = np.vstack([chain[:, 2 * i : 2 * i + 2] for i in range(N_obs)])
    cc_comb = ChainConsumer()
    cc_comb.add_chain(pooled, parameters=unit_labels, name="combined")
    cc_comb.configure(statistics="mean", summary=True, sigmas=[1, 2], shade_alpha=0.3)
    fig2 = cc_comb.plotter.plot(
        truth=truth_flat,
        parameters=unit_labels,
        figsize=(6, 6),
        extents=list(wide_param_ranges),
    )
    fig2.savefig(os.path.join(out_path, "sbi_agg_combined.png"))
    fig2.savefig(os.path.join(out_path, "sbi_agg_combined.pdf"))
    plt.close(fig2)


def plot_chainconsumer_agg2(chains, out_path, true_params):
    """
    chains: numpy array of shape (N,6)
    true_params: list/tuple of three (mass,conc) pairs:
       [ (m50,c50), (m25,c25), (m75,c75) ]  or however you have it stored
    """
    # Grab each percentile chain
    chain_25 = chains[:, 0:2]
    chain_50 = chains[:, 2:4]
    chain_75 = chains[:, 4:6]

    cc = ChainConsumer()
    cc.add_chain(chain_50, parameters=param_labels, name="median (50%)")
    cc.add_chain(chain_25, parameters=param_labels, name="25th percentile")
    cc.add_chain(chain_75, parameters=param_labels, name="75th percentile")

    cc.configure(
        statistics="mean",
        summary=True,
        sigmas=[1, 2],
        shade_alpha=0.3,
    )

    # true_params should be [(m50,c50), (m25,c25), (m75,c75)]
    # we pass the median as 'truth'
    truth = np.array(true_params[0])

    # DEBUG
    # 1) Verify shapes make sense
    print("chains shape:", chains.shape)
    chain_25, chain_50, chain_75 = chains[:, :2], chains[:, 2:4], chains[:, 4:6]
    print("chain_50 shape:", chain_50.shape)

    # 2) Check for NaN / Inf
    for name, ch in [("25th", chain_25), ("50th", chain_50), ("75th", chain_75)]:
        print(f"{name} has NaN/Inf?", not np.isfinite(ch).all())

    # 3) Check for zero width
    for name, ch in [("25th", chain_25), ("50th", chain_50), ("75th", chain_75)]:
        mins, maxs = ch.min(axis=0), ch.max(axis=0)
        print(
            f"{name} mins/maxs:",
            mins,
            "/",
            maxs,
            "  zero‐width?",
            np.allclose(mins, maxs),
        )

    fig = cc.plotter.plot(
        truth=truth,
        parameters=param_labels,
        figsize=(6, 6),
        extents=list(wide_param_ranges),
    )

    fig.savefig(os.path.join(out_path, "sbi_agg_cc.png"))
    fig.savefig(os.path.join(out_path, "sbi_agg_cc.pdf"))


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
        if np.shape(chain)[1] > len(param_labels):
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
    # sbi_posterior=None,
    sbi_ftj_mcs=None,
    sbi_jtf_mc=None,
):
    cosmo = cosmology.setCosmology("planck18")

    nfw_profiles = 10**nfw_profiles  # convert back from log space

    rbins = 10 ** np.arange(0, num_radial_bins / 10, 0.1)
    plt.figure(figsize=(8, 8))
    plt.loglog()
    plt.xlabel("radius [kpc/h]", fontsize="xx-large")
    plt.ylabel("$ \Delta \Sigma$ [$M_\odot h / kpc^2$]", fontsize="xx-large")
    plt.ylim(1e7, 1e10)
    plt.xlim(min(rbins), max(rbins))
    # yerr = sigmas

    for nfw_profile in nfw_profiles:
        plt.plot(rbins, nfw_profile, "-", alpha=0.1, zorder=0, color="gray")
    plt.plot(
        rbins,
        np.median(nfw_profiles, axis=0),
        "k",
        linewidth=2.0,
        label=f"Median Drawn NFW, {min_richness} < $\lambda$ < {max_richness}",
    )
    if is_noisy:
        upper_error = np.exp(
            np.log(np.median(nfw_profiles, axis=0)) + sigmas
        ) - np.median(nfw_profiles, axis=0)
        lower_error = np.median(nfw_profiles, axis=0) - np.exp(
            np.log(np.median(nfw_profiles, axis=0)) - sigmas
        )
        yerr = [lower_error, upper_error]

        plt.errorbar(
            rbins,
            np.median(nfw_profiles, axis=0),
            yerr=yerr,
            fmt="k",
            capsize=3.0,
            linewidth=4,
            elinewidth=1,
            zorder=1,
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
            # np.concatenate([s.get_log_prob(flat=True) for s in mcmc_ftj_samplers]),
        )
        plt.plot(
            rbins,
            mcmc_jtf_nfw,
            # linestyle='dashed',
            # linewidth=2.0,
            label=f"MCMC join-then-fit",
            color="g",
        )
        plt.plot(
            rbins,
            mcmc_ftj_nfw,
            linestyle="dashed",
            # linewidth=2.0,
            label=f"MCMC fit-then-join",
            color="g",
        )
        plt.ylim(
            min(min(mcmc_jtf_nfw), min(mcmc_ftj_nfw)) * 0.8,
            max(max(mcmc_jtf_nfw), max(mcmc_ftj_nfw)) * 1.2,
        )
    if sbi_chains:
        if sbi_ftj_mcs is not None:
            sbi_ftj_mc = np.median(sbi_ftj_mcs, axis=0)
            sbi_ftj_nfw = wlprofile.simulate_nfw(sbi_ftj_mc[0], sbi_ftj_mc[1], z=z)
            plt.plot(
                rbins,
                sbi_ftj_nfw,
                linestyle="dashed",
                label=f"SBI fit-then-join",
                color="b",
            )
        if sbi_jtf_mc is not None:
            sbi_jtf_nfw = wlprofile.simulate_nfw(
                float(sbi_jtf_mc[0]), float(sbi_jtf_mc[1]), z=z
            )
            plt.plot(
                rbins,
                sbi_jtf_nfw,
                label=f"SBI join-then-fit",
                color="b",
            )
        plt.ylim(
            min(
                min(sbi_jtf_nfw), min(sbi_ftj_nfw), min(np.median(nfw_profiles, axis=0))
            )
            * 0.8,
            max(
                max(sbi_jtf_nfw), max(sbi_ftj_nfw), max(np.median(nfw_profiles, axis=0))
            )
            * 1.2,
        )

    if sbi_chains and mcmc_chains:
        plt.ylim(
            min(
                min(min(sbi_jtf_nfw), min(sbi_ftj_nfw)),
                min(min(mcmc_jtf_nfw), min(mcmc_ftj_nfw)),
                min(np.median(nfw_profiles, axis=0)),
            )
            * 0.8,
            max(
                max(max(sbi_jtf_nfw), max(sbi_ftj_nfw)),
                max(max(mcmc_jtf_nfw), max(mcmc_ftj_nfw)),
                max(np.median(nfw_profiles, axis=0)),
            )
            * 1.2,
        )

    # Rescale the axes
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()
    plt.legend(fontsize="large")
    if is_noisy:
        plt.savefig(os.path.join(out_path, f"drawn_nfw_profiles.png"))
        plt.savefig(os.path.join(out_path, f"drawn_nfw_profiles.pdf"))
    else:
        plt.savefig(os.path.join(out_path, f"noiseless_drawn_nfw_profiles.png"))
        plt.savefig(os.path.join(out_path, f"noiseless_drawn_nfw_profiles.pdf"))
    plt.close()


# From chains, let's see what the inferred mc pair is to compare with drawn mc pairs.
def inferred_mc_from_chains(
    chains, method="joint_median", log_probs=None, sbi_posterior=None
):
    if method == "joint_median":
        inferred_log10mass = np.median(chains[:, 0])
        inferred_concentration = np.median(chains[:, 1])
        return (inferred_log10mass, inferred_concentration)
    elif method == "map" and log_probs is not None:
        idx = np.argmax(log_probs)
        return tuple(chains[idx])
    else:
        raise ValueError(
            "Invalid method. Choose either 'joint_median' or 'map' with log_probs."
        )


# From chains, generate an NFW from the inferred m-c pair to compare with drawn NFWs.
def inferred_nfw_from_chains(
    chains, z, method="joint_median", log_probs=None, sbi_posterior=None
):
    inferred_mc_pair = inferred_mc_from_chains(chains, method, log_probs, sbi_posterior)
    return wlprofile.simulate_nfw(inferred_mc_pair[0], inferred_mc_pair[1], z=z)


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
