import pygtc
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import os
from colossus.cosmology import cosmology

param_labels = ['log10mass', 'concentration']
chain_labels = ['join_then_fit', 'fit_then_join']
wide_param_ranges = ((12.5, 15.5), (3.5, 8.5))
narrow_param_ranges = ((14.2, 15.2), (4.2, 6.2))


def timestamp():
    import time
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    return timestr


def plot_pygtc(chains, out_path, infer_type, true_param_mean=()):
    # posterFont = {'family': 'Arial', 'size': 18}
    GTC = pygtc.plotGTC(
        chains=chains,
        chainLabels=chain_labels,
        paramNames=param_labels,
        figureSize=8.,
        paramRanges=wide_param_ranges,
        sigmaContourLevels=True,
        plotDensity=True,
        truths=true_param_mean,
        # customLabelFont=posterFont,
        # customTickFont=posterFont,
        # customLegendFont=posterFont,
        nContourLevels=2,
    )

    GTC.savefig(os.path.join(out_path, f'{infer_type}_gtc.png'))
    GTC.savefig(os.path.join(out_path, f'{infer_type}_gtc.pdf'))


def plot_chainconsumer(chains, out_path, infer_type, true_param_mean=[]):
    cc = ChainConsumer()
    cc.add_chain(chains[0], parameters=param_labels, name=chain_labels[0])
    cc.add_chain(chains[1], parameters=param_labels, name=chain_labels[1])
    cc.configure(
        statistics='max',
        summary=True,
        label_font_size=20,
        tick_font_size=16,
        usetex=False,
        serif=False,
        sigmas=[0, 1, 2],
    )
    fig = cc.plotter.plot(
        truth=true_param_mean,
        parameters=param_labels,
        # extents=list(wide_param_ranges),
        figsize=(8, 8))
    ax_list = fig.axes
    for ax in ax_list:
        ax.grid(False)

    plt.savefig(os.path.join(out_path, f'{infer_type}_cc.png'))
    plt.savefig(os.path.join(out_path, f'{infer_type}_cc.pdf'))
    plt.close()


### DIAGNOSTICS PLOTTING BELOW ###


def plot_walkers(sampler, out_path, prefix=''):
    ndim = 2
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ['log10mass', 'concentration']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(os.path.join(out_path, f'{prefix}mcmc_walkers.png'))
    # plt.savefig(os.path.join(out_path, f'{prefix}mcmc_walkers.pdf'))
    plt.close()


# Overplotting multiple chains (for each observation)
def plot_cc_diagnostic(chains, out_path, infer_type, true_param_mean=[]):
    cc = ChainConsumer()
    i = 0
    for chain in chains:
        cc.add_chain(chain, parameters=param_labels, name=f'chain_{i}')
        i += 1
    cc.configure(statistics='max',
                 summary=True,
                 label_font_size=20,
                 tick_font_size=16,
                 usetex=False,
                 serif=False,
                 sigmas=[0, 1, 2])
    fig = cc.plotter.plot(
        truth=true_param_mean,
        parameters=param_labels,
        # extents=list(wide_param_ranges),
        figsize=(8, 8))
    ax_list = fig.axes
    for ax in ax_list:
        ax.grid(False)

    plt.savefig(os.path.join(out_path, f'{infer_type}_cc.png'))
    # plt.savefig(os.path.join(out_path, f'{infer_type}_cc.pdf'))
    plt.close()


def plot_mc_pairs(mc_pairs, out_path):
    plt.scatter(mc_pairs[:, 0], mc_pairs[:, 1], s=50)
    plt.xlabel('log$_{10}$M [M$_\odot$]', fontsize='xx-large')
    plt.ylabel('Concentration', fontsize='xx-large')
    plt.title('Drawn mc_pairs')
    plt.savefig(os.path.join(out_path, f'drawn_mc_pairs.png'))
    # plt.savefig(os.path.join(out_path, f'drawn_mc_pairs.pdf'))
    plt.close()


def plot_nfw_profiles(nfw_profiles, out_path, num_radial_bins, min_richness,
                      max_richness):
    cosmo = cosmology.setCosmology('planck18')

    rbins = 10**np.arange(0, num_radial_bins / 10, 0.1)
    plt.figure()
    plt.loglog()
    plt.xlabel('radius [kpc/h]', fontsize='xx-large')
    plt.ylabel('Density profile [$\\rho/\\rho_m$]', fontsize='xx-large')
    for nfw_profile in nfw_profiles:
        plt.plot(rbins, nfw_profile / cosmo.rho_m(0), '-', alpha=0.3)
    plt.plot(
        rbins,
        np.mean(nfw_profiles, axis=0) / cosmo.rho_m(0),
        '-',
        label=f'Mean Drawn NFW, {min_richness} < $\lambda$ < {max_richness}')
    plt.legend(fontsize='large')
    plt.savefig(os.path.join(out_path, f'drawn_nfw_profiles.png'))
    # plt.savefig(os.path.join(out_path, f'drawn_nfw_profiles.pdf'))
    plt.close()
