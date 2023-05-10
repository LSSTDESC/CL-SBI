import pygtc
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

npar = 2  # TODO: don't hardcode this
param_labels = ['log10mass', 'concentration']


def plot_chainconsumer(sampler, truth_values=()):
    samples = sampler.get_chain(flat=True)

    plt.clf()
    c = ChainConsumer()
    c.add_chain(samples, parameters=param_labels)
    c.configure(
        statistics='max',
        summary=True,
        label_font_size=20,
        tick_font_size=16,
        usetex=False,
        serif=False,
    )
    fig = c.plotter.plot(
        # TODO: fix truth values and uncomment below line
        # truth=truth_values,
        parameters=['log10mass', 'concentration'],
        extents=[(13.5, 14.5), (5, 7)],
        figsize=(3 * npar, 3 * npar))
    ax_list = fig.axes
    for ax in ax_list:
        ax.grid(False)

    plt.show()
    print('hi')


def plot_walkers(sampler):
    chain = sampler.get_chain()

    fig, axes = plt.subplots(npar, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i in range(npar):
        ax = axes[i]
        # plot the chain
        ax.plot(chain[:, :, i])
        xmax = len(chain[:, i])
        # truths
        ax.plot([0 - 100, xmax + 100], [truths[i], truths[i]],
                'k-.',
                lw=2,
                label='truth')
        # starts
        ax.plot([0], [starts[:, i]], 'x', color='#d62728')

        ax.set_ylabel(r'%s' % param_labels[i])

    axes[0].legend(bbox_to_anchor=(1, 1))
    axes[-1].set_xlabel('# of steps')

    fig.set_size_inches(10, 6 * npar / 3)


def combine_chains_pygtc(join_then_fit_chain, chains, truth=()):
    chain_labels = ['join_then_fit', 'fit_then_join']
    wide_param_ranges = ((13, 15.5), (3, 7.5))
    narrow_param_ranges = ((14.2, 15.2), (4.2, 6.2))
    posterFont = {'family': 'Arial', 'size': 18}

    GTC = pygtc.plotGTC(
        chains=[join_then_fit_chain, np.vstack(chains)],
        chainLabels=chain_labels,
        paramNames=param_labels,
        figureSize=8.,
        # paramRanges=wide_param_ranges,
        sigmaContourLevels=True,
        plotDensity=True,
        # truths=truth,  # these aren't actually truth values, but a good way of visualizing the richness bin
        customLabelFont=posterFont,
        customTickFont=posterFont,
        customLegendFont=posterFont,
        nContourLevels=3,
    )


def combine_chains_chainconsumer(join_then_fit_chain, chains, truths=[]):
    cc = ChainConsumer()
    cc.configure(statistics='max',
                 summary=True,
                 label_font_size=20,
                 tick_font_size=16,
                 usetex=False,
                 serif=False,
                 sigmas=[1, 2])
    cc.add_chain(join_then_fit_chain,
                 parameters=param_labels,
                 name='join_then_fit')
    cc.add_chain(np.vstack(chains),
                 parameters=param_labels,
                 name='fit_then_join')
    fig = cc.plotter.plot(
        # truth=truths,
        parameters=param_labels,
        extents=[(13.5, 15.5), (2.5, 8)],
        figsize=(3 * npar, 3 * npar))
    ax_list = fig.axes
    for ax in ax_list:
        ax.grid(False)

    plt.show()
