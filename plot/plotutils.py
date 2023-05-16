import pygtc
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

param_labels = ['log10mass', 'concentration']
chain_labels = ['join_then_fit', 'fit_then_join']
wide_param_ranges = ((12.5, 15.5), (3.5, 8.5))
narrow_param_ranges = ((14.2, 15.2), (4.2, 6.2))


def timestamp():
    import time
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    return timestr


def plot_walkers(sampler):
    chain = sampler.get_chain()

    # TODO: don't hardcode this
    npar = 2

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
    plt.savefig(f'mcmc_walkers.png')


def plot_pygtc(chains, output_filename, truths2d=()):
    # posterFont = {'family': 'Arial', 'size': 18}
    GTC = pygtc.plotGTC(
        chains=chains,
        chainLabels=chain_labels,
        paramNames=param_labels,
        figureSize=8.,
        paramRanges=wide_param_ranges,
        sigmaContourLevels=True,
        plotDensity=True,
        truths=truths2d,
        # customLabelFont=posterFont,
        # customTickFont=posterFont,
        # customLegendFont=posterFont,
        nContourLevels=2,
    )
    GTC.savefig(f'images/{output_filename}_{timestamp()}.png')


def plot_chainconsumer(chains, output_filename, truths2d=[]):
    cc = ChainConsumer()
    cc.add_chain(chains[0], parameters=param_labels, name=chain_labels[0])
    cc.add_chain(chains[1], parameters=param_labels, name=chain_labels[1])
    cc.configure(statistics='max',
                 summary=True,
                 label_font_size=20,
                 tick_font_size=16,
                 usetex=False,
                 serif=False,
                 sigmas=[0, 1, 2])
    fig = cc.plotter.plot(truth=truths2d,
                          parameters=param_labels,
                          extents=list(wide_param_ranges),
                          figsize=(8, 8))
    ax_list = fig.axes
    for ax in ax_list:
        ax.grid(False)

    plt.savefig(f'images/{output_filename}_{timestamp()}.png')
