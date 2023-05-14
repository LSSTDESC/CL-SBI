import numpy as np
import sbi
import sbi.utils as utils
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot
import torch
from torch import zeros, ones
import pygtc


def sbi_config():
    pass


def run_sbi(simulated_nfw_profiles,
            log10mass_sample,
            concentration_sample,
            drawn_nfw_profiles,
            drawn_log10masses,
            drawn_concentrations,
            noise_dex=0.0):
    from .sbiutils import calculate_noise, create_fit_join_observation_nfw, create_join_fit_observation_nfw

    # Define our data in terms of parameters, theta, and data
    theta_np = np.array([log10mass_sample, concentration_sample]).T
    x_np = calculate_noise(simulated_nfw_profiles, noise_dex)

    # Uniform prior from -20 to 20 (which encompasses possible log10 masses and concentration values)
    num_priors = np.shape(theta_np)[1]
    # TODO: these priors are arbitrary - can we tighten these?
    prior = BoxUniform(-ones(num_priors) * 20, ones(num_priors) * 20)

    # turn into tensors
    theta = torch.as_tensor(theta_np, dtype=torch.float32)
    x = torch.as_tensor(x_np, dtype=torch.float32)

    # Create inference object: choose method and estimator
    inferer = SNPE(prior, density_estimator="mdn",
                   device="cpu")  # SNLE, SNRE are other options

    # Append training data
    inferer = inferer.append_simulations(theta, x)

    # Train  (note: Lots of training settings.)
    density_estimator = inferer.train(
        num_atoms=4,
        training_batch_size=50,
        learning_rate=0.0005,
        validation_fraction=0.1,
        stop_after_epochs=20,
        max_num_epochs=500,
        clip_max_norm=5.0,
        calibration_kernel=None,
        resume_training=False,
        discard_prior_samples=False,
        use_combined_loss=False,
        show_train_summary=False,
        dataloader_kwargs=None,
    )

    # Build posterior using trained density estimator and posterior sampling settings
    posterior = inferer.build_posterior(density_estimator)

    mc_pairs = list(zip(drawn_log10masses, drawn_concentrations))

    # Create an observation
    theta_o_fj, x_o_fj = create_fit_join_observation_nfw(
        mc_pairs, drawn_nfw_profiles, noise_dex)
    theta_o_jf, x_o_jf = create_join_fit_observation_nfw(mc_pairs, noise_dex)

    # Obtain samples of the posterior given the observation
    samples_fj = posterior.sample((10000, ), x=x_o_fj)
    samples_jf = posterior.sample((10000, ), x=x_o_jf)

    # Redefine priors and truths
    priors2d = ((-20, 20), (20, 20))
    truths2d = (np.mean(log10mass_sample), np.mean(concentration_sample))

    wide_param_ranges = ((13, 15.5), (3, 7.5))
    narrow_param_ranges = ((14.2, 15.2), (4.2, 6.2))
    # param_ranges = ((14.5, 14.75), (5, 5.5))
    param_labels = ['log10mass', 'concentration']
    chain_labels = ['join_then_fit', 'fit_then_join']

    # TODO: split plotting into separate functions and probs a separate file
    # The 2d panel and the 1d histograms
    GTC = pygtc.plotGTC(
        chains=[samples_fj.numpy(), samples_jf.numpy()],
        chainLabels=chain_labels,
        paramNames=param_labels,
        truths=truths2d,
        priors=priors2d,
        # paramRanges=wide_param_ranges,
        #figureSize='MNRAS_column',
        figureSize=8.,
        nContourLevels=3,
    )
    #figureSize='MNRAS_column')
    GTC.savefig('sbi_with_histogram.png')

    GTC = pygtc.plotGTC(
        chains=[samples_fj.numpy(), samples_jf.numpy()],
        chainLabels=chain_labels,
        paramNames=param_labels,
        truths=truths2d,
        priors=priors2d,
        # paramRanges=narrow_param_ranges,
        #figureSize='MNRAS_column',
        figureSize=8.,
        do1dPlots=False,
        customTickFont={
            'family': 'Arial',
            'size': 12
        })
    GTC.savefig('sbi_no_histogram.png')