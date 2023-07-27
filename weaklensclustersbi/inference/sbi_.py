import numpy as np
import sbi
import sbi.utils as utils
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot
import torch
from torch import zeros, ones


def sbi_config():
    pass


# Create an inferrer with the priors from the inference config
def gen_inferrer(priors):
    prior = BoxUniform(
        torch.as_tensor([priors['min_log10mass'],
                         priors['min_concentration']]),
        torch.as_tensor([priors['max_log10mass'],
                         priors['max_concentration']]))
    return SNPE(prior, density_estimator="mdn",
                device="cpu")  # SNLE, SNRE are other options


# Train the inferrer with the simulations. We'll pickle this posterior for future use.
# In a later step, we'll add observations to this (un)pickled posterior and then sample from that.
def gen_posterior(inferrer, sample_mc_pairs, simulated_nfw_profiles):
    # Define our data in terms of parameters, theta, and data
    theta_np = np.array(sample_mc_pairs.T).T
    x_np = simulated_nfw_profiles

    # turn into tensors
    theta = torch.as_tensor(theta_np, dtype=torch.float32)
    x = torch.as_tensor(x_np, dtype=torch.float32)

    # Append training data
    inferrer = inferrer.append_simulations(theta, x)

    # Train  (note: Lots of training settings.)
    density_estimator = inferrer.train(
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
    posterior = inferrer.build_posterior(density_estimator)

    return posterior


# Apply observations to the (un)pickled posterior and sample from the posterior
def apply_observations(posterior, drawn_mc_pairs, drawn_nfw_profiles):
    from .sbiutils import create_fit_join_observation_nfw, create_join_fit_observation_nfw

    # Create an observation
    theta_o_fj, x_o_fj = create_fit_join_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles)
    theta_o_jf, x_o_jf = create_join_fit_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles)

    # Obtain samples of the posterior given the observation
    samples_fj = posterior.sample((10000, ), x=x_o_fj)
    samples_jf = posterior.sample((10000, ), x=x_o_jf)

    return [samples_fj.numpy(), np.vstack(samples_jf.numpy())]
