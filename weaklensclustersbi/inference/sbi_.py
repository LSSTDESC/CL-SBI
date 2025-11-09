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
def gen_inferrer(priors, param_dim=2):
    if param_dim < 2 or param_dim % 2 not in (0, 1):
        raise ValueError("Unexpected parameter dimensionality: expected percentiles plus optional correlation.")

    has_correlation = param_dim % 2 == 1
    n_levels = (param_dim - int(has_correlation)) // 2
    lower = [priors["min_log10mass"]] * n_levels + [priors["min_concentration"]] * n_levels
    upper = [priors["max_log10mass"]] * n_levels + [priors["max_concentration"]] * n_levels

    if has_correlation:
        lower.append(-1.0)
        upper.append(1.0)

    prior = BoxUniform(
        torch.as_tensor(lower, dtype=torch.float32),
        torch.as_tensor(upper, dtype=torch.float32),
    )
    return SNPE(
        prior, density_estimator="mdn", device="cpu"
    )  # SNLE, SNRE are other options


# Train the inferrer with the simulations. We'll pickle this posterior for future use.
# In a later step, we'll add observations to this (un)pickled posterior and then sample from that.
def gen_posterior(inferrer, sample_mc_pairs, simulated_nfw_profiles):
    # Define our data in terms of parameters, theta, and data
    theta_np = np.array(sample_mc_pairs.T).T
    # x_np = simulated_nfw_profiles
    x_np = simulated_nfw_profiles.reshape(simulated_nfw_profiles.shape[0], -1)

    # print(np.shape(theta_np))
    # print(np.shape(x_np))

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
    # posterior = inferrer.build_posterior(density_estimator, sample_with="mcmc")
    posterior = inferrer.build_posterior(density_estimator)

    return posterior


# Apply observations to the (un)pickled posterior and sample from the posterior
def apply_observations(
    posterior, posterior_jtf, drawn_mc_pairs, drawn_nfw_profiles, err_dex=0.0
):
    from .sbiutils import (
        create_join_fit_observation_nfw,
        create_fit_join_observation_nfw,
    )

    # Join (take the median of) observations and then fit on that
    theta_o_jf, x_o_jf = create_join_fit_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles
    )

    # Obtain samples of the posterior given the observation
    samples_jf = posterior_jtf.sample((10000,), x=x_o_jf)

    # Calculate the log-probability of the samples given the observation to find the maximum a posteriori (MAP) estimate
    logp_jf = posterior_jtf.log_prob(samples_jf, x=x_o_jf)
    idx_jf = torch.argmax(logp_jf)
    map_mc_jf = samples_jf[idx_jf]

    # Fit each observation and join them (stack the chains) at the end
    theta_o_fj, x_o_fj = create_fit_join_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles
    )

    samples_fj = posterior.sample((10000,), x=x_o_fj)
    logp_fj = posterior.log_prob(samples_fj, x=x_o_fj)
    idx_fj = torch.argmax(logp_fj)
    map_mc_fj = samples_fj[idx_fj]

    return (
        samples_jf.numpy(),
        samples_fj.numpy(),
        map_mc_jf.numpy(),
        map_mc_fj.numpy(),
        logp_fj,
        logp_jf,
    )
