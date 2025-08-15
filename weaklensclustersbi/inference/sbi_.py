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
        torch.as_tensor([priors["min_log10mass"], priors["min_concentration"]]),
        torch.as_tensor([priors["max_log10mass"], priors["max_concentration"]]),
    )
    return SNPE(
        prior, density_estimator="mdn", device="cpu"
    )  # SNLE, SNRE are other options


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
def apply_observations(posterior, drawn_mc_pairs, drawn_nfw_profiles, err_dex=0.0):
    from .sbiutils import create_observation_nfw, create_join_fit_observation_nfw

    # Join (take the median of) observations and then fit on that
    theta_o_jf, x_o_jf = create_join_fit_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles
    )

    # Obtain samples of the posterior given the observation
    samples_jf = posterior.sample((10000,), x=x_o_jf)

    # Calculate the log-probability of the samples given the observation to find the maximum a posteriori (MAP) estimate
    logp_jf = posterior.log_prob(samples_jf, x=x_o_jf)
    idx_jf = np.argmax(logp_jf)
    map_mc_jf = samples_jf[idx_jf]

    # Fit each observation and join them (stack the chains) at the end
    samples_fj = []
    map_mc_fj = []
    for i in range(len(drawn_nfw_profiles)):
        theta_o_fj, x_o_fj = create_observation_nfw(
            drawn_mc_pairs[i], drawn_nfw_profiles[i]
        )
        s = posterior.sample((10000,), x=x_o_fj).numpy()
        lp = posterior.log_prob(s, x=x_o_fj)
        idx = np.argmax(lp)
        map_mc_fj.append(s[idx])
        samples_fj.append(s)

    return [samples_jf.numpy(), np.vstack(samples_fj)], samples_fj, map_mc_fj, map_mc_jf


# AGGREGATE SBI INFERENCE


def gen_agg_inferrer(priors, num_obs):
    # Infer 25th, 50th, and 75th percentile m-c pairs for distribution
    # low = torch.tensor(
    #     [
    #         priors["min_log10mass"],
    #         priors["min_concentration"],
    #         priors["min_log10mass"],
    #         priors["min_concentration"],
    #         priors["min_log10mass"],
    #         priors["min_concentration"],
    #     ]
    # )
    # high = torch.tensor(
    #     [
    #         priors["max_log10mass"],
    #         priors["max_concentration"],
    #         priors["max_log10mass"],
    #         priors["max_concentration"],
    #         priors["max_log10mass"],
    #         priors["max_concentration"],
    #     ]
    # )

    # Infer m-c pair for each observable
    mass_low, conc_low = priors["min_log10mass"], priors["min_concentration"]
    mass_high, conc_high = priors["max_log10mass"], priors["max_concentration"]

    # build two vectors of length 2 * num_obs:
    low = torch.tensor([mass_low, conc_low] * num_obs, dtype=torch.float32)
    high = torch.tensor([mass_high, conc_high] * num_obs, dtype=torch.float32)

    agg_prior = BoxUniform(low, high)
    return SNPE(agg_prior, density_estimator="mdn", device="cpu")


# Generate a data vector with num_sims x num_obs observations and num_sims x num_obs m-c pairs
def gen_agg_posterior(inferrer, sample_mc_pairs, simulated_nfw_profiles, num_obs):
    num_sims = len(sample_mc_pairs)
    agg_mc_pairs = np.zeros((num_sims, num_obs, 2))
    agg_nfw_profiles = np.zeros((num_sims, num_obs, simulated_nfw_profiles.shape[1]))
    print(f"agg_mc_pairs shape: {agg_mc_pairs.shape}")
    print(f"agg_nfw_profiles shape: {agg_nfw_profiles.shape}")
    for i in range(num_sims):
        agg_idx = np.random.choice(len(sample_mc_pairs), size=num_obs, replace=False)
        agg_nfw_profiles[i] = simulated_nfw_profiles[agg_idx]
        agg_mc_pairs[i] = sample_mc_pairs[agg_idx]
    agg_mc_pairs = agg_mc_pairs.reshape(num_sims, num_obs * 2)
    agg_nfw_profiles = agg_nfw_profiles.reshape(
        num_sims, num_obs * simulated_nfw_profiles.shape[1]
    )
    theta_np = np.array(agg_mc_pairs.T).T
    x_np = agg_nfw_profiles

    print(f"theta_np shape: {theta_np.shape}")
    print(f"x_np shape: {x_np.shape}")

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


# Inference output: 25th percentile m-c pair, median m-c pair, 75th percentile m-c pair
def gen_agg_posterior2(inferrer, sample_mc_pairs, simulated_nfw_profiles, num_obs):
    # Create a data vector with num_sims x num_obs observations and num_sims x num_obs m-c pairs
    num_sims = len(sample_mc_pairs)
    agg_mc_pairs = np.zeros((num_sims, 3, 2))
    agg_nfw_profiles = np.zeros((num_sims, num_obs, simulated_nfw_profiles.shape[1]))
    print(f"agg_mc_pairs shape: {agg_mc_pairs.shape}")
    print(f"agg_nfw_profiles shape: {agg_nfw_profiles.shape}")
    for i in range(num_sims):
        agg_idx = np.random.choice(len(sample_mc_pairs), size=num_obs, replace=False)
        agg_nfw_profiles[i] = simulated_nfw_profiles[agg_idx]
        # TODO check that this is the right percentile approach
        # agg_mc_pairs[i] = np.percentile(sample_mc_pairs[agg_idx], [25, 50, 75], axis=0)
        agg_mc_pairs[i] = np.column_stack(
            (
                np.percentile(sample_mc_pairs[agg_idx][:, 0], [25, 50, 75]),
                np.flip(np.percentile(sample_mc_pairs[agg_idx][:, 1], [25, 50, 75])),
            )
        )

    # 6 because 3 percentiles (25, 50, 75) and 2 params (m, c)
    agg_mc_pairs = agg_mc_pairs.reshape(num_sims, 6)
    agg_nfw_profiles = agg_nfw_profiles.reshape(
        num_sims, num_obs * simulated_nfw_profiles.shape[1]
    )
    theta_np = np.array(agg_mc_pairs.T).T
    x_np = agg_nfw_profiles

    print(f"theta_np shape: {theta_np.shape}")
    print(f"x_np shape: {x_np.shape}")

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


def apply_observations_agg(posterior, drawn_mc_pairs, drawn_nfw_profiles, err_dex=0.0):
    from .sbiutils import create_agg_observation_nfw

    theta_o_agg, x_o_agg = create_agg_observation_nfw(
        drawn_mc_pairs, drawn_nfw_profiles
    )

    # Obtain samples of the posterior given the observation
    samples_agg = posterior.sample((10000,), x=x_o_agg)

    # Calculate the log-probability of the samples given the observation to find the maximum a posteriori (MAP) estimate
    logp_agg = posterior.log_prob(samples_agg, x=x_o_agg)
    idx_agg = np.argmax(logp_agg)
    map_mc_agg = samples_agg[idx_agg]

    return samples_agg.numpy(), map_mc_agg  # .reshape(3, 2)
