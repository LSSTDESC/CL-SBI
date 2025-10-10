import numpy as np
import torch

PERCENTILE_LEVELS = [5, 16, 25, 50, 75, 84, 95]


def create_observation_nfw(mc_pair, nfw_profile):
    """
    Observation of an NFW profile for some log10 M and concentration.

    Returns the parameters (input), and the observations (radial profile) as tensors
    """
    theta_truth_np = np.array(mc_pair)
    x_truth_np = np.array(nfw_profile)

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o


def create_join_fit_observation_nfw(mc_pairs, nfw_profiles):
    """
    Takes many model inputs (mc_pairs) and outputs (nfw_profiles), and returns the (tensorized) median input and median output

    Returns the parameters (input), and the observations (radial profile) as tensors
    """

    percentiles = np.percentile(mc_pairs, PERCENTILE_LEVELS, axis=0)
    theta = np.concatenate((percentiles[:, 0], percentiles[:, 1]))
    median_nfw_profile = np.median(nfw_profiles, axis=0)
    return create_observation_nfw(theta, median_nfw_profile)


def create_fit_join_observation_nfw(mc_pairs, nfw_profiles):
    """
    Takes many model inputs (mc_pairs) and outputs (nfw_profiles), and returns the (tensorized) median input with concatenated output

    Returns the parameters (input), and the observations (radial profile) as tensors
    """

    percentiles = np.percentile(mc_pairs, PERCENTILE_LEVELS, axis=0)
    theta = np.concatenate((percentiles[:, 0], percentiles[:, 1]))
    return create_observation_nfw(theta, nfw_profiles.flatten())
