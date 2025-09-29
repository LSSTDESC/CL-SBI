import numpy as np
import torch


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

    median_mc_pair = np.median(mc_pairs, keepdims=True, axis=0)[0]
    median_nfw_profile = np.median(nfw_profiles, axis=0)
    return create_observation_nfw(median_mc_pair, median_nfw_profile)


def create_fit_join_observation_nfw(mc_pairs, nfw_profiles):
    """
    Takes many model inputs (mc_pairs) and outputs (nfw_profiles), and returns the (tensorized) median input with concatenated output

    Returns the parameters (input), and the observations (radial profile) as tensors
    """

    median_mc_pair = np.median(mc_pairs, keepdims=True, axis=0)[0]
    # print(np.shape(nfw_profiles))
    # nfw_profiles = np.reshape(np.shape(nfw_profiles)[0], -1)
    # print(np.shape(nfw_profiles))
    return create_observation_nfw(median_mc_pair, nfw_profiles.flatten())
