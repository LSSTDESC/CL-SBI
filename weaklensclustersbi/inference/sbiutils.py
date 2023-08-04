import numpy as np
import torch


def create_observation_nfw(mc_pair, nfw_profile):
    '''
    Observation of an NFW profile for some log10 M and concentration.

    Returns the parameters (input), and the observations (radial profile) as tensors
    '''
    theta_truth_np = np.array(mc_pair)
    x_truth_np = np.array(nfw_profile)

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o


def create_join_fit_observation_nfw(mc_pairs, nfw_profiles):
    '''
    Takes many model inputs (mc_pairs) and outputs (nfw_profiles), and returns the (tensorized) mean input and mean output

    Returns the parameters (input), and the observations (radial profile) as tensors
    '''

    mean_mc_pair = np.mean(mc_pairs, keepdims=True, axis=0)[0]
    mean_nfw_profile = np.mean(nfw_profiles, keepdims=True, axis=0)[0]

    return create_observation_nfw(mean_mc_pair, mean_nfw_profile)
