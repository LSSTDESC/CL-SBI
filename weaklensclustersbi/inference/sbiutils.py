import numpy as np
import torch


def create_observation_nfw(log10M_obs_true, concentration_obs_true):
    '''
    Observation of an NFW profile for some log10 M and concentration.

    Returns the parameters (input), and the observations (radial profile) as tensors
    '''
    from ..simulations.wlprofile import simulate_nfw
    theta_truth_np = np.array([log10M_obs_true, concentration_obs_true])
    x_truth_np = simulate_nfw(log10mass=log10M_obs_true,
                              concentration=concentration_obs_true)

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o


def create_join_fit_observation_nfw(mc_pairs, nfw_profiles):
    '''
    Observation of an NFW profile for some log10 M and concentration.

    Returns the parameters (input), and the observations (radial profile) as tensors
    '''

    from ..simulations.wlprofile import simulate_nfw

    theta_truth_np = np.mean(mc_pairs, keepdims=True, axis=0)[0]
    x_truth_np = simulate_nfw(log10mass=theta_truth_np[0],
                              concentration=theta_truth_np[1])

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o


def create_fit_join_observation_nfw(mc_pairs, nfw_profiles):
    '''
    Observation of an NFW profile for some log10 M and concentration.

    Returns the parameters (input), and the observations (radial profile) as tensors
    '''

    theta_truth_np = np.mean(mc_pairs, keepdims=True, axis=0)[0]
    x_truth_np = np.mean(nfw_profiles, keepdims=True, axis=0)[0]

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o
