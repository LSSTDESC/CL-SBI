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


def create_join_fit_observation_nfw(mc_pairs, noise_dex):
    '''
    Observation of an NFW profile for some log10 M and concentration.

    Returns the parameters (input), and the observations (radial profile) as tensors
    '''

    from ..simulations.wlprofile import simulate_nfw

    theta_truth_np = np.mean(mc_pairs, keepdims=True, axis=0)[0]
    nfw_profile = simulate_nfw(log10mass=theta_truth_np[0],
                               concentration=theta_truth_np[1])

    # add error up to 0.2 dex
    noisy_nfw_profile = calculate_noise(nfw_profile, noise_dex)

    x_truth_np = noisy_nfw_profile

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o


def create_fit_join_observation_nfw(mc_pairs, nfw_profiles, noise_dex):
    '''
    Observation of an NFW profile for some log10 M and concentration.

    Returns the parameters (input), and the observations (radial profile) as tensors
    '''

    theta_truth_np = np.mean(mc_pairs, keepdims=True, axis=0)[0]
    x_truth_np = np.mean(nfw_profiles, keepdims=True, axis=0)[0]

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o  #, theta_o_join_fit, x_o_join_fit


# TODO: dupe. If we need it here too, we should extract this to a diff file
def calculate_noise(sample, dex=0.0):
    random_noise = np.random.normal(1, dex, np.shape(sample))
    return 10**(random_noise * np.log10(sample))
