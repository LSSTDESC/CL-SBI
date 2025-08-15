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


# truth is the list of m-c pairs
def create_agg_observation_nfw(mc_pairs, nfw_profiles):
    theta_truth_np = mc_pairs.reshape(-1)
    x_truth_np = nfw_profiles.reshape(len(nfw_profiles) * len(nfw_profiles[0]))
    print(f"theta_truth_np shape: {theta_truth_np.shape}")
    # print(f"theta_truth_np: {theta_truth_np}")
    print(f"x_truth_np shape: {x_truth_np.shape}")

    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)
    return theta_o, x_o


def create_agg_observation_nfw2(mc_pairs, nfw_profiles):
    # theta_truth_np = np.array(mc_pairs).reshape(-1)
    # Flattened array of mc_pairs, percentiles calculated by mass
    theta_truth_np = np.column_stack(
        (
            np.percentile(mc_pairs[:, 0], [25, 50, 75]),
            np.flip(np.percentile(mc_pairs[:, 1], [25, 50, 75])),
        )
    ).reshape(-1)
    # theta_truth_np = np.percentile(mc_pairs, [25, 50, 75], axis=0).reshape(-1)
    x_truth_np = nfw_profiles.reshape(len(nfw_profiles) * len(nfw_profiles[0]))

    print(f"theta_truth_np shape: {theta_truth_np.shape}")
    print(f"theta_truth_np: {theta_truth_np}")
    print(f"x_truth_np shape: {x_truth_np.shape}")
    # turn into tensors
    theta_o = torch.as_tensor(theta_truth_np, dtype=torch.float32)
    x_o = torch.as_tensor(x_truth_np, dtype=torch.float32)

    return theta_o, x_o
