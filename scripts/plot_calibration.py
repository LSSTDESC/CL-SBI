#!/usr/bin/env python3
"""Generate calibration (coverage) plots for a single experiment."""
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from typing import Dict, Tuple

from context import plotutils, population
from runtime_log import append_runtime_log

CONF_LEVELS = np.linspace(0, 1, 20)
TRUTH_SAMPLE_SIZE = 5000
POSTERIOR_SAMPLE_SIZE = 5000
METHOD_LABELS = {
    ("mcmc", "jtf"): "MCMC join-then-fit",
    ("mcmc", "ftj"): "MCMC fit-then-join",
    ("sbi", "jtf"): "SBI join-then-fit",
    ("sbi", "ftj"): "SBI fit-then-join",
}


def draw_posterior_samples(chain: np.ndarray, n_samples: int) -> np.ndarray:
    split = plotutils.split_percentile_chain(chain)
    if split:
        summary = plotutils.build_gaussian_summary_from_chain(chain)
        if summary is None:
            raise ValueError("Unable to build Gaussian summary from percentile chain")
        mu = np.array([summary["mass"]["mu"], summary["concentration"]["mu"]])
        sigma_mass = summary["mass"]["sigma"]
        sigma_conc = summary["concentration"]["sigma"]
        rho = summary.get("correlation", {}).get("rho", 0.0)
        cov = np.array(
            [
                [sigma_mass**2, rho * sigma_mass * sigma_conc],
                [rho * sigma_mass * sigma_conc, sigma_conc**2],
            ]
        )
        rng = np.random.default_rng(42)
        return rng.multivariate_normal(mu, cov, size=n_samples)

    if chain.ndim == 2 and chain.shape[1] >= 2:
        base = chain[:, :2]
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(base), size=n_samples)
        return base[idx]
    raise ValueError("Unexpected chain format when drawing posterior samples")


def sample_truth_ftj(
    obs_config: Dict, n_samples: int = TRUTH_SAMPLE_SIZE
) -> np.ndarray:
    mc_pairs = population.gen_mc_pairs_in_richness_bin(
        obs_config["min_richness"],
        obs_config["max_richness"],
        rm_relation=obs_config["rm_relation"],
        mc_relation=obs_config["mc_relation"],
        num_samples=n_samples,
        mc_scatter=obs_config["mc_scatter"],
        rm_scatter=obs_config["rm_scatter"],
        min_z=obs_config["min_z"],
        max_z=obs_config["max_z"],
    )
    return np.asarray(mc_pairs)


def sample_truth_jtf(
    obs_config: Dict, num_obs: int, n_samples: int = TRUTH_SAMPLE_SIZE
) -> np.ndarray:
    medians = []
    for _ in range(n_samples):
        draws = population.gen_mc_pairs_in_richness_bin(
            obs_config["min_richness"],
            obs_config["max_richness"],
            rm_relation=obs_config["rm_relation"],
            mc_relation=obs_config["mc_relation"],
            num_samples=num_obs,
            mc_scatter=obs_config["mc_scatter"],
            rm_scatter=obs_config["rm_scatter"],
            min_z=obs_config["min_z"],
            max_z=obs_config["max_z"],
        )
        medians.append(np.median(draws, axis=0))
    return np.asarray(medians)


def compute_coverage(
    samples: np.ndarray, truth_samples: np.ndarray
) -> Dict[float, float]:
    if samples.ndim != 2:
        raise ValueError("Posterior samples must be 2-D")
    if truth_samples.ndim != 2 or truth_samples.shape[1] != samples.shape[1]:
        raise ValueError("Truth samples must match posterior dimensionality")

    cov = np.cov(samples.T)
    mean = np.mean(samples, axis=0)
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    dists_samples = np.sum((samples - mean) @ cov_inv * (samples - mean), axis=1)
    dists_truth = np.sum(
        (truth_samples - mean) @ cov_inv * (truth_samples - mean), axis=1
    )
    dists_samples_sorted = np.sort(dists_samples)
    percentiles = np.searchsorted(
        dists_samples_sorted, dists_truth, side="right"
    ) / len(dists_samples_sorted)

    coverage = {}
    for p in CONF_LEVELS:
        coverage[p] = float(np.mean(percentiles <= p))
    return coverage


def aggregate_coverage(
    run_path: str, truth_map: Dict[str, np.ndarray]
) -> Dict[Tuple[str, str], Dict[float, float]]:
    aggregates = {}

    chains_file = os.path.join(run_path, "mcmc_chains.pickle")
    sbi_file = os.path.join(run_path, "sbi_chains.pickle")
    if not (os.path.exists(chains_file) and os.path.exists(sbi_file)):
        raise FileNotFoundError(
            "Expected inference artifacts not found. Run inference first."
        )

    with open(chains_file, "rb") as handle:
        mcmc_jtf, mcmc_ftj = pickle.load(handle)
    with open(sbi_file, "rb") as handle:
        sbi_jtf, sbi_ftj = pickle.load(handle)
    chains = {
        ("mcmc", "jtf"): draw_posterior_samples(mcmc_jtf, POSTERIOR_SAMPLE_SIZE),
        ("mcmc", "ftj"): draw_posterior_samples(mcmc_ftj, POSTERIOR_SAMPLE_SIZE),
        ("sbi", "jtf"): draw_posterior_samples(sbi_jtf, POSTERIOR_SAMPLE_SIZE),
        ("sbi", "ftj"): draw_posterior_samples(sbi_ftj, POSTERIOR_SAMPLE_SIZE),
    }
    for key, samples in chains.items():
        branch = key[1]
        aggregates[key] = compute_coverage(samples, truth_map[branch])
    return aggregates


def plot_calibration(
    aggregates: Dict[Tuple[str, str], Dict[float, float]],
    output_dir: str,
    suffix: str,
) -> None:
    # plt.figure(figsize=(6, 6))
    x = CONF_LEVELS
    for key, cover_dict in aggregates.items():
        y = [cover_dict.get(p, np.nan) for p in x]
        plt.plot(x, y, label=METHOD_LABELS[key])
    plt.plot(x, x, color="black", linestyle="--", label="Ideal")
    # plt.fill_between(x, x, 1, color="gray", alpha=0.1)
    # plt.text(0.6, 0.85, "Underconfident", fontsize=10, color="gray")
    # plt.text(0.2, 0.2, "Overconfident", fontsize=10, color="gray")
    plt.xlabel("Estimated Confidence Level")
    plt.ylabel("Fraction of Truth Covered")
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"calibration_{suffix}.png"))
    plt.savefig(os.path.join(output_dir, f"calibration_{suffix}.pdf"))
    plt.close()

    # # Also produce combined plot when both FTJ and JTF are present
    # combined_path_png = os.path.join(output_dir, "calibration.png")
    # combined_path_pdf = os.path.join(output_dir, "calibration.pdf")
    # if suffix == "combined":  # assume FTJ was plotted first
    #     # plt.figure(figsize=(6, 6))
    #     x = CONF_LEVELS
    #     for key, cover_dict in aggregates.items():
    #         y = [cover_dict.get(p, np.nan) for p in x]
    #         plt.plot(x, y, label=METHOD_LABELS[key])
    #     plt.plot(x, x, color="black", linestyle="--", label="Ideal")
    #     # plt.fill_between(x, x, 1, color="gray", alpha=0.1)
    #     plt.xlabel("Estimated Confidence Level")
    #     plt.ylabel("Fraction of Truth Covered")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(combined_path_png)
    #     plt.savefig(combined_path_pdf)
    #     plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot calibration curve for an experiment"
    )
    parser.add_argument("--sim_id", required=True)
    parser.add_argument("--infer_id", required=True)
    parser.add_argument("--obs_id", required=True)
    parser.add_argument("--num_sims", required=True)
    parser.add_argument("--num_obs", required=True)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    script_start = time.perf_counter()

    def log_runtime(status="success", details=""):
        append_runtime_log(
            stage="plot_calibration",
            seconds=time.perf_counter() - script_start,
            sim_id=args.sim_id,
            infer_id=args.infer_id,
            obs_id=args.obs_id,
            num_sims=args.num_sims,
            num_obs=args.num_obs,
            details=details,
            status=status,
        )

    script_dir = os.path.dirname(__file__)
    run_path = os.path.join(
        script_dir,
        f"../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}",
    )
    out_dir = os.path.join(
        script_dir,
        f"../outputs/plots/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}/calibration",
    )

    obs_config_path = os.path.join(
        script_dir, f"../configs/observations/{args.obs_id}.json"
    )
    with open(obs_config_path, "r") as f:
        obs_config = json.load(f)
    truth_map = {
        "ftj": sample_truth_ftj(obs_config),
        "jtf": sample_truth_jtf(obs_config, int(args.num_obs)),
    }

    os.makedirs(out_dir, exist_ok=True)
    existing = os.path.exists(os.path.join(out_dir, "calibration_ftj.png"))
    if existing and not args.regenerate:
        print("Calibration plots already exist. Re-run with --regenerate to overwrite.")
        log_runtime(status="skipped", details="existing calibration")
        return
    plt.style.use(os.path.join(script_dir, f"../plot/mplstyle.txt"))

    try:
        aggregates = aggregate_coverage(run_path, truth_map)
        ftj_only = {k: v for k, v in aggregates.items() if k[1] == "ftj"}
        jtf_only = {k: v for k, v in aggregates.items() if k[1] == "jtf"}
        plot_calibration(ftj_only, out_dir, "ftj")
        plot_calibration(jtf_only, out_dir, "jtf")
        # plot calibration with both methods together
        plot_calibration(aggregates, out_dir, "combined")
        log_runtime()
    except FileNotFoundError as exc:
        log_runtime(status="error", details=str(exc))
        raise


if __name__ == "__main__":
    main()
