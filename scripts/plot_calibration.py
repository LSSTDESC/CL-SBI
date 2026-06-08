#!/usr/bin/env python3
"""Generate calibration (coverage) plots for a single experiment."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from typing import Dict, Tuple
from scipy import stats

from plot import plotutils
from weaklensclustersbi.simulations import population
from runtime_log import append_runtime_log

CONF_LEVELS = np.linspace(0, 1, 20)
TRUTH_SAMPLE_SIZE = 1000  # Reduced from 5000 for speed
POSTERIOR_SAMPLE_SIZE = 1000  # Reduced from 5000 for speed

# Cache for truth samples (keyed by obs_config hash)
_TRUTH_CACHE: dict = {}
METHOD_LABELS = {
    ("mcmc", "jtf"): "MCMC join-then-fit",
    ("mcmc", "ftj"): "MCMC fit-then-join",
    ("mcmc", "ftj_twostage"): "MCMC fit-then-join (two-stage)",
    ("mcmc", "ftj_naive"): "MCMC fit-then-join (naive stacking)",
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


def _obs_config_hash(obs_config: Dict, suffix: str = "") -> str:
    """Create a hashable key from obs_config for caching."""
    key_parts = [
        str(obs_config.get("min_richness")),
        str(obs_config.get("max_richness")),
        str(obs_config.get("rm_relation")),
        str(obs_config.get("mc_relation")),
        str(obs_config.get("mc_scatter")),
        str(obs_config.get("rm_scatter")),
        str(obs_config.get("min_z")),
        str(obs_config.get("max_z")),
        str(obs_config.get("richness_contam_frac", 0.0)),
        suffix,
    ]
    return "|".join(key_parts)


def sample_truth_ftj(
    obs_config: Dict, n_samples: int = TRUTH_SAMPLE_SIZE
) -> np.ndarray:
    cache_key = _obs_config_hash(obs_config, f"ftj_{n_samples}")
    if cache_key in _TRUTH_CACHE:
        return _TRUTH_CACHE[cache_key]

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
        richness_contam_frac=obs_config.get("richness_contam_frac", 0.0),
        lambda_min_contam=obs_config.get("min_richness_contam", None),
    )
    result = np.asarray(mc_pairs)
    _TRUTH_CACHE[cache_key] = result
    return result


def sample_truth_jtf(
    obs_config: Dict, num_obs: int, n_samples: int = TRUTH_SAMPLE_SIZE
) -> np.ndarray:
    cache_key = _obs_config_hash(obs_config, f"jtf_{n_samples}_{num_obs}")
    if cache_key in _TRUTH_CACHE:
        return _TRUTH_CACHE[cache_key]

    # Vectorized: generate all samples at once, then reshape and compute medians
    total_samples = n_samples * num_obs
    all_draws = population.gen_mc_pairs_in_richness_bin(
        obs_config["min_richness"],
        obs_config["max_richness"],
        rm_relation=obs_config["rm_relation"],
        mc_relation=obs_config["mc_relation"],
        num_samples=total_samples,
        mc_scatter=obs_config["mc_scatter"],
        rm_scatter=obs_config["rm_scatter"],
        min_z=obs_config["min_z"],
        max_z=obs_config["max_z"],
    )
    # Reshape to (n_samples, num_obs, 2) and take median over the num_obs axis
    reshaped = np.asarray(all_draws).reshape(n_samples, num_obs, 2)
    result = np.median(reshaped, axis=1)
    _TRUTH_CACHE[cache_key] = result
    return result


def compute_ks_pvalues(
    samples: np.ndarray, truth_samples: np.ndarray
) -> Dict[str, float]:
    """Compute 2-sample KS-test p-values for mass and concentration.

    A high p-value (>0.05) indicates the posterior is consistent with the truth distribution.
    A low p-value suggests the posterior is biased or miscalibrated.
    """
    ks_mass = stats.ks_2samp(samples[:, 0], truth_samples[:, 0])
    ks_conc = stats.ks_2samp(samples[:, 1], truth_samples[:, 1])
    return {
        "mass_ks_stat": float(ks_mass.statistic),
        "mass_ks_pvalue": float(ks_mass.pvalue),
        "conc_ks_stat": float(ks_conc.statistic),
        "conc_ks_pvalue": float(ks_conc.pvalue),
    }


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
    run_path: str, truth_map: Dict[str, np.ndarray], ftj_only: bool = False
) -> Tuple[Dict[Tuple[str, str], Dict[float, float]], Dict[Tuple[str, str], Dict[str, float]]]:
    """Compute coverage and KS-test metrics for all methods.

    Returns:
        Tuple of (coverage_dict, ks_dict) where each maps (method, branch) to metrics.
    """
    aggregates = {}
    ks_results = {}

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
        ("mcmc", "ftj"): draw_posterior_samples(mcmc_ftj, POSTERIOR_SAMPLE_SIZE),
        ("sbi", "ftj"): draw_posterior_samples(sbi_ftj, POSTERIOR_SAMPLE_SIZE),
    }
    if not ftj_only:
        chains[("mcmc", "jtf")] = draw_posterior_samples(mcmc_jtf, POSTERIOR_SAMPLE_SIZE)
        chains[("sbi", "jtf")] = draw_posterior_samples(sbi_jtf, POSTERIOR_SAMPLE_SIZE)

    # Load two-stage FTJ population samples if available
    population_file = os.path.join(run_path, "mcmc_ftj_population_samples.pickle")
    if os.path.exists(population_file):
        with open(population_file, "rb") as handle:
            mcmc_ftj_population = pickle.load(handle)
        chains[("mcmc", "ftj_twostage")] = draw_posterior_samples(
            mcmc_ftj_population, POSTERIOR_SAMPLE_SIZE
        )

    # Load naive stacking samples if available (from individual samplers)
    individual_samplers_file = os.path.join(run_path, "mcmc_ftj_individual_samplers.pickle")
    if os.path.exists(individual_samplers_file):
        with open(individual_samplers_file, "rb") as handle:
            individual_samplers = pickle.load(handle)
        # Concatenate flatchain from each individual sampler (only M, c columns)
        naive_chains = [s.flatchain[:, :2] for s in individual_samplers]
        mcmc_ftj_naive = np.concatenate(naive_chains, axis=0)
        chains[("mcmc", "ftj_naive")] = draw_posterior_samples(
            mcmc_ftj_naive, POSTERIOR_SAMPLE_SIZE
        )

    for key, samples in chains.items():
        branch = key[1]
        # Map ftj_twostage and ftj_naive to ftj truth for comparison
        truth_key = "ftj" if branch in ("ftj_twostage", "ftj_naive") else branch
        aggregates[key] = compute_coverage(samples, truth_map[truth_key])
        ks_results[key] = compute_ks_pvalues(samples, truth_map[truth_key])
    return aggregates, ks_results


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

    # also save the raw data
    with open(os.path.join(output_dir, f"calibration_{suffix}.pickle"), "wb") as handle:
        pickle.dump(aggregates, handle)


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
    parser.add_argument("--ftj-only", action="store_true",
                        help="Skip JTF calibration (much faster)")
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

    # Load actual observations as FTJ truth (not freshly generated samples)
    # This avoids sampling variance issues when comparing posteriors
    obs_path = os.path.join(
        script_dir,
        f"../outputs/observations/{args.obs_id}.{args.num_obs}",
    )
    actual_obs = np.load(os.path.join(obs_path, "drawn_mc_pairs.npy"))
    truth_map = {"ftj": actual_obs}

    if not args.ftj_only:
        print("Generating JTF truth samples (slow)...")
        truth_map["jtf"] = sample_truth_jtf(obs_config, int(args.num_obs))

    os.makedirs(out_dir, exist_ok=True)
    existing = os.path.exists(os.path.join(out_dir, "calibration_ftj.png"))
    if existing and not args.regenerate:
        print("Calibration plots already exist. Re-run with --regenerate to overwrite.")
        log_runtime(status="skipped", details="existing calibration")
        return
    plt.style.use(os.path.join(script_dir, f"../plot/mplstyle.txt"))

    try:
        aggregates, ks_results = aggregate_coverage(run_path, truth_map, ftj_only=args.ftj_only)
        # Include ftj, ftj_twostage, and ftj_naive in FTJ comparison
        ftj_agg = {k: v for k, v in aggregates.items() if k[1] in ("ftj", "ftj_twostage", "ftj_naive")}
        plot_calibration(ftj_agg, out_dir, "ftj")

        if not args.ftj_only:
            jtf_agg = {k: v for k, v in aggregates.items() if k[1] == "jtf"}
            plot_calibration(jtf_agg, out_dir, "jtf")
            # plot calibration with both methods together
            plot_calibration(aggregates, out_dir, "combined")

        # Save KS-test results as CSV for easy inspection
        ks_csv_path = os.path.join(out_dir, "ks_test_results.csv")
        with open(ks_csv_path, "w") as f:
            f.write("method,branch,mass_ks_stat,mass_ks_pvalue,conc_ks_stat,conc_ks_pvalue\n")
            for (method, branch), metrics in sorted(ks_results.items()):
                f.write(f"{method},{branch},{metrics['mass_ks_stat']:.4f},{metrics['mass_ks_pvalue']:.4f},"
                        f"{metrics['conc_ks_stat']:.4f},{metrics['conc_ks_pvalue']:.4f}\n")
        print(f"\nKS-test results saved to {ks_csv_path}")
        print("\nKS-test p-values (high = consistent with truth):")
        print("-" * 60)
        for (method, branch), metrics in sorted(ks_results.items()):
            label = METHOD_LABELS.get((method, branch), f"{method} {branch}")
            print(f"{label:35s}  M: p={metrics['mass_ks_pvalue']:.3f}  c: p={metrics['conc_ks_pvalue']:.3f}")

        # Also save as pickle for programmatic access
        with open(os.path.join(out_dir, "ks_test_results.pickle"), "wb") as f:
            pickle.dump(ks_results, f)

        log_runtime()
    except FileNotFoundError as exc:
        log_runtime(status="error", details=str(exc))
        raise


if __name__ == "__main__":
    main()
