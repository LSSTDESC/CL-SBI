import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weaklensclustersbi.inference import sbi_, mcmc
import numpy as np
import json
import argparse
import os
import pickle
import time
import multiprocessing
from runtime_log import append_runtime_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_id")
    parser.add_argument("--infer_id")
    parser.add_argument("--obs_id")
    parser.add_argument("--num_sims")
    parser.add_argument("--num_obs")

    # Add regenerate flag if we want to overwrite any existing posterior.
    # If false or not set, skip posterior generation if they already exist from an earlier run.
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    script_start = time.perf_counter()

    script_dir = os.path.dirname(__file__)
    # Go to/create output directory
    out_rel_path = f"../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}"
    out_path = os.path.join(script_dir, out_rel_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Checking if output already exists from an earlier script run
    if os.path.isfile(os.path.join(out_path, "true_param_median.npy")):
        # Rerunning inference (continuing script)
        if args.regenerate:
            print("Overwriting existing inference outputs because of --regenerate flag")
        # Using existing inference outputs (terminating script)
        else:
            print(
                "Inference output already exists. If you want to regenerate, re-run with the --regenerate flag"
            )
            append_runtime_log(
                stage="run_inference_total",
                seconds=time.perf_counter() - script_start,
                sim_id=args.sim_id,
                infer_id=args.infer_id,
                obs_id=args.obs_id,
                num_sims=args.num_sims,
                num_obs=args.num_obs,
                details="existing inference",
                status="skipped",
            )
            quit()

    # Load posterior
    posterior_rel_path = f"../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}.{args.num_obs}"
    posterior_path = os.path.join(script_dir, posterior_rel_path)
    posterior_filename = os.path.join(posterior_path, "posterior.pickle")
    with open(posterior_filename, "rb") as handle:
        posterior = pickle.load(handle)

    posterior_jtf_filename = os.path.join(posterior_path, "posterior_jtf.pickle")
    with open(posterior_jtf_filename, "rb") as handle:
        posterior_jtf = pickle.load(handle)

    # Load infer config
    infer_config_rel_path = "../configs/inference/"
    infer_config_path = os.path.join(script_dir, infer_config_rel_path)
    infer_config_filename = os.path.join(infer_config_path, f"{args.infer_id}.json")
    with open(infer_config_filename, "r") as f:
        infer_config = json.load(f)

    # Load observations
    obs_rel_path = f"../outputs/observations/{args.obs_id}.{args.num_obs}"
    obs_path = os.path.join(script_dir, obs_rel_path)
    drawn_mc_pairs_filename = os.path.join(obs_path, "drawn_mc_pairs.npy")
    drawn_nfw_profiles_filename = os.path.join(obs_path, "drawn_nfw_profiles.npy")
    sigmas_filename = os.path.join(obs_path, "sigmas.npy")
    drawn_mc_pairs = np.load(drawn_mc_pairs_filename)
    drawn_nfw_profiles = np.load(drawn_nfw_profiles_filename)
    sigmas = np.load(sigmas_filename)

    def log_stage(stage, seconds, details="", status="success"):
        append_runtime_log(
            stage=stage,
            seconds=seconds,
            sim_id=args.sim_id,
            infer_id=args.infer_id,
            obs_id=args.obs_id,
            num_sims=args.num_sims,
            num_obs=args.num_obs,
            details=details,
            status=status,
        )

    # Run SBI inference
    t0_total_sbi = time.perf_counter()
    t0 = time.perf_counter()
    (
        sbi_jtf_chains,
        sbi_ftj_chains,
        sbi_jtf_mc,
        sbi_ftj_mc,
        sbi_jtf_logprob,
        sbi_ftj_logprob,
    ) = sbi_.apply_observations(
        posterior, posterior_jtf, drawn_mc_pairs, drawn_nfw_profiles
    )
    log_stage("sbi_apply_observations", time.perf_counter() - t0, details="stack=both")
    log_stage("sbi_total", time.perf_counter() - t0_total_sbi)

    # Output SBI chains
    with open(os.path.join(out_path, "sbi_chains.pickle"), "wb") as handle:
        pickle.dump([sbi_jtf_chains, sbi_ftj_chains], handle, protocol=4)
    # Output each of the fit-then-join chains (for diagnostics)
    # with open(os.path.join(out_path, "sbi_ftj_chains.pickle"), "wb") as handle:
    #     pickle.dump(sbi_ftj_chains, handle, protocol=4)

    # Output inferred m-c pairs from SBI
    with open(os.path.join(out_path, "sbi_ftj_mc.pickle"), "wb") as handle:
        pickle.dump(sbi_ftj_mc, handle, protocol=4)

    with open(os.path.join(out_path, "sbi_jtf_mc.pickle"), "wb") as handle:
        pickle.dump(sbi_jtf_mc, handle, protocol=4)

    with open(os.path.join(out_path, "sbi_jtf_logprob.pickle"), "wb") as handle:
        pickle.dump(sbi_jtf_logprob, handle, protocol=4)

    with open(os.path.join(out_path, "sbi_ftj_logprob.pickle"), "wb") as handle:
        pickle.dump(sbi_ftj_logprob, handle, protocol=4)

    # Run MCMC inference
    t0_total_mcmc = time.perf_counter()
    with multiprocessing.Pool() as pool:
        t0 = time.perf_counter()
        jtf_sigmas = np.sqrt(np.pi / 2) * sigmas / np.sqrt(int(args.num_obs))
        mcmc_jtf_chain, mcmc_jtf_sampler = mcmc.join_then_fit(
            drawn_nfw_profiles, jtf_sigmas, infer_config["priors"], pool=pool
        )
        log_stage("mcmc_join_then_fit", time.perf_counter() - t0)

        t0 = time.perf_counter()
        mcmc_ftj_chains, mcmc_ftj_samplers = mcmc.fit_then_join(
            drawn_nfw_profiles, sigmas, infer_config["priors"], pool=pool
        )
        log_stage("mcmc_fit_then_join", time.perf_counter() - t0)
    log_stage("mcmc_total", time.perf_counter() - t0_total_mcmc)

    # Output MCMC chains (pickling because diff sizes)
    with open(os.path.join(out_path, "mcmc_chains.pickle"), "wb") as handle:
        pickle.dump([mcmc_jtf_chain, mcmc_ftj_chains], handle, protocol=4)

    # Output MCMC join-then-fit sampler (for diagnostics)
    with open(os.path.join(out_path, "mcmc_jtf_sampler.pickle"), "wb") as handle:
        pickle.dump(mcmc_jtf_sampler, handle, protocol=4)

    # Output MCMC fit-then-join samplers (for diagnostics)
    with open(os.path.join(out_path, "mcmc_ftj_samplers.pickle"), "wb") as handle:
        pickle.dump(mcmc_ftj_samplers, handle, protocol=4)

    # Output median and percentile (25th and 75th) of drawn m-c pairs as "truth" value for plotting
    true_param_median = (np.median(drawn_mc_pairs.T[0]), np.median(drawn_mc_pairs.T[1]))
    true_param_25 = (
        np.percentile(drawn_mc_pairs.T[0], 25),
        np.percentile(drawn_mc_pairs.T[1], 25),
    )
    true_param_75 = (
        np.percentile(drawn_mc_pairs.T[0], 75),
        np.percentile(drawn_mc_pairs.T[1], 75),
    )

    np.save(os.path.join(out_path, "true_param_median.npy"), true_param_median)
    np.save(os.path.join(out_path, "true_param_25th_percentile.npy"), true_param_25)
    np.save(os.path.join(out_path, "true_param_75th_percentile.npy"), true_param_75)

    log_stage("run_inference_total", time.perf_counter() - script_start)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # windows-safe startup
    main()
