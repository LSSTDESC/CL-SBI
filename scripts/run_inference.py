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
from weaklensclustersbi.inference import stackutils


def run_stacked_only(args, out_path, infer_config, drawn_nfw_profiles, sigmas, log_stage):
    """Two-stage + naive stacked MCMC only, appended to an existing inference dir."""
    with multiprocessing.Pool() as pool:
        t0 = time.perf_counter()
        (population_samples, stacked_samplers, population_params) = mcmc.fit_then_join_stacked(
            drawn_nfw_profiles, sigmas, infer_config["priors"], pool=pool
        )
        log_stage("mcmc_fit_then_join_stacked", time.perf_counter() - t0)
    with open(os.path.join(out_path, "mcmc_ftj_population_samples.pickle"), "wb") as handle:
        pickle.dump(population_samples, handle, protocol=4)
    with open(os.path.join(out_path, "mcmc_ftj_population_params.pickle"), "wb") as handle:
        pickle.dump(population_params, handle, protocol=4)
    with open(os.path.join(out_path, "mcmc_ftj_individual_samplers.pickle"), "wb") as handle:
        pickle.dump(stacked_samplers, handle, protocol=4)
    print("stacked-only outputs written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_id")
    parser.add_argument("--infer_id")
    parser.add_argument("--obs_id")
    parser.add_argument("--num_sims")
    parser.add_argument("--num_obs")
    # Observable: "surface_density" (default) or "delta_sigma". Must match the observable
    # used to generate sims/observations and train the posterior. Reads suffixed inputs,
    # writes suffixed inference outputs, and sets the MCMC forward-model observable.
    parser.add_argument("--observable", default="surface_density")
    # JTF stack estimator: "median" (default) or "corrected_mean". Must match the estimator
    # used for gen_simulations/train_inferrer; reads/writes further-suffixed dirs.
    parser.add_argument("--stack_estimator", default="median")
    # MCMC fit-then-join depends only on the individual observed profiles and priors --
    # NOT on the JTF stack estimator -- so for a non-median estimator we reuse the FTJ
    # outputs from the median twin dir when available (identical computation, different
    # emcee seed only). Disable to force a fresh FTJ run.
    parser.add_argument("--no_reuse_mcmc_ftj", action="store_true")
    # Explicit source dir for FTJ reuse (overrides the auto median-twin lookup). FTJ depends
    # only on the observations and priors, so any inference dir with the SAME obs_id and
    # infer_id is a valid source regardless of sim_id (e.g. hypermix runs reuse sim_z1's).
    parser.add_argument("--reuse_mcmc_ftj_from", default=None)

    # Add regenerate flag if we want to overwrite any existing posterior.
    # If false or not set, skip posterior generation if they already exist from an earlier run.
    parser.add_argument("--regenerate", action="store_true")
    # Flag to run the slow two-stage MCMC FTJ (individual fits + population inference)
    # Disabled by default since it runs 376 individual MCMCs
    parser.add_argument("--run_mcmc_stacked", action="store_true")
    # Run ONLY the two-stage/naive stacked MCMC into an EXISTING inference dir
    # (skips SBI and the standard JTF/FTJ MCMC; requires prior outputs present).
    parser.add_argument("--stacked_only", action="store_true")
    args = parser.parse_args()

    script_start = time.perf_counter()

    script_dir = os.path.dirname(__file__)
    obs_suffix = "" if args.observable == "surface_density" else f".{args.observable}"
    # observations are individual profiles (estimator-independent); posteriors/inference
    # outputs depend on the JTF stack estimator and get the extra suffix
    full_suffix = obs_suffix + stackutils.stack_suffix(args.stack_estimator)
    # Go to/create output directory
    out_rel_path = f"../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}{full_suffix}"
    out_path = os.path.join(script_dir, out_rel_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Checking if output already exists from an earlier script run
    if os.path.isfile(os.path.join(out_path, "true_param_median.npy")) and not args.stacked_only:
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
    posterior_rel_path = f"../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}.{args.num_obs}{full_suffix}"
    posterior_path = os.path.join(script_dir, posterior_rel_path)
    posterior_filename = os.path.join(posterior_path, "posterior.pickle")
    with open(posterior_filename, "rb") as handle:
        posterior = pickle.load(handle)

    posterior_jtf_filename = os.path.join(posterior_path, "posterior_jtf.pickle")
    with open(posterior_jtf_filename, "rb") as handle:
        posterior_jtf = pickle.load(handle)

    # Load inference/prior config (contains all MCMC prior parameters)
    infer_config_rel_path = "../configs/inference/"
    infer_config_path = os.path.join(script_dir, infer_config_rel_path)
    infer_config_filename = os.path.join(infer_config_path, f"{args.infer_id}.json")
    with open(infer_config_filename, "r") as f:
        infer_config = json.load(f)
    # Tell the MCMC forward model which observable to compute (Sigma vs DeltaSigma),
    # so the likelihood forward-models the same quantity as the observations.
    infer_config["priors"]["observable"] = args.observable

    # Load observations
    obs_rel_path = f"../outputs/observations/{args.obs_id}.{args.num_obs}{obs_suffix}"
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
    if args.stacked_only:
        return run_stacked_only(args, out_path, infer_config, drawn_nfw_profiles, sigmas, log_stage)
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
        posterior, posterior_jtf, drawn_mc_pairs, drawn_nfw_profiles,
        stack_estimator=args.stack_estimator, sigmas=sigmas,
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
        mcmc_jtf_chain, mcmc_jtf_sampler = mcmc.join_then_fit(
            drawn_nfw_profiles, sigmas, infer_config["priors"], pool=pool,
            stack_estimator=args.stack_estimator,
        )
        log_stage("mcmc_join_then_fit", time.perf_counter() - t0)

        # FTJ is estimator-independent: reuse an existing dir's outputs when available
        # (explicit --reuse_mcmc_ftj_from first, else the auto median-twin lookup)
        ftj_twin = None
        candidates = []
        if args.reuse_mcmc_ftj_from and not args.no_reuse_mcmc_ftj:
            candidates.append(args.reuse_mcmc_ftj_from)
        if args.stack_estimator != "median" and not args.no_reuse_mcmc_ftj:
            twin_rel = f"../outputs/inference/{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}{obs_suffix}"
            candidates.append(os.path.join(script_dir, twin_rel))
        for twin in candidates:
            if os.path.isfile(os.path.join(twin, "mcmc_ftj_samplers.pickle")) and os.path.isfile(
                os.path.join(twin, "mcmc_chains.pickle")
            ):
                ftj_twin = twin
                break
        if ftj_twin is not None:
            with open(os.path.join(ftj_twin, "mcmc_chains.pickle"), "rb") as handle:
                mcmc_ftj_chains = pickle.load(handle)[1]
            with open(os.path.join(ftj_twin, "mcmc_ftj_samplers.pickle"), "rb") as handle:
                mcmc_ftj_samplers = pickle.load(handle)
            print(f"reusing estimator-independent MCMC FTJ from {ftj_twin}")
            log_stage("mcmc_fit_then_join", 0.0, details="reused from median twin", status="reused")
        else:
            t0 = time.perf_counter()
            mcmc_ftj_chains, mcmc_ftj_samplers = mcmc.fit_then_join(
                drawn_nfw_profiles, sigmas, infer_config["priors"], pool=pool
            )
            log_stage("mcmc_fit_then_join", time.perf_counter() - t0)

        # Two-stage MCMC FTJ: individual fits + population inference (slow - 376 MCMCs)
        # Only run if --run_mcmc_stacked flag is set
        mcmc_ftj_population_samples = None
        mcmc_ftj_stacked_samplers = None
        mcmc_ftj_population_params = None
        if args.run_mcmc_stacked:
            t0 = time.perf_counter()
            mcmc_ftj_population_samples, mcmc_ftj_stacked_samplers, mcmc_ftj_population_params = mcmc.fit_then_join_stacked(
                drawn_nfw_profiles, sigmas, infer_config["priors"], pool=pool
            )
            log_stage("mcmc_fit_then_join_stacked", time.perf_counter() - t0)
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

    # The two-stage/naive-stacked FTJ outputs are likewise estimator-independent; carry
    # them over from the twin so downstream figures work without a --run_mcmc_stacked rerun.
    if ftj_twin is not None:
        import shutil

        for fn in (
            "mcmc_ftj_population_samples.pickle",
            "mcmc_ftj_population_params.pickle",
            "mcmc_ftj_individual_samplers.pickle",
        ):
            src = os.path.join(ftj_twin, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_path, fn))

    # Output MCMC fit-then-join stacked results (only if --run_mcmc_stacked was set)
    if args.run_mcmc_stacked and mcmc_ftj_population_samples is not None:
        with open(os.path.join(out_path, "mcmc_ftj_population_samples.pickle"), "wb") as handle:
            pickle.dump(mcmc_ftj_population_samples, handle, protocol=4)

        with open(os.path.join(out_path, "mcmc_ftj_population_params.pickle"), "wb") as handle:
            pickle.dump(mcmc_ftj_population_params, handle, protocol=4)

        with open(os.path.join(out_path, "mcmc_ftj_individual_samplers.pickle"), "wb") as handle:
            pickle.dump(mcmc_ftj_stacked_samplers, handle, protocol=4)

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
