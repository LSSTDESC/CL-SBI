"""
Run hierarchical HMC (NumPyro/NUTS) on all experiment observation sets, publication-grade,
with full timing + convergence diagnostics. Results -> hierarchical_poc_outputs/hier_all_experiments.csv

Key point: the hierarchical model fits the *observed* profiles with the differentiable NFW
forward model and broad population hyperpriors. It does NOT use the SBI training sims or an
informed MCMC prior, so its result depends only on the observation set. The 16 experiment
directories share 8 unique observation sets, so we run 8 unique HMC fits and map them back.

Timing is recorded separately for JIT-compile (one-time, analogous to SBI training cost) and
sampling, so the SBI-vs-HMC speed comparison is fair.
"""
import os
# MUST set device count before JAX initializes (importing H triggers JAX init).
NCHAINS = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NCHAINS}"

import time, csv, pickle
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary

import hierarchical_mcmc_poc as H  # reuse forward model + config

jax.config.update("jax_enable_x64", True)
_NDEV = jax.local_device_count()
CHAIN_METHOD = "parallel" if _NDEV >= NCHAINS else "sequential"
print(f"JAX devices={_NDEV} -> running {NCHAINS} chains '{CHAIN_METHOD}'")

REPO = H.REPO
OUT_DIR = H.OUT_DIR
CSV = f"{OUT_DIR}/hier_all_experiments.csv"

NWARMUP, NSAMPLES = 1000, 1500   # NCHAINS set at top (before JAX init)

# ---- discover unique observation sets and which experiment dirs use each ----
exp_dirs = sorted(d for d in os.listdir(f"{REPO}/outputs/inference") if d.endswith("376"))
obs_to_dirs = {}
for d in exp_dirs:
    obs_to_dirs.setdefault(d.split(".")[2], []).append(d)


def load_obs(obs_id):
    base = f"{REPO}/outputs/observations/{obs_id}.376"
    profiles = jnp.array(np.load(f"{base}/drawn_nfw_profiles.npy"))
    sigmas = jnp.array(np.load(f"{base}/sigmas.npy"))
    true_mc = np.load(f"{base}/drawn_mc_pairs.npy")
    return profiles, sigmas, true_mc


def make_model(profiles, sigmas):
    N_c = profiles.shape[0]
    def model(obs=None):
        mu_M  = numpyro.sample("mu_M",  dist.Normal(14.4, 0.4))
        sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.4))
        c0    = numpyro.sample("c0",    dist.Normal(4.6, 1.5))
        beta  = numpyro.sample("beta",  dist.Normal(0.0, 2.0))
        sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.7))
        sig_extra = numpyro.sample("sig_extra", dist.HalfNormal(0.15))
        with numpyro.plate("clusters", N_c):
            zM = numpyro.sample("zM", dist.Normal(0, 1)); logM_j = mu_M + sig_M * zM
            zc = numpyro.sample("zc", dist.Normal(0, 1)); c_j = c0 + beta * (logM_j - mu_M) + sig_c * zc
        model_logSig = H.nfw_logSigma_vmap(logM_j, c_j)
        sig_tot = jnp.sqrt(sigmas[None, :] ** 2 + sig_extra ** 2)
        numpyro.sample("obs", dist.Normal(model_logSig, sig_tot), obs=obs)
    return model


HYPER = ["mu_M", "sig_M", "c0", "beta", "sig_c", "sig_extra"]


def run_one(obs_id):
    profiles, sigmas, true_mc = load_obs(obs_id)
    model = make_model(profiles, sigmas)
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median)
    mcmc = MCMC(kernel, num_warmup=NWARMUP, num_samples=NSAMPLES, num_chains=NCHAINS,
                chain_method=CHAIN_METHOD, progress_bar=False)

    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(0), obs=profiles)
    mcmc.get_samples()["mu_M"].block_until_ready()
    t_total = time.time() - t0

    # second run, same shapes -> compiled; isolates pure sampling to back out compile cost
    t1 = time.time()
    mcmc2 = MCMC(kernel, num_warmup=NWARMUP, num_samples=NSAMPLES, num_chains=NCHAINS,
                 chain_method=CHAIN_METHOD, progress_bar=False)
    mcmc2.run(jax.random.PRNGKey(1), obs=profiles)
    mcmc2.get_samples()["mu_M"].block_until_ready()
    t_sample = time.time() - t1
    t_compile = max(t_total - t_sample, 0.0)

    # diagnostics from the grouped (per-chain) samples
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = nps_summary({k: np.array(grouped[k]) for k in HYPER}, prob=0.9)
    post = mcmc.get_samples()
    rhat_max = max(float(diag[k]["r_hat"]) for k in HYPER)
    ess_min = min(float(diag[k]["n_eff"]) for k in HYPER)
    ndiv = int(mcmc.get_extra_fields().get("diverging", np.array([])).sum()) if "diverging" in mcmc.get_extra_fields() else None

    rec = dict(
        obs_id=obs_id, n_experiment_dirs=len(obs_to_dirs[obs_id]),
        N_c=int(profiles.shape[0]), nwarmup=NWARMUP, nsamples=NSAMPLES, nchains=NCHAINS,
        t_total_s=round(t_total, 2), t_sample_s=round(t_sample, 2), t_compile_s=round(t_compile, 2),
        rhat_max=round(rhat_max, 4), ess_min=round(ess_min, 1),
        mu_M=round(float(post["mu_M"].mean()), 4),   sig_M=round(float(post["sig_M"].mean()), 4),
        c0=round(float(post["c0"].mean()), 4),       sig_c=round(float(post["sig_c"].mean()), 4),
        beta=round(float(post["beta"].mean()), 4),
        true_mu_logM=round(float(true_mc[:, 0].mean()), 4), true_sig_logM=round(float(true_mc[:, 0].std()), 4),
        true_mu_c=round(float(true_mc[:, 1].mean()), 4),    true_sig_c=round(float(true_mc[:, 1].std()), 4),
    )
    # save full posterior for plotting
    with open(f"{OUT_DIR}/hier_post_{obs_id}.pkl", "wb") as f:
        pickle.dump({"samples": {k: np.array(post[k]) for k in HYPER}, "record": rec,
                     "true_mc": true_mc}, f)
    return rec


if __name__ == "__main__":
    obs_ids = sorted(obs_to_dirs)
    print(f"{len(exp_dirs)} experiment dirs -> {len(obs_ids)} unique obs sets; "
          f"{NCHAINS} chains x ({NWARMUP}+{NSAMPLES}) each\n")
    records = []
    grand_t0 = time.time()
    for i, obs_id in enumerate(obs_ids, 1):
        print(f"[{i}/{len(obs_ids)}] {obs_id} ...", flush=True)
        rec = run_one(obs_id)
        records.append(rec)
        print(f"    t_total={rec['t_total_s']}s (compile {rec['t_compile_s']}s + sample {rec['t_sample_s']}s) "
              f"| rhat_max={rec['rhat_max']} ess_min={rec['ess_min']}")
        print(f"    sig_M={rec['sig_M']} (true {rec['true_sig_logM']}) | "
              f"c0={rec['c0']} (true mu_c {rec['true_mu_c']})")
    total = time.time() - grand_t0

    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    print(f"\nTotal wall time for {len(obs_ids)} unique HMC fits (x2 for compile timing): {total/60:.1f} min")
    print(f"Wrote {CSV}")
