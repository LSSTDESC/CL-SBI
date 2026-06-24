"""
Hybrid SBI-HMC across all 8 experiments.

This is the per-cluster-SBI + NUTS-population approach (the "Hybrid SBI-HMC" method): train ONE
amortized per-cluster posterior q(M,c | profile, noise_dex), evaluate it on each experiment's 376
observed profiles, then run a NumPyro/NUTS population hierarchy via sample-reweighting (logsumexp over
each cluster's posterior samples -- non-Gaussian-safe, the better POC variant). Contrast with the
fully-amortized Hierarchical SBI (afull_neural_hbi.py), which maps the whole stack to hyperparameters
in one forward pass with no per-dataset MCMC.

To be a fair cross-experiment comparison (matching the Hierarchical SBI setup):
 - per-cluster net is CONDITIONED ON noise_dex (so high_noise / high_rm are not mis-specified),
 - trained over the WIDE box logM in [12,15.2], c in [2,8] (high_rm reaches logM~12.2, c~7.8).

Saves per-experiment relation params (for the aggregate overlay) AND the per-cluster sample tensors
(for PPCs). Run under base env:
  /Users/akumgill/anaconda3/bin/python notebooks/hybrid_sbi_hmc_allexp.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle, numpy as np, torch, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary
import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"; sys.path.insert(0, REPO)
from weaklensclustersbi.simulations import wlprofile
from weaklensclustersbi.inference import sbi_
from sbi.inference import SNPE
from sbi.utils import BoxUniform, posterior_nn
jax.config.update("jax_enable_x64", True)

OUT = f"{REPO}/notebooks/hierarchical_poc_outputs"
Z = 0.275; RB = 10 ** np.arange(0, 3.0, 0.1); NBINS = len(RB)
# wide per-cluster box (matches Hierarchical SBI coverage)
BOX_LO = np.array([12.0, 2.0]); BOX_HI = np.array([15.2, 8.0])
NOISE_LO, NOISE_HI = 0.10, 0.80
N_TRAIN = 150000; N_SAMP = 300        # large per-cluster training set (wide box needs the resolution)
CHILD18_BETA = -0.855
GATE_MUM = (14.20, 14.55)             # baseline mu_M must land here, else the per-cluster net is broken
EXPERIMENTS = ["obs_z1_lambda5", "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_rm_scatter",
               "obs_z1_lambda5_high_noise", "obs_z1_lambda5_ludlow", "obs_z1_lambda5_prada",
               "obs_z1_lambda5_low_richness_contam", "obs_z1_lambda5_high_richness_contam"]
HNAMES = ["mu_M", "sig_M", "c0", "beta", "sig_c"]


# ---- per-cluster net q(M,c | profile, noise_dex): profile(30) + noise(1) -> (logM,c) ----
def simulate_single(theta, noise, rng):
    p = np.log10(wlprofile.simulate_nfw(float(theta[0]), float(theta[1]), rbins=RB, z=Z))
    return (p + rng.normal(0, noise, size=p.shape)).astype(np.float32)


def train_per_cluster():
    cache = f"{OUT}/hybrid_per_cluster_post.pkl"
    if os.path.exists(cache):
        print("## loading cached per-cluster posterior"); return pickle.load(open(cache, "rb"))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    theta = rng.uniform(BOX_LO, BOX_HI, size=(N_TRAIN, 2)).astype(np.float32)
    noise = rng.uniform(NOISE_LO, NOISE_HI, size=N_TRAIN).astype(np.float32)
    x = np.empty((N_TRAIN, NBINS + 1), dtype=np.float32)
    print(f"## simulating {N_TRAIN} single-cluster training pairs ...", flush=True)
    for i in range(N_TRAIN):
        x[i, :-1] = simulate_single(theta[i], noise[i], rng); x[i, -1] = noise[i]
    prior = BoxUniform(torch.as_tensor(BOX_LO, dtype=torch.float32),
                       torch.as_tensor(BOX_HI, dtype=torch.float32))
    # larger flow (more transforms + hidden units) to resolve (M,c) over the wide box + variable noise
    inf = SNPE(prior, density_estimator=posterior_nn(model="maf", hidden_features=80, num_transforms=8),
               device="cpu")
    inf = inf.append_simulations(torch.as_tensor(theta), torch.as_tensor(x))
    print("## training per-cluster q(M,c | profile, noise) ...", flush=True)
    de = inf.train(training_batch_size=300, stop_after_epochs=25, show_train_summary=False)
    post = inf.build_posterior(de)
    pickle.dump(post, open(cache, "wb"))
    return post


def per_cluster_samples(post, obs_id):
    """Evaluate the per-cluster net on each of the 376 observed profiles -> (376, N_SAMP, 2)."""
    d = f"{REPO}/outputs/observations/{obs_id}.376"
    prof = np.load(f"{d}/drawn_nfw_profiles.npy").astype(np.float32)
    noise = float(np.load(f"{d}/sigmas.npy").mean())
    S = []
    for p in prof:
        xc = torch.as_tensor(np.concatenate([p, [noise]]).astype(np.float32))
        S.append(post.sample((N_SAMP,), x=xc, show_progress_bars=False).numpy())
    return np.array(S), noise


# ---- NUTS population hierarchy via sample-reweighting (logsumexp) ----
def run_hier(samples, seed=0):
    S = jnp.array(samples); logM_s = S[:, :, 0]; c_s = S[:, :, 1]
    def model():
        mu_M = numpyro.sample("mu_M", dist.Normal(14.2, 0.6))
        sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.6))
        c0 = numpyro.sample("c0", dist.Normal(4.8, 1.2))
        beta = numpyro.sample("beta", dist.Normal(CHILD18_BETA, 0.8))
        sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.6))
        lp_M = -0.5 * ((logM_s - mu_M) / sig_M) ** 2 - jnp.log(sig_M)
        lp_c = -0.5 * ((c_s - (c0 + beta * (logM_s - mu_M))) / sig_c) ** 2 - jnp.log(sig_c)
        logLj = jax.scipy.special.logsumexp(lp_M + lp_c, axis=1) - jnp.log(S.shape[1])
        numpyro.factor("loglik", jnp.sum(logLj))
    mc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median),
              num_warmup=800, num_samples=1000, num_chains=2, chain_method="sequential", progress_bar=False)
    t = time.time(); mc.run(jax.random.PRNGKey(seed)); mc.get_samples()["mu_M"].block_until_ready()
    t = time.time() - t
    p = mc.get_samples(); g = mc.get_samples(group_by_chain=True)
    diag = nps_summary({k: np.array(g[k]) for k in HNAMES}, prob=0.9)
    rhat = max(float(diag[k]["r_hat"]) for k in diag)
    est = {k: (float(p[k].mean()), float(p[k].std())) for k in HNAMES}
    return est, rhat, t


if __name__ == "__main__":
    t0 = time.time()
    post = train_per_cluster()

    # SANITY GATE: baseline mu_M must be ~correct, else the per-cluster net is mis-resolved over the
    # wide box and the whole comparison is untrustworthy (see the 30k-sim run that biased mu_M ~0.75 low).
    S0, _ = per_cluster_samples(post, "obs_z1_lambda5")
    est0, rhat0, _ = run_hier(S0)
    mu0 = est0["mu_M"][0]
    print(f"\n## GATE: baseline mu_M = {mu0:.3f} (must be in {GATE_MUM}; true ~14.376)", flush=True)
    if not (GATE_MUM[0] <= mu0 <= GATE_MUM[1]):
        print(f"## GATE FAILED -- per-cluster net still mis-resolved (mu_M={mu0:.3f}). "
              f"Saving diagnostic but NOT trusting the all-8 Hybrid overlay.", flush=True)

    rows = {}
    for e in EXPERIMENTS:
        S, noise = per_cluster_samples(post, e)
        est, rhat, t = run_hier(S)
        true_mc = np.load(f"{REPO}/outputs/observations/{e}.376/drawn_mc_pairs.npy")
        lm, c = true_mc[:, 0], true_mc[:, 1]
        bt = float(np.polyfit(lm - lm.mean(), c, 1)[0])
        truth = dict(mu_M=float(lm.mean()), sig_M=float(lm.std()), c0=float(c.mean()),
                     beta=bt, sig_c=float((c - (c.mean() + bt * (lm - lm.mean()))).std()))
        rows[e] = dict(est=est, truth=truth, noise=noise, rhat=rhat, t_sample=t, samples=S)
        print(f"\n  {e} (noise={noise:.3f}, rhat={rhat:.3f}, {t:.0f}s)")
        for k in HNAMES:
            print(f"    {k:6s} = {est[k][0]:7.3f} +/- {est[k][1]:.3f}  (true {truth[k]:7.3f})")
    pickle.dump({"experiments": rows, "names": HNAMES, "n_train": N_TRAIN},
                open(f"{OUT}/hybrid_sbi_hmc_allexp.pkl", "wb"))
    print(f"\n=== total wall {(time.time()-t0)/60:.1f} min ===")
    print(f"saved -> {OUT}/hybrid_sbi_hmc_allexp.pkl")
