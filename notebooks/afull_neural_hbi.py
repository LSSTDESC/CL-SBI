"""
A-full: fully-amortized neural hierarchical SBI over the full nuisance envelope (Option 2: single
Gaussian population; the two richness-contamination experiments are deferred to a follow-up mixture
run and reported as a known limitation).

This is the science run the A-smoke gate + N_TRAIN sweep + architecture probe cleared:
  - moment-pool deep-set embedding (concat [mean, std] over clusters) -- the probe winner: it
    calibrates the slope beta (KS 0.109->0.078) and sig_c (0.083->0.043) that mean-pool left
    over-confident, because beta/sig_c are covariance/spread parameters needing the inter-cluster
    spread that mean-pool discards.
  - BROAD hyperpriors enveloping all 6 non-contam experiments (with padding so each sits inside,
    not at the edge of, the trained box), plus noise_dex as a varied per-stack nuisance fed to the
    embedding as a known input.
  - one network q_phi(mu_M, sig_M, c0, beta, sig_c | {profiles}, noise_dex); single forward pass at
    inference -> population posterior, zero per-dataset MCMC.

Cost: A draws (M,c) DIRECTLY from the population (no colossus call), so a stack is ~69 ms at N_c=376
-> ~30-60 min of sims for 30-50k training stacks (NOT the 18 hr the colossus-based estimate implied).

Run under the base env:
  /Users/akumgill/anaconda3/bin/python notebooks/afull_neural_hbi.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle
import numpy as np
import torch
import torch.nn as nn

import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"
sys.path.insert(0, REPO)
from weaklensclustersbi.simulations import wlprofile
from sbi.inference import SNPE
from sbi.utils import BoxUniform, posterior_nn

OUT = f"{REPO}/notebooks/hierarchical_poc_outputs"
os.makedirs(OUT, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
Z = 0.275
RBINS = 10 ** np.arange(0, 3.0, 0.1)
NBINS = len(RBINS)
N_C = 376                                    # full stack (matches the real obs sets)

# Hyperprior box: padded beyond the true range of the 6 NON-CONTAM experiments so each lands inside,
# not at the edge of, the trained envelope. (true ranges, measured: muM[13.21,14.38] sigM[0.118,0.535]
# c0[4.66,6.15] beta[-1.36,-0.23] sigc[0.134,0.713]; noise[0.307,0.703].)
HYPER_NAMES = ["mu_M", "sig_M", "c0", "beta", "sig_c"]
HYPER_LO = np.array([13.0, 0.08, 3.8, -1.8, 0.08])
HYPER_HI = np.array([14.7, 0.65, 6.8,  0.1, 0.80])
NOISE_LO, NOISE_HI = 0.10, 0.80              # noise_dex nuisance, fed to embedding as a known input

# The 6 experiments A-full is evaluated on (Option 2: contam excluded).
EXPERIMENTS = ["obs_z1_lambda5", "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_rm_scatter",
               "obs_z1_lambda5_high_noise", "obs_z1_lambda5_ludlow", "obs_z1_lambda5_prada"]

N_TRAIN = 40000        # ~46 min of sims at ~69 ms/stack
N_SBC = 500            # held-out datasets for SBC over the full envelope
N_POST = 1000


# ---------------------------------------------------------------------------
# Stack simulator: (hyperparameters, noise) -> stack of N_C noisy log10 profiles
# ---------------------------------------------------------------------------
def simulate_stack(hyper, noise, n_c=N_C, rng=None):
    """Gaussian population in log10M; c = c0 + beta*(logM-mu_M) + N(0,sig_c); add noise_dex scatter.
    Mirrors hier_model() in hierarchical_mcmc_poc.py (the explicit-HMC generative model)."""
    rng = rng or np.random.default_rng()
    mu_M, sig_M, c0, beta, sig_c = hyper
    logM = rng.normal(mu_M, sig_M, n_c)
    c = c0 + beta * (logM - mu_M) + rng.normal(0, sig_c, n_c)
    c = np.clip(c, 2.0, 8.0)
    logM = np.clip(logM, 12.0, 15.2)
    prof = np.array([np.log10(wlprofile.simulate_nfw(m, ci, RBINS, Z)) for m, ci in zip(logM, c)])
    prof = prof + rng.normal(0, noise, prof.shape)
    return prof.astype(np.float32)


def make_training_set(n, rng):
    """theta=(n,5) hyperparameters; x=(n, N_C*NBINS + 1): flattened stack with noise_dex appended."""
    theta = rng.uniform(HYPER_LO, HYPER_HI, size=(n, 5)).astype(np.float32)
    noise = rng.uniform(NOISE_LO, NOISE_HI, size=n).astype(np.float32)
    x = np.empty((n, N_C * NBINS + 1), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        x[i, :-1] = simulate_stack(theta[i], noise[i], rng=rng).reshape(-1)
        x[i, -1] = noise[i]                              # known per-stack nuisance
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  simulated {i+1}/{n} ({el/(i+1)*1e3:.0f} ms/stack, ETA {el/(i+1)*(n-i-1)/60:.0f} min)", flush=True)
    return theta, x


# ---------------------------------------------------------------------------
# Moment-pool deep-set embedding (the architecture-probe winner)
# Splits the trailing noise_dex scalar off x, embeds the profile set, concatenates noise back in.
# ---------------------------------------------------------------------------
class MomentPoolEmbedding(nn.Module):
    def __init__(self, n_c=N_C, nbins=NBINS, h=64, out=48):
        super().__init__()
        self.n_c, self.nbins = n_c, nbins
        self.phi = nn.Sequential(nn.Linear(nbins, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, out))
        self.out_dim = out + 1                            # + noise_dex passed through

    def forward(self, x):
        b = x.shape[0]
        noise = x[:, -1:]                                 # (b,1) known nuisance
        prof = x[:, :-1].view(b, self.n_c, self.nbins)
        h = self.phi(prof)                                # (b, N_C, h)
        pooled = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1)   # moment-pool: mean + spread
        emb = self.rho(pooled)                            # (b, out)
        return torch.cat([emb, noise], dim=-1)            # condition also on the known noise level


def train(theta, x):
    prior = BoxUniform(torch.as_tensor(HYPER_LO, dtype=torch.float32),
                       torch.as_tensor(HYPER_HI, dtype=torch.float32))
    emb = MomentPoolEmbedding()
    estimator = posterior_nn(model="maf", embedding_net=emb, hidden_features=60, num_transforms=6)
    inf = SNPE(prior, density_estimator=estimator, device="cpu")
    inf = inf.append_simulations(torch.as_tensor(theta), torch.as_tensor(x))
    print("## training SNPE + moment-pool embedding ...", flush=True)
    t0 = time.time()
    de = inf.train(training_batch_size=100, stop_after_epochs=25, show_train_summary=False)
    post = inf.build_posterior(de)
    print(f"## trained in {time.time()-t0:.0f}s", flush=True)
    return post


# ---------------------------------------------------------------------------
# SBC over the full envelope
# ---------------------------------------------------------------------------
def sbc(post, rng, n=N_SBC, n_post=N_POST):
    print(f"## SBC over {n} held-out datasets (full envelope) ...", flush=True)
    ranks = np.empty((n, 5), dtype=int)
    t0 = time.time()
    for i in range(n):
        th = rng.uniform(HYPER_LO, HYPER_HI).astype(np.float32)
        nz = float(rng.uniform(NOISE_LO, NOISE_HI))
        x = np.concatenate([simulate_stack(th, nz, rng=rng).reshape(-1), [nz]]).astype(np.float32)
        s = post.sample((n_post,), x=torch.as_tensor(x), show_progress_bars=False).numpy()
        ranks[i] = (s < th[None, :]).sum(axis=0)
        if (i + 1) % 100 == 0:
            print(f"  sbc {i+1}/{n} ({(time.time()-t0)/(i+1)*1e3:.0f} ms/ea)", flush=True)
    return ranks


def ks_uniform(ranks, n_post=N_POST):
    from scipy.stats import kstest
    return {nm: float(kstest((ranks[:, d] + 0.5) / (n_post + 1), "uniform").statistic)
            for d, nm in enumerate(HYPER_NAMES)}


# ---------------------------------------------------------------------------
# Apply to the 6 real observation stacks
# ---------------------------------------------------------------------------
def apply_to_experiments(post):
    rows = {}
    print("\n## applying A-full to the 6 real observation stacks ...", flush=True)
    for e in EXPERIMENTS:
        d = f"{REPO}/outputs/observations/{e}.376"
        prof = np.load(f"{d}/drawn_nfw_profiles.npy").astype(np.float32)   # (376,30) log10 Sigma
        true_mc = np.load(f"{d}/drawn_mc_pairs.npy")
        sig = np.load(f"{d}/sigmas.npy")
        noise = float(sig.mean())
        x = np.concatenate([prof.reshape(-1), [noise]]).astype(np.float32)
        s = post.sample((4000,), x=torch.as_tensor(x), show_progress_bars=False).numpy()
        lm, c = true_mc[:, 0], true_mc[:, 1]
        beta_true = float(np.polyfit(lm - lm.mean(), c, 1)[0])
        sigc_true = float((c - (c.mean() + beta_true * (lm - lm.mean()))).std())
        truth = dict(mu_M=float(lm.mean()), sig_M=float(lm.std()), c0=float(c.mean()),
                     beta=beta_true, sig_c=sigc_true)
        est = {nm: (float(s[:, k].mean()), float(s[:, k].std())) for k, nm in enumerate(HYPER_NAMES)}
        rows[e] = dict(truth=truth, est=est, noise=noise, samples=s)
        print(f"\n  {e}  (noise={noise:.3f})")
        for nm in HYPER_NAMES:
            m, sd = est[nm]
            print(f"    {nm:6s} = {m:7.3f} +/- {sd:.3f}   (true {truth[nm]:7.3f})")
    return rows


if __name__ == "__main__":
    t_start = time.time()
    rng = np.random.default_rng(0)
    print(f"=== A-full (Option 2: single Gaussian, 6 exps) | N_C={N_C}, N_TRAIN={N_TRAIN} ===", flush=True)
    print(f"hyperprior box: {dict(zip(HYPER_NAMES, zip(HYPER_LO, HYPER_HI)))}", flush=True)
    print(f"noise_dex ~ U({NOISE_LO},{NOISE_HI}) (fed to embedding)", flush=True)

    theta, x = make_training_set(N_TRAIN, rng)
    post = train(theta, x)

    ranks = sbc(post, rng)
    ks = ks_uniform(ranks)
    print("\n## SBC KS-from-uniform (full envelope; <0.1 pass; N_SBC=500 -> alpha=0.01 crit ~0.073):")
    for nm in HYPER_NAMES:
        print(f"  {nm:6s} KS={ks[nm]:.3f}  [{'PASS' if ks[nm] < 0.08 else 'CHECK'}]")

    rows = apply_to_experiments(post)

    pickle.dump({"ks": ks, "ranks": ranks, "experiments": rows,
                 "hyper_lo": HYPER_LO, "hyper_hi": HYPER_HI, "noise_range": (NOISE_LO, NOISE_HI),
                 "N_C": N_C, "N_TRAIN": N_TRAIN, "names": HYPER_NAMES},
                open(f"{OUT}/afull_neural_hbi.pkl", "wb"))
    print(f"\n=== total wall {(time.time()-t_start)/60:.1f} min ===")
    print(f"saved -> {OUT}/afull_neural_hbi.pkl")
