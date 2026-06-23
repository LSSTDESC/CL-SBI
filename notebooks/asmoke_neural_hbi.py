"""
A-smoke: go/no-go gate for fully-amortized neural hierarchical SBI (architecture A).

Question this answers (and ONLY this): does a permutation-invariant deep-set embedding + SNPE
recover the 5 population hyperparameters (mu_M, sig_M, c0, beta, sig_c) with CALIBRATED coverage on
EASY data? If the SBC rank histograms are ~uniform, the architecture works and we scale to A-full
(broad hyperpriors + all nuisance axes, N_c=376, ~18hr sims). If not, we fall back to the explicit
HMC-HBI result already in the paper, having spent ~half a day.

Deliberately easy (this is a smoke test, not the science run):
  - narrow hyperpriors, IN-DISTRIBUTION only (baseline regime, child18, no nuisance variation)
  - small stacks (N_c=50) so each training example is cheap
  - mean-pool deep-set embedding (simplest permutation-invariant choice)

Each training example is ONE STACK:
  theta = (mu_M, sig_M, c0, beta, sig_c)                       [5 hyperparameters]
  x     = N_c profiles, each 30 log10-Sigma bins + fixed noise [the "set" we embed]
The deep-set embedding phi->mean-pool->rho makes the network insensitive to cluster ordering.

Run (base env has the sbi/torch/jax/numpyro/colossus stack):
  /Users/akumgill/anaconda3/bin/python notebooks/asmoke_neural_hbi.py
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
# Config -- EASY / in-distribution (baseline regime; cf. configs/inference/infer_z1.json)
# ---------------------------------------------------------------------------
Z = 0.275
RBINS = 10 ** np.arange(0, 3.0, 0.1)          # 30 bins, matches gen_observations.py
NBINS = len(RBINS)
N_C = 50                                       # small stack for the smoke test
NOISE_DEX = 0.3                                # fixed (in-distribution); A-full varies this

# Narrow hyperpriors centered on the baseline truth (mu_M~14.4, sig_M~0.12, c0~4.6, beta~-0.85,
# sig_c~0.18). Intentionally tight: the gate is "does it calibrate on easy data," not "is it robust."
HYPER_NAMES = ["mu_M", "sig_M", "c0", "beta", "sig_c"]
HYPER_LO = np.array([14.20, 0.06, 4.0, -1.4, 0.08])
HYPER_HI = np.array([14.60, 0.20, 5.2, -0.3, 0.30])

N_TRAIN = 4000        # training stacks (each = N_C profiles); ~minutes of sim at ~17ms/pair
N_SBC = 300           # held-out datasets for simulation-based calibration
N_POST = 1000         # posterior samples per SBC dataset (rank statistic)


# ---------------------------------------------------------------------------
# Stack simulator: hyperparameters -> one stack of N_C noisy log10 profiles
# ---------------------------------------------------------------------------
def simulate_stack(hyper, n_c=N_C, noise=NOISE_DEX, rng=None):
    """One population draw. hyper=(mu_M,sig_M,c0,beta,sig_c) -> (n_c, NBINS) log10-Sigma stack.

    Gaussian population in log10M; linear M-c relation c = c0 + beta*(logM-mu_M) + N(0,sig_c).
    This mirrors hier_model() in hierarchical_mcmc_poc.py (the explicit-HMC generative model),
    so a pass here means A's network matches the same population the HMC paper uses.
    """
    rng = rng or np.random.default_rng()
    mu_M, sig_M, c0, beta, sig_c = hyper
    logM = rng.normal(mu_M, sig_M, n_c)
    c = c0 + beta * (logM - mu_M) + rng.normal(0, sig_c, n_c)
    c = np.clip(c, 2.0, 8.0)                    # keep inside the forward model's valid range
    prof = np.array([np.log10(wlprofile.simulate_nfw(m, ci, RBINS, Z)) for m, ci in zip(logM, c)])
    prof = prof + rng.normal(0, noise, prof.shape)
    return prof.astype(np.float32)              # (n_c, NBINS)


def make_training_set(n, rng):
    """Returns theta (n,5) and x (n, N_C*NBINS) flattened stacks."""
    theta = rng.uniform(HYPER_LO, HYPER_HI, size=(n, 5)).astype(np.float32)
    x = np.empty((n, N_C * NBINS), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        x[i] = simulate_stack(theta[i], rng=rng).reshape(-1)
        if (i + 1) % 500 == 0:
            print(f"  simulated {i+1}/{n} stacks ({(time.time()-t0)/(i+1)*1e3:.0f} ms/stack)", flush=True)
    return theta, x


# ---------------------------------------------------------------------------
# Permutation-invariant deep-set embedding: phi (per-profile) -> mean-pool -> rho
# ---------------------------------------------------------------------------
class DeepSetEmbedding(nn.Module):
    """Maps a flattened stack (N_C*NBINS,) to a fixed embedding, invariant to cluster ordering.

    phi: per-profile MLP (NBINS -> h);  aggregate by MEAN over the N_C clusters;  rho: MLP (h -> out).
    Mean-pool is the simplest permutation-invariant aggregator (the smoke-test choice); A-full can
    swap in attention/deep-sets-with-sum if mean-pool under-calibrates.
    """
    def __init__(self, n_c=N_C, nbins=NBINS, h=64, out=32):
        super().__init__()
        self.n_c, self.nbins = n_c, nbins
        self.phi = nn.Sequential(nn.Linear(nbins, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, x):
        # x: (batch, N_C*NBINS) -> (batch, N_C, NBINS)
        b = x.shape[0]
        x = x.view(b, self.n_c, self.nbins)
        h = self.phi(x)                 # (b, N_C, h)
        h = h.mean(dim=1)               # mean-pool over clusters -> (b, h)  [permutation-invariant]
        return self.rho(h)              # (b, out)


# ---------------------------------------------------------------------------
# Train SNPE with the deep-set embedding
# ---------------------------------------------------------------------------
def train(theta, x):
    prior = BoxUniform(torch.as_tensor(HYPER_LO, dtype=torch.float32),
                       torch.as_tensor(HYPER_HI, dtype=torch.float32))
    embedding = DeepSetEmbedding()
    estimator = posterior_nn(model="maf", embedding_net=embedding, hidden_features=50, num_transforms=5)
    inferrer = SNPE(prior, density_estimator=estimator, device="cpu")
    inferrer = inferrer.append_simulations(torch.as_tensor(theta), torch.as_tensor(x))
    print("## training SNPE + deep-set embedding ...", flush=True)
    t0 = time.time()
    de = inferrer.train(training_batch_size=50, stop_after_epochs=20, show_train_summary=False)
    post = inferrer.build_posterior(de)
    print(f"## trained in {time.time()-t0:.0f}s", flush=True)
    return post


# ---------------------------------------------------------------------------
# Simulation-based calibration: the actual gate
# ---------------------------------------------------------------------------
def sbc(post, rng, n=N_SBC, n_post=N_POST):
    """For each held-out (theta*, x*), rank theta* among posterior samples, per dimension.
    Calibrated <=> ranks are uniform on [0, n_post]. Returns ranks (n,5)."""
    print(f"## SBC over {n} held-out datasets ...", flush=True)
    ranks = np.empty((n, 5), dtype=int)
    t0 = time.time()
    for i in range(n):
        th = rng.uniform(HYPER_LO, HYPER_HI).astype(np.float32)
        x = simulate_stack(th, rng=rng).reshape(-1)
        s = post.sample((n_post,), x=torch.as_tensor(x), show_progress_bars=False).numpy()
        ranks[i] = (s < th[None, :]).sum(axis=0)     # rank of truth among samples, per dim
        if (i + 1) % 50 == 0:
            print(f"  sbc {i+1}/{n} ({(time.time()-t0)/(i+1)*1e3:.0f} ms/ea)", flush=True)
    return ranks


def ks_uniform(ranks, n_post=N_POST):
    """KS distance of each dimension's rank distribution from Uniform[0,1]. Small => calibrated."""
    from scipy.stats import kstest
    out = {}
    for d, nm in enumerate(HYPER_NAMES):
        u = (ranks[:, d] + 0.5) / (n_post + 1)
        out[nm] = float(kstest(u, "uniform").statistic)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t_start = time.time()

    print(f"=== A-smoke: N_C={N_C}, N_TRAIN={N_TRAIN}, noise={NOISE_DEX}, in-distribution ===", flush=True)
    print(f"hyperprior box: {dict(zip(HYPER_NAMES, zip(HYPER_LO, HYPER_HI)))}", flush=True)

    theta, x = make_training_set(N_TRAIN, rng)
    post = train(theta, x)

    # quick point-accuracy check on one fresh baseline-truth stack
    th_true = np.array([14.40, 0.12, 4.60, -0.85, 0.18], dtype=np.float32)
    x_true = simulate_stack(th_true, rng=rng).reshape(-1)
    s = post.sample((2000,), x=torch.as_tensor(x_true), show_progress_bars=False).numpy()
    print("\n## point check on baseline-truth stack:")
    for d, nm in enumerate(HYPER_NAMES):
        print(f"  {nm:6s} = {s[:,d].mean():.4f} +/- {s[:,d].std():.4f}   (true {th_true[d]:.4f})")

    ranks = sbc(post, rng)
    ks = ks_uniform(ranks)
    print("\n## SBC KS-from-uniform (smaller = better calibrated; <~0.1 is a pass at N_SBC=300):")
    for nm in HYPER_NAMES:
        flag = "PASS" if ks[nm] < 0.1 else "CHECK"
        print(f"  {nm:6s} KS={ks[nm]:.3f}  [{flag}]")

    gate = "GO (architecture calibrates -> proceed to A-full)" if max(ks.values()) < 0.12 \
        else "NO-GO (miscalibrated -> inspect before scaling; HMC-HBI is the fallback)"
    print(f"\n=== GATE: {gate} ===")
    print(f"=== total wall {time.time()-t_start:.0f}s ===")

    pickle.dump({"ranks": ranks, "ks": ks, "point_samples": s, "th_true": th_true,
                 "hyper_lo": HYPER_LO, "hyper_hi": HYPER_HI, "N_C": N_C, "N_TRAIN": N_TRAIN},
                open(f"{OUT}/asmoke_neural_hbi.pkl", "wb"))
    print(f"saved -> {OUT}/asmoke_neural_hbi.pkl")
