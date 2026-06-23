"""
A-smoke follow-up: N_TRAIN sweep at the SAME easy in-distribution settings.

Tests one hypothesis: are the borderline SBC ranks for (beta, sig_c) in the first A-smoke
(N_TRAIN=4000, KS 0.114 / 0.115, mild U-shape over-confidence) just under-training, or a real
ceiling of the mean-pool deep-set architecture?

If KS for beta/sig_c falls monotonically toward the clean-pass dims (mu_M/sig_M/c0 ~0.05-0.07) as
N_TRAIN grows, it is under-training -> A-full's large sim budget fixes it. If it plateaus high, the
aggregator needs more capacity (deep-set-with-sum/attention) before A-full.

Reuses simulate_stack / DeepSetEmbedding / train / sbc / ks_uniform from asmoke_neural_hbi.py so the
ONLY thing that changes is N_TRAIN. Same hyperprior box, same N_C=50, same noise=0.3, in-distribution.

Run under the base env:
  /Users/akumgill/anaconda3/bin/python notebooks/asmoke_ntrain_sweep.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle
import numpy as np
import torch

import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"
sys.path.insert(0, REPO)
import notebooks.asmoke_neural_hbi as A

OUT = A.OUT
N_TRAIN_GRID = [4000, 8000, 16000]   # 4000 reproduces the first run; 8k/16k test the trend
HYPER_NAMES = A.HYPER_NAMES

if __name__ == "__main__":
    t_start = time.time()
    results = {}
    # Reuse ONE big training pool so smaller N_TRAIN are strict subsets (clean comparison, less sim).
    rng = np.random.default_rng(0)
    n_max = max(N_TRAIN_GRID)
    print(f"=== N_TRAIN sweep {N_TRAIN_GRID} | N_C={A.N_C}, noise={A.NOISE_DEX}, in-distribution ===", flush=True)
    print(f"## simulating shared pool of {n_max} stacks ...", flush=True)
    theta_all, x_all = A.make_training_set(n_max, rng)

    for n in N_TRAIN_GRID:
        print(f"\n##### N_TRAIN={n} #####", flush=True)
        torch.manual_seed(0)                       # same net init across n for fair comparison
        post = A.train(theta_all[:n], x_all[:n])
        sbc_rng = np.random.default_rng(12345)     # SAME held-out SBC datasets across n
        ranks = A.sbc(post, sbc_rng, n=A.N_SBC, n_post=A.N_POST)
        ks = A.ks_uniform(ranks)
        results[n] = {"ks": ks, "ranks": ranks}
        print(f"  KS: " + "  ".join(f"{nm}={ks[nm]:.3f}" for nm in HYPER_NAMES), flush=True)

    print("\n=== SWEEP SUMMARY (KS-from-uniform; <0.1 clean pass) ===")
    hdr = "N_TRAIN  " + "  ".join(f"{nm:>7s}" for nm in HYPER_NAMES)
    print(hdr); print("-" * len(hdr))
    for n in N_TRAIN_GRID:
        ks = results[n]["ks"]
        print(f"{n:>7d}  " + "  ".join(f"{ks[nm]:>7.3f}" for nm in HYPER_NAMES))

    # trend verdict for the two watch-items
    def trend(nm):
        v = [results[n]["ks"][nm] for n in N_TRAIN_GRID]
        return v[0], v[-1], ("falling" if v[-1] < v[0] - 0.02 else "flat/rising")
    print("\n=== watch-items ===")
    for nm in ["beta", "sig_c"]:
        a, b, t = trend(nm)
        print(f"  {nm:6s}: {a:.3f} (4k) -> {b:.3f} (16k)  [{t}]")
    print(f"\n=== total wall {time.time()-t_start:.0f}s ===")
    pickle.dump({"grid": N_TRAIN_GRID, "results": results, "names": HYPER_NAMES},
                open(f"{OUT}/asmoke_ntrain_sweep.pkl", "wb"))
    print(f"saved -> {OUT}/asmoke_ntrain_sweep.pkl")
