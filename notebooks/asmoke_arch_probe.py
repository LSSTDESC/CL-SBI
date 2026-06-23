"""
A-smoke architecture probe: can a better SET AGGREGATOR calibrate beta (the M-c slope)?

Motivation: the N_TRAIN sweep showed beta's SBC miscalibration is flat in sim count (0.096->0.109
from 4k->16k) -> NOT under-training. beta is the slope of c vs logM, identified by how concentration
CO-VARIES with mass ACROSS clusters. The mean-pool deep-set embedding averages clusters together,
washing out exactly that inter-cluster spread. This probe swaps the aggregator and re-checks beta.

Variants (all on the SAME cached 16k pool, SAME SBC seed -> only the architecture changes):
  - mean   : baseline (reproduces the sweep's 16k row)
  - sum    : sum-pool (preserves total inter-cluster signal instead of averaging it away)
  - moment : concat [mean, std] across clusters (the std channel carries the spread beta needs)
  - moment_big : moment-pool + larger embedding (out 32->64) and more MAF transforms (5->8)

Pass target: beta KS < ~0.08 for some variant -> lock it into A-full. If all ~0.11 -> slope is an
information limit; greenlight A-full and report beta as mildly over-confident.

Run under base env:
  /Users/akumgill/anaconda3/bin/python notebooks/asmoke_arch_probe.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle
import numpy as np
import torch
import torch.nn as nn

import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"
sys.path.insert(0, REPO)
import notebooks.asmoke_neural_hbi as A
from sbi.inference import SNPE
from sbi.utils import BoxUniform, posterior_nn

OUT = A.OUT
HYPER_NAMES = A.HYPER_NAMES


class PoolEmbedding(nn.Module):
    """Deep-set embedding with selectable aggregator over the N_C clusters.

    agg='mean' : mean-pool (baseline)
    agg='sum'  : sum-pool
    agg='moment': concat [mean, std] over clusters before rho (doubles the pooled width)
    """
    def __init__(self, agg="mean", n_c=A.N_C, nbins=A.NBINS, h=64, out=32):
        super().__init__()
        self.agg, self.n_c, self.nbins = agg, n_c, nbins
        self.phi = nn.Sequential(nn.Linear(nbins, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        pooled = 2 * h if agg == "moment" else h
        self.rho = nn.Sequential(nn.Linear(pooled, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, self.n_c, self.nbins)
        h = self.phi(x)                              # (b, N_C, h)
        if self.agg == "mean":
            p = h.mean(dim=1)
        elif self.agg == "sum":
            p = h.sum(dim=1)
        elif self.agg == "moment":
            p = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1)  # spread channel for the slope
        else:
            raise ValueError(self.agg)
        return self.rho(p)


def train_variant(theta, x, agg, h=64, out=32, num_transforms=5):
    prior = BoxUniform(torch.as_tensor(A.HYPER_LO, dtype=torch.float32),
                       torch.as_tensor(A.HYPER_HI, dtype=torch.float32))
    emb = PoolEmbedding(agg=agg, h=h, out=out)
    estimator = posterior_nn(model="maf", embedding_net=emb, hidden_features=50,
                             num_transforms=num_transforms)
    inf = SNPE(prior, density_estimator=estimator, device="cpu")
    inf = inf.append_simulations(torch.as_tensor(theta), torch.as_tensor(x))
    t0 = time.time()
    de = inf.train(training_batch_size=50, stop_after_epochs=20, show_train_summary=False)
    post = inf.build_posterior(de)
    return post, time.time() - t0


VARIANTS = [
    ("mean",       dict(agg="mean",   h=64, out=32, num_transforms=5)),
    ("sum",        dict(agg="sum",    h=64, out=32, num_transforms=5)),
    ("moment",     dict(agg="moment", h=64, out=32, num_transforms=5)),
    ("moment_big", dict(agg="moment", h=64, out=64, num_transforms=8)),
]

if __name__ == "__main__":
    t_start = time.time()
    # Regenerate the identical 16k pool (same seed as the sweep -> same data).
    rng = np.random.default_rng(0)
    N = 16000
    print(f"## regenerating cached pool: {N} stacks (seed 0, identical to sweep) ...", flush=True)
    theta, x = A.make_training_set(N, rng)

    results = {}
    for name, kw in VARIANTS:
        print(f"\n##### variant={name}  {kw} #####", flush=True)
        torch.manual_seed(0)
        post, t_train = train_variant(theta, x, **kw)
        sbc_rng = np.random.default_rng(12345)        # SAME held-out SBC datasets as the sweep
        ranks = A.sbc(post, sbc_rng, n=A.N_SBC, n_post=A.N_POST)
        ks = A.ks_uniform(ranks)
        results[name] = {"ks": ks, "ranks": ranks, "t_train": t_train, "kw": kw}
        print(f"  ({t_train:.0f}s) KS: " + "  ".join(f"{nm}={ks[nm]:.3f}" for nm in HYPER_NAMES), flush=True)

    print("\n=== ARCH PROBE SUMMARY (KS-from-uniform; <0.1 clean pass; N_SBC=300 -> alpha=0.01 crit ~0.094) ===")
    hdr = f"{'variant':>11s}  " + "  ".join(f"{nm:>7s}" for nm in HYPER_NAMES)
    print(hdr); print("-" * len(hdr))
    for name, _ in VARIANTS:
        ks = results[name]["ks"]
        print(f"{name:>11s}  " + "  ".join(f"{ks[nm]:>7.3f}" for nm in HYPER_NAMES))

    base_beta = results["mean"]["ks"]["beta"]
    best = min(results, key=lambda n: results[n]["ks"]["beta"])
    bb = results[best]["ks"]["beta"]
    print(f"\n=== beta: baseline mean-pool {base_beta:.3f}  ->  best '{best}' {bb:.3f} ===")
    if bb < 0.08:
        print(f"=> LOCK '{best}' into A-full (beta now calibrated).")
    else:
        print("=> No aggregator gets beta < 0.08: likely a slope information-limit. Greenlight A-full")
        print("   with the cleanest variant and report beta as mildly over-confident/weakly-identified.")
    print(f"\n=== total wall {time.time()-t_start:.0f}s ===")
    pickle.dump({"results": results, "variants": [v[0] for v in VARIANTS], "names": HYPER_NAMES},
                open(f"{OUT}/asmoke_arch_probe.pkl", "wb"))
    print(f"saved -> {OUT}/asmoke_arch_probe.pkl")
