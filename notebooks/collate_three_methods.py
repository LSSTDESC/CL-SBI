"""
Collate population (M,c) estimates from all three methods across the 16 experiment configs:
  - MCMC joint-likelihood FTJ : posterior mean/std of the (freshly regenerated, converged) emcee chain
  - SBI FTJ                   : 2D-Gaussian summary from the percentile chain
  - hierarchical HMC          : from hier_all_experiments.csv (keyed by obs set)
plus a convergence flag for MCMC. Writes _all_methods_rows.pkl (consumed by the summary/PPC figs).
"""
import os, sys, pickle, csv
import numpy as np
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/plot")
import plotutils as P

INF = f"{REPO}/outputs/inference"; OBS = f"{REPO}/outputs/observations"; OUT = f"{REPO}/notebooks/hierarchical_poc_outputs"
hmc = {r["obs_id"]: r for r in csv.DictReader(open(f"{OUT}/hier_all_experiments.csv"))}

rows = []
for d in sorted(x for x in os.listdir(INF) if x.endswith("376")):
    sim, infer, obs, nsim, nobs = d.split(".")
    base = f"{INF}/{d}"
    true_mc = np.load(f"{OBS}/{obs}.376/drawn_mc_pairs.npy")
    r = dict(exp=d, obs=obs, sim=sim, infer=infer,
             true_muM=true_mc[:, 0].mean(), true_sigM=true_mc[:, 0].std(),
             true_muc=true_mc[:, 1].mean(), true_sigc=true_mc[:, 1].std())
    # MCMC joint FTJ from regenerated chain
    try:
        with open(f"{base}/mcmc_ftj_samplers.pickle", "rb") as f:
            j = pickle.load(f)
        ch = j.get_chain(); b = int(ch.shape[0] * 0.3); jf = ch[b:].reshape(-1, ch.shape[-1])
        r.update(mcmc_muM=jf[:, 0].mean(), mcmc_sigM=jf[:, 0].std(),
                 mcmc_muc=jf[:, 1].mean(), mcmc_sigc=jf[:, 1].std())
        # convergence: use stamped attrs if present, else heuristic
        conv = getattr(j, "converged", None)
        if conv is None:
            conv = (jf[:, 0].std() < 0.3) and (13.5 < jf[:, 0].mean() < 15.5)
        r["mcmc_converged"] = bool(conv)
        r["mcmc_steps"] = int(ch.shape[0])
        r["mcmc_rhat"] = float(getattr(j, "rhat_max", np.nan))
    except Exception as e:
        r["mcmc_muM"] = None; r["mcmc_converged"] = False
    # SBI FTJ
    try:
        with open(f"{base}/sbi_chains.pickle", "rb") as f:
            sbi = pickle.load(f)
        s = P.build_gaussian_summary_from_chain(sbi[1])
        r.update(sbi_muM=s["mass"]["mu"], sbi_sigM=s["mass"]["sigma"],
                 sbi_muc=s["concentration"]["mu"], sbi_sigc=s["concentration"]["sigma"])
    except Exception:
        r["sbi_muM"] = None
    # HMC by obs set
    h = hmc.get(obs)
    if h:
        r.update(hmc_muM=float(h["mu_M"]), hmc_sigM=float(h["sig_M"]),
                 hmc_muc=float(h["c0"]), hmc_sigc=float(h["sig_c"]))
    rows.append(r)

pickle.dump(rows, open(f"{OUT}/_all_methods_rows.pkl", "wb"))
nconv = sum(r.get("mcmc_converged", False) for r in rows)
print(f"collated {len(rows)} configs; MCMC converged: {nconv}/{len(rows)}")
print(f"{'experiment':52s} {'mcmc_conv':>9s} {'steps':>6s} sigM:true/mcmc/sbi/hmc")
for r in rows:
    print(f"{r['exp'][:52]:52s} {str(r.get('mcmc_converged')):>9s} {r.get('mcmc_steps','-'):>6} "
          f"{r['true_sigM']:.3f}/{r.get('mcmc_sigM',0) or 0:.3f}/{r.get('sbi_sigM',0) or 0:.3f}/{r.get('hmc_sigM',0) or 0:.3f}")
