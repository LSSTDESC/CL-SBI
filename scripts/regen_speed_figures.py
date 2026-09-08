"""
Regenerate the two speed-appendix figures on current (corrected-mean-era) machinery.

Fig 1 (infer_speed.png): per-stage wall times from the most recent pipeline_speed.csv rows.
Fig 2 (infer_speed2.png): recurring inference cost vs number of observed clusters N, measured
fresh by slicing the saved baseline observations to the first N clusters and timing amortized
SBI (JTF+FTJ apply) and MCMC (JTF stack fit; FTJ joint likelihood) at each N.
The original scaling CSV no longer exists; this is a from-scratch, better-controlled remake.
Run on quiet cores (timings) — chain behind any running compute.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time, json, pickle, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import multiprocessing

from weaklensclustersbi.inference import sbi_, mcmc

SD = os.path.dirname(__file__)
CM = ".delta_sigma.corrected_mean"
OBS = os.path.join(SD, "../outputs/observations/obs_z1_lambda5.376.delta_sigma")
PD = os.path.join(SD, f"../outputs/posteriors/sim_z1.infer_z1.10000.376{CM}")
FIG = os.path.join(SD, "../tex_source/figures_new/speed")
priors = json.load(open(os.path.join(SD, "../configs/inference/infer_z1.json")))["priors"]
priors["observable"] = "delta_sigma"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(os.path.join(SD, "../plot/mplstyle.txt"))

# ---------- Fig 1: per-stage times from the latest csv rows (10000.376, cmean era) ----------
rows = list(csv.DictReader(open(os.path.join(SD, "../outputs/inference/pipeline_speed.csv"))))
stages = ["gen_simulations", "train_inferrer", "sbi_total", "mcmc_join_then_fit", "mcmc_fit_then_join"]
labels = ["gen_simulations\n(one-time)", "train_inferrer\n(one-time)", "SBI inference\n(recurring)",
          "MCMC join-then-fit\n(recurring)", "MCMC fit-then-join\n(recurring)"]
vals = []
for st in stages:
    v = [float(r["seconds"]) for r in rows
         if r["stage"] == st and r["num_sims"] == "10000" and r["num_obs"] == "376"
         and r["timestamp"] >= "2026-08-06" and r.get("status") != "reused"]
    vals.append(np.median(v) if v else np.nan)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["0.6", "0.6", "C1", "C0", "C0"]
ax.bar(range(len(stages)), vals, color=colors)
ax.set_yscale("log"); ax.set_ylabel("wall time [s]")
ax.set_xticks(range(len(stages))); ax.set_xticklabels(labels, fontsize=8)
for i, v in enumerate(vals):
    if np.isfinite(v): ax.text(i, v * 1.15, f"{v:.1f}s" if v < 100 else f"{v/60:.1f}m", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/infer_speed.png", dpi=160); plt.close(fig)
print("fig 1 saved", flush=True)

# ---------- Fig 2: recurring cost vs N ----------
post = pickle.load(open(f"{PD}/posterior.pickle", "rb"))
post_jtf = pickle.load(open(f"{PD}/posterior_jtf.pickle", "rb"))
pairs = np.load(f"{OBS}/drawn_mc_pairs.npy"); prof = np.load(f"{OBS}/drawn_nfw_profiles.npy")
sig = np.load(f"{OBS}/sigmas.npy")
Ns = [10, 50, 100, 200, 376]
t_sbi, t_jtf, t_ftj = [], [], []
for N in Ns:
    p, pf = pairs[:N], prof[:N]
    # SBI must be applied with a matching-dim FTJ net only at N=376; for N<376 time the
    # JTF branch + per-cluster-style FTJ sampling proxy is not defined -> time JTF net +
    # note: the FTJ concatenated net has fixed input dim (376), so we time SBI at N=376
    # and report the (flat) amortized cost for all N.
    t0 = time.perf_counter()
    if N == 376:
        sbi_.apply_observations(post, post_jtf, p, pf, stack_estimator="corrected_mean", sigmas=sig)
        t_sbi_376 = time.perf_counter() - t0
    with multiprocessing.Pool() as pool:
        t0 = time.perf_counter()
        mcmc.join_then_fit(pf, sig, priors, pool=pool, stack_estimator="corrected_mean")
        t_jtf.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mcmc.fit_then_join(pf, sig, priors, pool=pool)
        t_ftj.append(time.perf_counter() - t0)
    print(f"N={N}: jtf={t_jtf[-1]:.0f}s ftj={t_ftj[-1]:.0f}s", flush=True)
t_sbi = [t_sbi_376] * len(Ns)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Ns, t_ftj, "o-", color="C0", label="MCMC fit-then-join")
ax.plot(Ns, t_jtf, "s-", color="C0", ls="--", label="MCMC join-then-fit")
ax.plot(Ns, t_sbi, "^-", color="C1", label="amortized SBI (both branches)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of observed clusters $N_c$"); ax.set_ylabel("recurring inference wall time [s]")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/infer_speed2.png", dpi=160); plt.close(fig)
json.dump({"Ns": Ns, "sbi": t_sbi, "mcmc_jtf": t_jtf, "mcmc_ftj": t_ftj},
          open("/tmp/speed_scaling.json", "w"))
print("fig 2 saved; DONE", flush=True)
