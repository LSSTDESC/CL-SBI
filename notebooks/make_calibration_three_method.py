"""
Three-method calibration (coverage) plot for Paper 2: SBI / MCMC joint-likelihood / hierarchical HMC.
SBI + MCMC-FTJ coverage come from Paper 1's per-config pipeline pickle; HMC coverage is computed from
the saved hyperparameter posterior using the SAME compute_coverage routine (Mahalanobis CDF).
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, sys
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
})
sys.path.insert(0, "/Users/akumgill/Documents/GitHub/CL-SBI/scripts")
from plot_calibration import compute_coverage, CONF_LEVELS

REPO="/Users/akumgill/Documents/GitHub/CL-SBI"; OUT=f"{REPO}/notebooks/hierarchical_poc_outputs"
OBS="obs_z1_lambda5"
true_mc = np.load(f"{REPO}/outputs/observations/{OBS}.376/drawn_mc_pairs.npy")

# SBI + MCMC FTJ coverage from the pipeline pickle (baseline)
cal = pickle.load(open(f"{REPO}/outputs/plots/sim_z1.infer_z1.{OBS}.10000.376/calibration/calibration_ftj.pickle","rb"))
cov_sbi  = cal[("sbi","ftj")]
cov_mcmc = cal[("mcmc","ftj")]

# HMC: draw a population from the hyperparameter posterior, then same compute_coverage vs true clusters
h = pickle.load(open(f"{OUT}/hier_post_{OBS}.pkl","rb"))["samples"]
rng = np.random.default_rng(0); n = len(h["mu_M"]); idx = rng.integers(0,n,20000)
logM = h["mu_M"][idx] + h["sig_M"][idx]*rng.standard_normal(20000)
c    = h["c0"][idx] + h["beta"][idx]*(logM-h["mu_M"][idx]) + h["sig_c"][idx]*rng.standard_normal(20000)
hmc_samples = np.column_stack([logM, c])
cov_hmc = compute_coverage(hmc_samples, true_mc)

x = np.array(CONF_LEVELS)
def y(cov): return [cov[p] for p in CONF_LEVELS]
def dmax(cov): return max(abs(cov[p]-p) for p in CONF_LEVELS)

fig, ax = plt.subplots(figsize=(6.2,6))
ax.plot([0,1],[0,1],"k--",lw=1.5,label="perfect calibration")
ax.plot(x, y(cov_mcmc), color="#1f77b4", lw=2.2, marker="o", ms=3, label=f"MCMC joint-likelihood (FTJ), $\\Delta_{{\\max}}$={dmax(cov_mcmc):.2f}")
ax.plot(x, y(cov_sbi),  color="#ff7f0e", lw=2.2, marker="s", ms=3, label=f"SBI FTJ, $\\Delta_{{\\max}}$={dmax(cov_sbi):.2f}")
ax.plot(x, y(cov_hmc),  color="#2ca02c", lw=2.2, marker="^", ms=3, label=f"Hierarchical HMC (this work), $\\Delta_{{\\max}}$={dmax(cov_hmc):.2f}")
ax.set_xlabel("nominal credible level $p$"); ax.set_ylabel("empirical coverage")
ax.set_title("Calibration: baseline experiment ($N_c=376$)")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/calibration_three_method.png", dpi=150)
print(f"wrote calibration_three_method.png | Dmax: MCMC={dmax(cov_mcmc):.3f} SBI={dmax(cov_sbi):.3f} HMC={dmax(cov_hmc):.3f}")
