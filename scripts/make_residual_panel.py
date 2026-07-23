"""
P2-d fit-consistency diagnostic (Payerne): per-radial-bin residuals of the fitted profile
relative to the true (noiseless median) profile, in units of the stacked-profile
uncertainty: (log10 X_model - log10 X_true) / sigma_stack, for MCMC FTJ and SBI FTJ.
A 1-sigma-consistent fit should keep its 68% band within the +/-1 lines.
Usage: python3 make_residual_panel.py --observable delta_sigma [--obs_id obs_z1_lambda5 ...]
"""
import sys, os, json, pickle, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plot"))
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from weaklensclustersbi.simulations import wlprofile
from plot.plotutils import build_gaussian_summary_from_chain

ap = argparse.ArgumentParser()
ap.add_argument("--observable", default="delta_sigma")
ap.add_argument("--sim_id", default="sim_z1")
ap.add_argument("--infer_id", default="infer_z1")
ap.add_argument("--obs_id", default="obs_z1_lambda5")
args = ap.parse_args()
SD = os.path.dirname(__file__)
SUF = "" if args.observable == "surface_density" else f".{args.observable}"
plt.style.use(os.path.join(SD, "..", "plot", "mplstyle.txt"))

key = f"{args.sim_id}.{args.infer_id}.{args.obs_id}.10000.376{SUF}"
infer = os.path.join(SD, f"../outputs/inference/{key}")
obsd = os.path.join(SD, f"../outputs/observations/{args.obs_id}.376{SUF}")
cfg = json.load(open(os.path.join(SD, f"../configs/observations/{args.obs_id}.json")))
rbins = 10 ** np.arange(0, cfg["num_radial_bins"] / 10, 0.1)
z = (cfg["min_z"] + cfg["max_z"]) / 2

# true profile = median of the noiseless drawn profiles (log10 space, as stored)
true_prof = np.median(np.load(os.path.join(obsd, "noiseless_drawn_nfw_profiles.npy")), axis=0)
# stacked-profile uncertainty: per-bin log-sigma reduced by sqrt(N_c), median-penalty sqrt(pi/2)
sig = np.load(os.path.join(obsd, "sigmas.npy")) * np.sqrt(np.pi / 2) / np.sqrt(376)

rng = np.random.default_rng(0)


def model_profiles(theta_samples, n=150):
    idx = rng.choice(len(theta_samples), size=min(n, len(theta_samples)), replace=False)
    return np.array([np.log10(wlprofile.simulate_nfw(m, c, rbins, z, kind=args.observable))
                     for m, c in theta_samples[idx]])


# MCMC FTJ
mcmc = pickle.load(open(os.path.join(infer, "mcmc_ftj_samplers.pickle"), "rb")).flatchain[:, :2]
res_mcmc = (model_profiles(mcmc) - true_prof) / sig
# SBI FTJ (percentile chain -> gaussian -> samples)
sbi_chain = np.asarray(pickle.load(open(os.path.join(infer, "sbi_chains.pickle"), "rb"))[1])
s = build_gaussian_summary_from_chain(sbi_chain)
rho = s["correlation"]["rho"]; sm, sc = s["mass"]["sigma"], s["concentration"]["sigma"]
cov = [[sm**2, rho * sm * sc], [rho * sm * sc, sc**2]]
sbi = rng.multivariate_normal([s["mass"]["mu"], s["concentration"]["mu"]], cov, 150)
res_sbi = (model_profiles(sbi) - true_prof) / sig

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for res, color, label in ((res_mcmc, "#0077BB", "MCMC fit-then-join"),
                          (res_sbi, "#EE7733", "SBI fit-then-join")):
    med = np.median(res, axis=0)
    lo, hi = np.percentile(res, [16, 84], axis=0)
    ax.plot(rbins, med, color=color, lw=1.8, label=label)
    ax.fill_between(rbins, lo, hi, color=color, alpha=0.25)
ax.axhline(0, color="gray", ls=":", lw=1)
for yy in (-1, 1):
    ax.axhline(yy, color="gray", ls="--", lw=1, alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("radius [kpc/h]", fontsize="large")
lab = r"\Delta\Sigma" if args.observable == "delta_sigma" else r"\Sigma"
ax.set_ylabel(rf"$(\log_{{10}}{lab}_{{\rm model}} - \log_{{10}}{lab}_{{\rm true}})\,/\,\sigma_{{\rm stack}}$", fontsize="large")
ax.legend(fontsize="medium")
ax.set_title(f"Fit consistency, {args.obs_id.replace('obs_z1_','')} ({args.observable})", fontsize="medium")
fig.tight_layout()
outdir = os.path.join(SD, f"../outputs/plots/residual_consistency{SUF}")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, f"residual_{args.sim_id}.{args.obs_id}.{ext}"), dpi=120)
print(f"saved {outdir}/residual_{args.sim_id}.{args.obs_id}.pdf")
print(f"max |median residual|: MCMC {np.max(np.abs(np.median(res_mcmc,0))):.2f}sigma, SBI {np.max(np.abs(np.median(res_sbi,0))):.2f}sigma")
