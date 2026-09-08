"""
Radial-range refit (IR2/Zhou, Figs 3+7 comment): repeat the baseline inference using ONLY
the radial bins measurable in real stacked weak lensing (R > 200 kpc/h; 6 of 30 bins),
by slicing the saved 30-bin data vectors. Quantifies how much of the (M, c) constraint
in the fiducial setup originates at unmeasurable radii (cf. the Fisher split: 59% of the
mass information and 97% of the concentration information lie below the cut).

Produces: outer-range MCMC JTF/FTJ chains + SBI JTF/FTJ chains, a JSON summary comparing
full-range vs outer-only posteriors, and the appendix figure (full vs outer posteriors,
MCMC and SBI panels).
Slicing is exact for both methods: the likelihood factorizes over bins, and the per-bin
corrected-mean stack of a slice equals the slice of the stack.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import emcee

from weaklensclustersbi.simulations import wlprofile
from weaklensclustersbi.inference import sbi_, mcmcutils, stackutils

SD = os.path.dirname(__file__)
CM = ".delta_sigma.corrected_mean"
OBS = os.path.join(SD, "../outputs/observations/obs_z1_lambda5.376.delta_sigma")
SIM = os.path.join(SD, f"../outputs/simulations/sim_z1.10000.376{CM}")
INF_FULL = os.path.join(SD, f"../outputs/inference/sim_z1.infer_z1.obs_z1_lambda5.10000.376{CM}")
OUT = os.path.join(SD, f"../outputs/inference/radial_range_refit{CM}")
FIG = os.path.join(SD, "../tex_source/figures_new/radial_range")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

R_ALL = 10 ** np.arange(0, 3, 0.1)
OUTER = R_ALL > 200.0
R_OUT = R_ALL[OUTER]
print(f"outer bins: {OUTER.sum()}/30 (R > 200 kpc/h)", flush=True)

priors = json.load(open(os.path.join(SD, "../configs/inference/infer_z1.json")))["priors"]
priors["observable"] = "delta_sigma"
ZMID = (priors["min_z"] + priors["max_z"]) / 2

prof = np.load(f"{OBS}/drawn_nfw_profiles.npy")[:, OUTER]   # (376, 6) log10
sig = np.load(f"{OBS}/sigmas.npy")[OUTER]                    # (6,)

# ---------------- MCMC (custom outer-radius likelihood; vectorized FTJ) ----------------
def model_log_prof(m, c):
    return np.log10(wlprofile.simulate_nfw(m, c, R_OUT, ZMID, kind="delta_sigma"))

def logpost_jtf(p, stack, ssig):
    lp = mcmcutils.logprior(p, priors)
    if not np.isfinite(lp): return -np.inf
    m, c, lnf = p
    est = model_log_prof(m, c)
    s2 = ssig**2 + (np.exp(lnf) * est) ** 2
    return lp - 0.5 * np.sum((stack - est) ** 2 / s2 + np.log(s2))

def logpost_ftj(p):
    lp = mcmcutils.logprior(p, priors)
    if not np.isfinite(lp): return -np.inf
    m, c, lnf = p
    est = model_log_prof(m, c)
    s2 = sig**2 + (np.exp(lnf) * est) ** 2
    return lp - 0.5 * np.sum((prof - est[None, :]) ** 2 / s2[None, :] + np.log(s2)[None, :].repeat(len(prof), 0))

def run_emcee(logpost, args=(), nw=32, nsteps=4000, burn=1500, seed=0):
    rng = np.random.default_rng(seed)
    p0 = np.column_stack([rng.normal(14.4, 0.1, nw), rng.normal(4.6, 0.3, nw), rng.uniform(-9, -6, nw)])
    sam = emcee.EnsembleSampler(nw, 3, logpost, args=args)
    sam.run_mcmc(p0, nsteps, progress=False)
    return sam.get_chain(discard=burn, flat=True)[:, :2]

t0 = time.time()
stack, ssig = stackutils.stack_log_profiles(prof, sig, "corrected_mean")
mcmc_jtf = run_emcee(logpost_jtf, args=(stack, ssig))
print(f"MCMC JTF done {time.time()-t0:.0f}s", flush=True)
mcmc_ftj = run_emcee(logpost_ftj)
print(f"MCMC FTJ done {time.time()-t0:.0f}s", flush=True)

# ---------------- SBI (retrain on sliced vectors) ----------------
theta_ftj = np.load(f"{SIM}/sample_mc_pairs.npy")            # (n, 15)
x_ftj = np.load(f"{SIM}/simulated_nfw_profiles.npy")[:, :, OUTER]  # (n, 376, 6)
theta_jtf = np.load(f"{SIM}/sample_mc_percentiles.npy")
corr = np.load(f"{SIM}/sample_mc_correlations.npy")
theta_jtf = np.column_stack((np.concatenate((theta_jtf[:, :, 0], theta_jtf[:, :, 1]), axis=1),
                             np.clip(corr, -0.99, 0.99)))
x_jtf = np.load(f"{SIM}/simulated_jtf_nfw_profiles.npy")[:, OUTER]  # sliced stack == stack of slice

inf1 = sbi_.gen_inferrer(priors, theta_ftj.shape[1])
post_ftj = sbi_.train_inferrer(inf1, theta_ftj, x_ftj)
print(f"SBI FTJ trained {time.time()-t0:.0f}s", flush=True)
inf2 = sbi_.gen_inferrer(priors, theta_jtf.shape[1])
post_jtf = sbi_.train_inferrer(inf2, theta_jtf, x_jtf)
print(f"SBI JTF trained {time.time()-t0:.0f}s", flush=True)

pairs = np.load(f"{OBS}/drawn_mc_pairs.npy")
sbi_jtf_c, sbi_ftj_c = sbi_.apply_observations(post_ftj, post_jtf, pairs, prof,
                                               stack_estimator="corrected_mean", sigmas=sig)[:2]
sbi_jtf_c, sbi_ftj_c = np.asarray(sbi_jtf_c), np.asarray(sbi_ftj_c)
print(f"SBI applied {time.time()-t0:.0f}s", flush=True)

# ---------------- summary vs full-range ----------------
full = pickle.load(open(f"{INF_FULL}/mcmc_chains.pickle", "rb"))
full_sbi = pickle.load(open(f"{INF_FULL}/sbi_chains.pickle", "rb"))
def stats(ch, cols=(0, 1)):
    ch = np.asarray(ch)
    return [float(np.median(ch[:, cols[0]])), float(np.std(ch[:, cols[0]])),
            float(np.median(ch[:, cols[1]])), float(np.std(ch[:, cols[1]]))]
summary = {
    "outer_bins": int(OUTER.sum()),
    "mcmc_jtf_outer": stats(mcmc_jtf), "mcmc_jtf_full": stats(np.asarray(full[0])),
    "mcmc_ftj_outer": stats(mcmc_ftj), "mcmc_ftj_full": stats(np.asarray(full[1])),
    "sbi_jtf_outer": stats(sbi_jtf_c, (3, 10)), "sbi_jtf_full": stats(np.asarray(full_sbi[0]), (3, 10)),
    "sbi_ftj_outer": stats(sbi_ftj_c, (3, 10)), "sbi_ftj_full": stats(np.asarray(full_sbi[1]), (3, 10)),
    "truth_median": [float(np.median(pairs[:, 0])), float(np.median(pairs[:, 1]))],
    "truth_std": [float(np.std(pairs[:, 0])), float(np.std(pairs[:, 1]))],
}
pickle.dump({"mcmc_jtf": mcmc_jtf, "mcmc_ftj": mcmc_ftj,
             "sbi_jtf": sbi_jtf_c, "sbi_ftj": sbi_ftj_c},
            open(f"{OUT}/chains_outer.pickle", "wb"))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)

# ---------------- appendix figure ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
sys.path.insert(0, os.path.join(SD, "..", "plot"))
from plotutils import build_gaussian_summary_from_chain
def pop_gauss_samples(chain, n=8000, seed=0):
    # the SBI FTJ *population estimate*: sample the reconstructed 2D Gaussian
    # (NOT columns 3/10, whose spread is only the median-summary estimation scatter)
    g = build_gaussian_summary_from_chain(np.asarray(chain))
    mM, mc = g["mass"]["mu"], g["concentration"]["mu"]
    sM, sc, rho = g["mass"]["sigma"], g["concentration"]["sigma"], g["correlation"]["rho"]
    cov = [[sM**2, rho*sM*sc], [rho*sM*sc, sc**2]]
    return np.random.default_rng(seed).multivariate_normal([mM, mc], cov, size=n)
for ax, (name, fu, ou) in zip(axes, [
        ("MCMC fit-then-join", np.asarray(full[1]), mcmc_ftj),
        ("SBI fit-then-join (population estimate)",
         pop_gauss_samples(full_sbi[1]), pop_gauss_samples(sbi_ftj_c))]):
    sns.kdeplot(x=pairs[:, 0], y=pairs[:, 1], color="k", levels=[0.05, 0.3173], ax=ax, linestyles=["-", "--"])
    sns.kdeplot(x=fu[:, 0], y=fu[:, 1], color="C0", levels=[0.05, 0.3173], ax=ax)
    sns.kdeplot(x=ou[:, 0], y=ou[:, 1], color="C3", levels=[0.05, 0.3173], ax=ax)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(r"$\log_{10} M$")
axes[0].set_ylabel(r"$c$")
import matplotlib.lines as mlines
axes[1].legend(handles=[mlines.Line2D([], [], color="k", label="true population"),
                        mlines.Line2D([], [], color="C0", label="all 30 bins"),
                        mlines.Line2D([], [], color="C3", label=r"$R > 200\,h^{-1}$kpc only")],
               fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG}/radial_range_refit.pdf")
print(json.dumps(summary, indent=1), flush=True)
print(f"DONE {time.time()-t0:.0f}s -> {OUT} + {FIG}/radial_range_refit.pdf", flush=True)
