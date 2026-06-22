"""
Paper-II analogue of Paper-I's aggregate FTJ figure (Fig 12): one row per experiment, with the
2D (M,c) population contour + 1D mass and concentration marginals, overlaying the three methods
(SBI FTJ, MCMC joint-likelihood FTJ, hierarchical HMC) against the true population.
"""
import pickle, csv, numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
})
from matplotlib.patches import Ellipse

OUT = "/Users/akumgill/Documents/GitHub/CL-SBI/notebooks/hierarchical_poc_outputs"
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"
rows = pickle.load(open(f"{OUT}/_all_methods_rows.pkl", "rb"))

# one canonical (sim_z1/infer_z1) row per unique obs set, in-distribution comparison
seen, canon = set(), []
order = ["obs_z1_lambda5", "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_noise",
         "obs_z1_lambda5_high_rm_scatter", "obs_z1_lambda5_low_richness_contam",
         "obs_z1_lambda5_high_richness_contam", "obs_z1_lambda5_prada", "obs_z1_lambda5_ludlow"]
by_obs = {}
for r in rows:
    if r["sim"] == "sim_z1" and r["infer"] == "infer_z1" and r["obs"] not in by_obs:
        by_obs[r["obs"]] = r
canon = [by_obs[o] for o in order if o in by_obs]

LABEL = {"obs_z1_lambda5": "Baseline", "obs_z1_lambda5_high_mc_scatter": "High M-c scatter",
         "obs_z1_lambda5_high_noise": "High noise", "obs_z1_lambda5_high_rm_scatter": r"High $\lambda$-M scatter",
         "obs_z1_lambda5_low_richness_contam": "Low richness contam.",
         "obs_z1_lambda5_high_richness_contam": "High richness contam.",
         "obs_z1_lambda5_prada": "Prada M-c (OOD)", "obs_z1_lambda5_ludlow": "Ludlow M-c (OOD)"}

C_TRUE, C_MCMC, C_SBI, C_HMC = "k", "#1f77b4", "#ff7f0e", "#2ca02c"


def cov_from(mu, cov):
    return np.array(mu), np.array(cov)


def ellipse(ax, mu, cov, nsig, **kw):
    vals, vecs = np.linalg.eigh(cov)
    o = vals.argsort()[::-1]; vals, vecs = vals[o], vecs[:, o]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nsig * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(mu, w, h, angle=ang, fill=False, **kw))


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


n = len(canon)
fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n))
mass_grid = np.linspace(13.8, 15.0, 300)
conc_grid = np.linspace(3.5, 7.0, 300)

for i, r in enumerate(canon):
    true_mc = np.load(f"{REPO}/outputs/observations/{r['obs']}.376/drawn_mc_pairs.npy")
    mu_t, cov_t = true_mc.mean(0), np.cov(true_mc.T)
    # method (mu, cov) builders
    def gcov(sM, sC, rho=0.0):
        return np.array([[sM**2, rho*sM*sC], [rho*sM*sC, sC**2]])
    methods = []
    if r.get("mcmc_converged", True) and r.get("mcmc_muM") is not None:
        methods.append((C_MCMC, [r["mcmc_muM"], r["mcmc_muc"]], gcov(r["mcmc_sigM"], r["mcmc_sigc"]), "MCMC joint"))
    if r.get("sbi_muM") is not None:
        methods.append((C_SBI, [r["sbi_muM"], r["sbi_muc"]], gcov(r["sbi_sigM"], r["sbi_sigc"]), "SBI FTJ"))
    if r.get("hmc_muM") is not None:
        methods.append((C_HMC, [r["hmc_muM"], r["hmc_muc"]], gcov(r["hmc_sigM"], r["hmc_sigc"]), "Hierarchical HMC"))

    # --- col 0: 2D plane ---
    ax = axes[i, 0]
    ax.scatter(true_mc[:, 0], true_mc[:, 1], s=4, color="0.6", alpha=0.4)
    ellipse(ax, mu_t, cov_t, 2, color=C_TRUE, lw=2, ls="--")
    for col, mu, cov, _ in methods:
        ellipse(ax, mu, cov, 2, color=col, lw=2)
    ax.set_ylabel(LABEL.get(r["obs"], r["obs"]) + "\n\n$c$", fontsize=14)
    ax.set_xlim(13.9, 14.9); ax.set_ylim(3.8, 6.6)
    if i == n - 1: ax.set_xlabel(r"$\log_{10} M$")

    # --- col 1: mass marginal ---
    ax = axes[i, 1]
    ax.plot(mass_grid, gauss(mass_grid, mu_t[0], np.sqrt(cov_t[0, 0])), C_TRUE, ls="--", lw=2)
    for col, mu, cov, _ in methods:
        ax.plot(mass_grid, gauss(mass_grid, mu[0], np.sqrt(cov[0, 0])), col, lw=2)
    ax.set_yticks([])
    if i == n - 1: ax.set_xlabel(r"$\log_{10} M$")
    if i == 0: ax.set_title("mass marginal")

    # --- col 2: concentration marginal ---
    ax = axes[i, 2]
    ax.plot(conc_grid, gauss(conc_grid, mu_t[1], np.sqrt(cov_t[1, 1])), C_TRUE, ls="--", lw=2)
    for col, mu, cov, _ in methods:
        ax.plot(conc_grid, gauss(conc_grid, mu[1], np.sqrt(cov[1, 1])), col, lw=2)
    ax.set_yticks([])
    if i == n - 1: ax.set_xlabel(r"$c$")
    if i == 0: ax.set_title("concentration marginal")

# shared legend
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=C_TRUE, ls="--", lw=2, label="TRUE population"),
           Line2D([], [], color=C_MCMC, lw=2, label="MCMC joint-likelihood (FTJ)"),
           Line2D([], [], color=C_SBI, lw=2, label="SBI FTJ"),
           Line2D([], [], color=C_HMC, lw=2, label="Hierarchical HMC (this work)")]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005), fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(f"{OUT}/aggregate_hbi_comparison.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/aggregate_hbi_comparison.png  ({n} experiments)")
