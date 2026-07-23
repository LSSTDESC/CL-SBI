"""
Script version of notebooks/figure_aggregate_comparison.ipynb (Figs 8/10/11 generator),
adapted for: (1) --observable delta_sigma (suffixed inference dirs), (2) the truth
reference = a large seeded POPULATION sample from the obs config rather than the finite
N_c observed draw (Payerne P2-d), (3) Fig 11 with the paper's 3 rows (contamination first).
Outputs: outputs/plots/infer_agg{suffix}/aggregate_ftj_{in_distro,ood_noise_mc,ood_rm}.{pdf,png}
"""
import sys, os, json, pickle, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from weaklensclustersbi.simulations import population

ap = argparse.ArgumentParser()
ap.add_argument("--observable", default="delta_sigma")
args = ap.parse_args()
SD = os.path.dirname(__file__)
SUF = "" if args.observable == "surface_density" else f".{args.observable}"
plt.style.use(os.path.join(SD, "..", "plot", "mplstyle.txt"))
NUM_SIMS, NUM_OBS = 10000, 376

COLORS = {"obs": "black", "mcmc": "#0077BB", "sbi": "#EE7733", "truth": "#CC3311"}

EXPERIMENTS_FIG8 = [
    ("sim_z1", "infer_z1", "obs_z1_lambda5", "Baseline"),
    ("sim_z1_high_mc_scatter", "infer_z1_high_mc_scatter", "obs_z1_lambda5_high_mc_scatter", "High M-c Scatter"),
    ("sim_z1_high_noise", "infer_z1", "obs_z1_lambda5_high_noise", "High Noise"),
    ("sim_z1_high_rm_scatter", "infer_z1_high_rm_scatter", "obs_z1_lambda5_high_rm_scatter", r"High $\lambda$-M Scatter"),
]
EXPERIMENTS_FIG10 = [
    ("sim_z1", "infer_z1", "obs_z1_lambda5_prada", "Prada M-c"),
    ("sim_z1", "infer_z1", "obs_z1_lambda5_ludlow", "Ludlow M-c"),
    ("sim_z1", "infer_z1", "obs_z1_lambda5_high_noise", "Obs higher noise than sims"),
    ("sim_z1_high_noise", "infer_z1", "obs_z1_lambda5", "Obs lower noise than sims"),
]
EXPERIMENTS_FIG11 = [  # paper order: contamination first (3 rows)
    ("sim_z1", "infer_z1", "obs_z1_lambda5_low_richness_contam", "Richness contamination"),
    ("sim_z1", "infer_z1", "obs_z1_lambda5_high_rm_scatter", r"Obs: Higher $\lambda$-M Scatter"),
    ("sim_z1_high_rm_scatter", "infer_z1_high_rm_scatter", "obs_z1_lambda5", r"Obs: Lower $\lambda$-M Scatter"),
]


def population_reference(obs_id, n_samples=5000, seed=4242):
    """Large seeded sample of the TRUE (M,c) population for this obs config."""
    cfg = json.load(open(os.path.join(SD, f"../configs/observations/{obs_id}.json")))
    np.random.seed(seed)
    pairs = np.asarray(population.gen_mc_pairs_in_richness_bin(
        cfg["min_richness"], cfg["max_richness"], rm_relation=cfg["rm_relation"],
        mc_relation=cfg["mc_relation"], num_samples=n_samples, mc_scatter=cfg["mc_scatter"],
        rm_scatter=cfg["rm_scatter"], min_z=cfg["min_z"], max_z=cfg["max_z"],
        richness_contam_frac=cfg.get("richness_contam_frac", 0.0),
        lambda_min_contam=cfg.get("min_richness_contam", None)))
    return pairs


def load_experiment_data(sim_id, infer_id, obs_id):
    infer_path = os.path.join(SD, f"../outputs/inference/{sim_id}.{infer_id}.{obs_id}.{NUM_SIMS}.{NUM_OBS}{SUF}")
    with open(os.path.join(infer_path, "mcmc_ftj_samplers.pickle"), "rb") as f:
        mcmc_ftj = pickle.load(f).flatchain[:, :2]
    with open(os.path.join(infer_path, "sbi_chains.pickle"), "rb") as f:
        sbi_ftj = np.asarray(pickle.load(f)[1])
    n_levels, med_idx, idx_16, idx_84 = 7, 3, 1, 5
    mp, cp = sbi_ftj[:, :n_levels], sbi_ftj[:, n_levels:2 * n_levels]
    mu_m, mu_c = np.median(mp[:, med_idx]), np.median(cp[:, med_idx])
    sig_m = (np.median(mp[:, idx_84]) - np.median(mp[:, idx_16])) / 2.0
    sig_c = (np.median(cp[:, idx_84]) - np.median(cp[:, idx_16])) / 2.0
    rho = np.median(np.clip(sbi_ftj[:, -1], -0.99, 0.99))
    cov = np.array([[sig_m**2, rho * sig_m * sig_c], [rho * sig_m * sig_c, sig_c**2]])
    sbi_mc = np.random.default_rng(42).multivariate_normal([mu_m, mu_c], cov, size=10000)
    # truth reference = population, not the finite N_c draw
    pop = population_reference(obs_id)
    return {"mcmc_ftj": mcmc_ftj, "sbi_ftj": sbi_mc,
            "true_median": (np.median(pop[:, 0]), np.median(pop[:, 1])), "obs_mc_pairs": pop}


def plot_kde_contours(ax, samples, color, levels=[0.6827, 0.95], fill=True, ls="-", lw=1.5, alpha_fill=0.2):
    x, y = samples[:, 0], samples[:, 1]
    xx, yy = np.mgrid[x.min() - 0.2:x.max() + 0.2:100j, y.min() - 0.5:y.max() + 0.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    try:
        f = np.reshape(stats.gaussian_kde(samples.T)(positions).T, xx.shape)
    except np.linalg.LinAlgError:
        return
    sf = np.sort(f.ravel())[::-1]
    cs = np.cumsum(sf); cs /= cs[-1]
    cl = sorted(sf[np.searchsorted(cs, lv)] for lv in levels)
    if fill:
        ax.contourf(xx, yy, f, levels=[cl[0], cl[-1], f.max()], colors=[color], alpha=[alpha_fill, alpha_fill * 1.5])
    ax.contour(xx, yy, f, levels=cl, colors=[color], linewidths=lw, linestyles=ls)


def create_row_figure(data_dict, output_name, figsize_per_row=(12, 2.8)):
    n = len(data_dict)
    fig, axes = plt.subplots(n, 3, figsize=(figsize_per_row[0], figsize_per_row[1] * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for idx, (label, d) in enumerate(data_dict.items()):
        true, mcmc, sbi, obs = d["true_median"], d["mcmc_ftj"], d["sbi_ftj"], d["obs_mc_pairs"]
        is_last = idx == n - 1
        ax = axes[idx, 0]
        plot_kde_contours(ax, obs, COLORS["obs"], fill=False, ls="--", lw=2)
        plot_kde_contours(ax, mcmc, COLORS["mcmc"], fill=True, alpha_fill=0.25)
        plot_kde_contours(ax, sbi, COLORS["sbi"], fill=True, alpha_fill=0.25)
        ax.scatter(true[0], true[1], marker="*", s=180, c=COLORS["truth"], edgecolors="k", linewidths=0.5, zorder=10)
        ax.text(0.03, 0.97, f"({chr(97 + idx)}) {label}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_ylabel("Concentration", fontsize=11)
        if is_last: ax.set_xlabel(r"$\log_{10}(M_{200m}/M_\odot)$", fontsize=11)
        if idx == 0: ax.set_title("2D Posterior", fontsize=12, fontweight="bold")
        for col, pi, ttl, xlab in ((1, 0, "Mass Marginal", r"$\log_{10}(M_{200m}/M_\odot)$"),
                                   (2, 1, "Concentration Marginal", "Concentration")):
            ax = axes[idx, col]
            pad = 0.1 if pi == 0 else 0.2
            bins = np.linspace(min(obs[:, pi].min(), mcmc[:, pi].min(), sbi[:, pi].min()) - pad,
                               max(obs[:, pi].max(), mcmc[:, pi].max(), sbi[:, pi].max()) + pad, 35)
            ax.hist(obs[:, pi], bins=bins, density=True, alpha=0.4, color=COLORS["obs"],
                    label="True population", histtype="stepfilled", edgecolor=COLORS["obs"], linewidth=1.5)
            ax.hist(mcmc[:, pi], bins=bins, density=True, alpha=0.5, color=COLORS["mcmc"], label="MCMC", histtype="stepfilled")
            ax.hist(sbi[:, pi], bins=bins, density=True, alpha=0.5, color=COLORS["sbi"], label="SBI", histtype="stepfilled")
            ax.axvline(true[pi], color=COLORS["truth"], ls="--", lw=2.5)
            ax.set_ylabel("Density", fontsize=11)
            if is_last: ax.set_xlabel(xlab, fontsize=11)
            if idx == 0:
                ax.set_title(ttl, fontsize=12, fontweight="bold")
                if col == 1: ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    outdir = os.path.join(SD, f"../outputs/plots/infer_agg{SUF}")
    os.makedirs(outdir, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(os.path.join(outdir, f"{output_name}.{ext}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"saved {outdir}/{output_name}.pdf")


for exps, name in ((EXPERIMENTS_FIG8, "aggregate_ftj_in_distro"),
                   (EXPERIMENTS_FIG10, "aggregate_ftj_ood_noise_mc"),
                   (EXPERIMENTS_FIG11, "aggregate_ftj_ood_rm")):
    data = {}
    for sim, inf, obs, label in exps:
        try:
            data[label] = load_experiment_data(sim, inf, obs)
        except FileNotFoundError:
            print(f"  MISSING: {label}")
    if data:
        create_row_figure(data, name)
