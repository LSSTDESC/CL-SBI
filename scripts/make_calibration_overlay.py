"""
Rebuild the Fig-13 calibration overlays (in-distribution + OOD panels): FTJ coverage curves
for MCMC (blue) and SBI (orange), one linestyle per experiment, vs the y=x diagonal.
Truth = large population sample from each obs config (not the finite N_c draw).
Usage: python3 make_calibration_overlay.py --observable delta_sigma
"""
import sys, os, json, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_calibration import sample_truth_ftj, aggregate_coverage, CONF_LEVELS

ap = argparse.ArgumentParser()
ap.add_argument("--observable", default="delta_sigma")
args = ap.parse_args()
sd = os.path.dirname(__file__)
suf = "" if args.observable == "surface_density" else f".{args.observable}"

IN_DISTRO = [
    ("Near ideal",              "sim_z1.infer_z1.obs_z1_lambda5"),
    ("High noise",              "sim_z1_high_noise.infer_z1.obs_z1_lambda5_high_noise"),
    ("High M-C scatter",        "sim_z1_high_mc_scatter.infer_z1_high_mc_scatter.obs_z1_lambda5_high_mc_scatter"),
    ("High $\\lambda$-M scatter", "sim_z1_high_rm_scatter.infer_z1_high_rm_scatter.obs_z1_lambda5_high_rm_scatter"),
]
OOD = [
    ("Prada M-c",               "sim_z1.infer_z1.obs_z1_lambda5_prada"),
    ("Ludlow M-c",              "sim_z1.infer_z1.obs_z1_lambda5_ludlow"),
    ("Low noise in sims",       "sim_z1.infer_z1.obs_z1_lambda5_high_noise"),
    ("High noise in sims",      "sim_z1_high_noise.infer_z1.obs_z1_lambda5"),
    ("Richness contam.",        "sim_z1.infer_z1.obs_z1_lambda5_low_richness_contam"),
    ("Low $\\lambda$-M in sims",  "sim_z1.infer_z1.obs_z1_lambda5_high_rm_scatter"),
    ("High $\\lambda$-M in sims", "sim_z1_high_rm_scatter.infer_z1.obs_z1_lambda5"),
]
LINESTYLES = ["-", (0, (5, 1)), ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 3))]
COLORS = {"mcmc": "#1f77b4", "sbi": "#ff7f0e"}

def build(panel, name):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    style_handles = []
    for i, (label, key) in enumerate(panel):
        # Prefer the saved per-experiment coverage pickle (written by plot_calibration with
        # the population truth) so the overlay is exactly consistent with the per-experiment
        # calibration plots; fall back to recomputing if absent.
        pkl = os.path.join(sd, f"../outputs/plots/{key}.10000.376{suf}/calibration/calibration_ftj.pickle")
        if os.path.isfile(pkl):
            import pickle as _pickle
            agg = _pickle.load(open(pkl, "rb"))
        else:
            run_path = os.path.join(sd, f"../outputs/inference/{key}.10000.376{suf}")
            if not os.path.isdir(run_path):
                print(f"  SKIP missing {key}"); continue
            obs_id = key.split(".")[2]
            obs_cfg = json.load(open(os.path.join(sd, f"../configs/observations/{obs_id}.json")))
            truth = {"ftj": sample_truth_ftj(obs_cfg, n_samples=5000)}
            agg, _ = aggregate_coverage(run_path, truth, ftj_only=True, ks_nc=376)
        ls = LINESTYLES[i % len(LINESTYLES)]
        for method in ("mcmc", "sbi"):
            cov = agg.get((method, "ftj"))
            if cov is None: continue
            x = sorted(cov.keys()); y = [cov[p] for p in x]
            ax.plot(x, y, ls=ls, color=COLORS[method], lw=1.6)
        style_handles.append(plt.Line2D([], [], color="k", ls=ls, label=label))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Estimated confidence level", fontsize="large")
    ax.set_ylabel("Fraction of truth covered", fontsize="large")
    # legends: opaque white boxes drawn ABOVE the curves (they were disappearing behind lines),
    # and the Method legend moved off the MCMC curves that hug the lower-right corner
    leg1 = ax.legend(handles=style_handles, title="Experiment", fontsize="small",
                     loc="upper left", framealpha=1.0, facecolor="white", edgecolor="0.7")
    leg1.set_zorder(20)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=[plt.Line2D([], [], color=COLORS["mcmc"], lw=2, label="MCMC fit-then-join"),
                              plt.Line2D([], [], color=COLORS["sbi"], lw=2, label="SBI fit-then-join")],
                     title="Method", fontsize="small", loc="center right",
                     bbox_to_anchor=(0.98, 0.35), framealpha=1.0, facecolor="white", edgecolor="0.7")
    leg2.set_zorder(20)
    outdir = os.path.join(sd, f"../outputs/plots/calibration_overlay{suf}")
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=120)
    plt.close(fig)
    print(f"  saved {outdir}/{name}.pdf")

build(IN_DISTRO, "calibration_ftj_overlay_10k_376")
build(OOD, "calibration_ftj_overlay_10k_376_ood")
