"""Full-dataset (McClintock 4x3 grid) recovery figure: pooled (M,c) cloud + c0/beta/gamma recovery."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 13, "legend.fontsize": 10})

REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"; OUT = f"{REPO}/notebooks/hierarchical_poc_outputs"
hmc = pickle.load(open(f"{OUT}/fulldataset_hmc.pkl", "rb"))
hyb = pickle.load(open(f"{OUT}/hybrid_fulldataset.pkl", "rb"))
tru = hmc["true"]  # c0, beta, gamma (same generated data)
LOGM_REF, Z_REF = 14.3, 0.35

CELLS = [("z1", 0.275, "lambda4", 762), ("z1", 0.275, "lambda5", 376), ("z1", 0.275, "lambda6", 123),
         ("z1", 0.275, "lambda7", 91), ("z2", 0.425, "lambda4", 1549), ("z2", 0.425, "lambda5", 672),
         ("z2", 0.425, "lambda6", 187), ("z2", 0.425, "lambda7", 148), ("z3", 0.575, "lambda4", 1612),
         ("z3", 0.575, "lambda5", 687), ("z3", 0.575, "lambda6", 205), ("z3", 0.575, "lambda7", 92)]
ZCOL = {0.275: "#1f77b4", 0.425: "#2ca02c", 0.575: "#d62728"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))

# --- left: pooled (M,c) cloud, colored by redshift, with the recovered relation at each z ---
for ztag, zc, ltag, N in CELLS:
    mc = np.load(f"{REPO}/outputs/observations/obs_{ztag}_{ltag}.{N}/drawn_mc_pairs.npy")
    a1.scatter(mc[:, 0], mc[:, 1], s=4, color=ZCOL[zc], alpha=0.25)
xs = np.linspace(13.9, 15.1, 100)
b = hmc["est"]["beta"][0]; c0 = hmc["est"]["c0"][0]; g = hmc["est"]["gamma"][0]
for zc in [0.275, 0.425, 0.575]:
    a1.plot(xs, c0 + b * (xs - LOGM_REF) + g * (zc - Z_REF), color=ZCOL[zc], lw=2.5,
            label=fr"$z={zc}$ fit")
from matplotlib.lines import Line2D
a1.legend(handles=[Line2D([], [], color=ZCOL[z], lw=2, label=fr"$z={z}$") for z in ZCOL],
          title="redshift", loc="upper right")
a1.set_xlabel(r"$\log_{10} M$"); a1.set_ylabel(r"$c$")
a1.set_title(r"Pooled McClintock $4\times3$ grid ($N=6504$): $M$--$c$ relation + $z$-evolution")

# --- right: recovered vs true for beta and gamma (the lever-arm + z-evolution payoff) ---
params = ["beta", "gamma"]; labels = [r"$\beta$ ($M$--$c$ slope)", r"$\gamma$ ($z$-evolution)"]
x = np.arange(len(params)); w = 0.25
a2.bar(x - w, [tru[p] for p in params], w, color="0.6", label="true")
a2.bar(x, [hmc["est"][p][0] for p in params], w, yerr=[hmc["est"][p][1] for p in params],
       color="#2ca02c", capsize=4, label="Hierarchical HMC")
a2.bar(x + w, [hyb["est"][p][0] for p in params], w, yerr=[hyb["est"][p][1] for p in params],
       color="#8c564b", capsize=4, label="Hybrid SBI--HMC")
a2.axhline(0, color="k", lw=0.8); a2.set_xticks(x); a2.set_xticklabels(labels)
a2.set_ylabel("recovered value"); a2.legend()
a2.set_title(r"Slope $\beta$ and $z$-evolution $\gamma$ recovered (vs single-bin: $\beta$ unconstrained)")
# annotate single-bin beta for contrast
a2.annotate("single-bin $\\beta$:\n$-1.28\\pm0.31$ (biased)", xy=(0, -0.68), xytext=(0.35, -1.15),
            fontsize=8, ha="left", color="0.4", arrowprops=dict(arrowstyle="->", color="0.6"))

fig.tight_layout(); fig.savefig(f"{OUT}/fulldataset_recovery.png", dpi=140, bbox_inches="tight")
print("saved", f"{OUT}/fulldataset_recovery.png")
print(f"HMC:    c0={hmc['est']['c0'][0]:.3f} beta={hmc['est']['beta'][0]:.3f} gamma={hmc['est']['gamma'][0]:.3f}")
print(f"Hybrid: c0={hyb['est']['c0'][0]:.3f} beta={hyb['est']['beta'][0]:.3f} gamma={hyb['est']['gamma'][0]:.3f}")
print(f"TRUE:   c0={tru['c0']:.3f} beta={tru['beta']:.3f} gamma={tru['gamma']:.3f}")
