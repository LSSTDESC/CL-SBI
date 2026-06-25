"""Multi-bin beta figure: pooled 7-bin (M,c) cloud + slope, and the beta-posterior tightening."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 13})

REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"; OUT = f"{REPO}/notebooks/hierarchical_poc_outputs"
d = pickle.load(open(f"{OUT}/multibin_beta.pkl", "rb"))
beta, beta_sd = d["est"]["beta"]; c0, _ = d["est"]["c0"]; mu_all = d["true_mu_all"]
tb = d["true_beta"]; sbeta, sbeta_sd = d["single_bin_beta"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
cols = plt.cm.viridis(np.linspace(0, 1, 7))
allmc = []
for k, b in enumerate(range(1, 8)):
    mc = np.load(f"{REPO}/outputs/observations/obs_z1_lambda{b}.376/drawn_mc_pairs.npy")
    allmc.append(mc)
    a1.scatter(mc[:, 0], mc[:, 1], s=6, color=cols[k], alpha=0.35, label=fr"$\lambda$ bin {b}")
allmc = np.concatenate(allmc); ct = allmc[:, 1].mean()
xs = np.linspace(13.0, 15.2, 100)
a1.plot(xs, c0 + beta * (xs - mu_all), "r-", lw=2.5, zorder=5,
        label=fr"multi-bin fit $\beta={beta:.2f}\pm{beta_sd:.2f}$")
a1.fill_between(xs, c0 + (beta - beta_sd) * (xs - mu_all), c0 + (beta + beta_sd) * (xs - mu_all),
                color="r", alpha=0.2, zorder=4)
a1.plot(xs, ct + tb * (xs - mu_all), "k--", lw=2, zorder=5, label=fr"true slope {tb:.2f}")
a1.set_xlabel(r"$\log_{10} M$"); a1.set_ylabel(r"$c$"); a1.legend(fontsize=8, ncol=2, loc="upper right")
a1.set_title(r"Pooled 7-bin $(M,c)$: $\sim$2.1 dex baseline reveals the slope")

bx = np.linspace(-2.2, 0.3, 400)
a2.plot(bx, norm.pdf(bx, sbeta, sbeta_sd), color="#9467bd", lw=2,
        label=fr"single bin: ${sbeta:.2f}\pm{sbeta_sd:.2f}$")
a2.fill_between(bx, norm.pdf(bx, sbeta, sbeta_sd), color="#9467bd", alpha=0.2)
a2.plot(bx, norm.pdf(bx, beta, beta_sd), color="r", lw=2,
        label=fr"multi-bin: ${beta:.2f}\pm{beta_sd:.2f}$")
a2.fill_between(bx, norm.pdf(bx, beta, beta_sd), color="r", alpha=0.2)
a2.axvline(tb, color="k", ls="--", lw=2, label=fr"true {tb:.2f}")
a2.set_xlabel(r"$\beta$ ($M$--$c$ slope)"); a2.set_yticks([]); a2.legend(fontsize=10)
a2.set_title(r"$\beta$ posterior: $7\times$ tighter with the multi-bin lever arm")
fig.tight_layout(); fig.savefig(f"{OUT}/multibin_beta.png", dpi=140, bbox_inches="tight")
print("saved", f"{OUT}/multibin_beta.png")
