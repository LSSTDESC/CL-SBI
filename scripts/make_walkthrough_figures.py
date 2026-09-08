"""
Regenerate the profile-based obs_walkthrough figures (example nfw, noisy stacked nfws,
fractional_diff) on the DeltaSigma observable, matching the styling of the originals from
adhoc/notebooks/'obs walkthrough.ipynb' (seed=2, obs_z1_lambda5). The r-m / m-c relation
walkthrough figures are observable-independent and are not touched.
Writes into outputs/plots/obs_walkthrough.delta_sigma/ (copy script moves them into the paper).
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from weaklensclustersbi.simulations import population, wlprofile

SD = os.path.dirname(__file__)
KIND = "delta_sigma"
YLAB = r"$\Delta\Sigma$ [$M_\odot h / kpc^2$]"
plt.style.use(os.path.join(SD, "..", "plot", "mplstyle.txt"))

cfg = json.load(open(os.path.join(SD, "../configs/observations/obs_z1_lambda5.json")))
NUM_OBS = 376
z = (cfg["min_z"] + cfg["max_z"]) / 2
rbins = 10 ** np.arange(0, cfg["num_radial_bins"] / 10, 0.1)
noise_dex = cfg["profile_noise_dex"]

np.random.seed(2)  # notebook seed
pairs = np.asarray(population.gen_mc_pairs_in_richness_bin(
    cfg["min_richness"], cfg["max_richness"], rm_relation=cfg["rm_relation"],
    mc_relation=cfg["mc_relation"], num_samples=NUM_OBS, mc_scatter=cfg["mc_scatter"],
    rm_scatter=cfg["rm_scatter"], min_z=cfg["min_z"], max_z=cfg["max_z"]))
log10m, conc = pairs[:, 0], pairs[:, 1]
idx1 = np.random.randint(0, NUM_OBS)

nfw1 = wlprofile.simulate_nfw(log10m[idx1], conc[idx1], rbins, z, kind=KIND)
nfws = np.array([population.calculate_noise(
    wlprofile.simulate_nfw(m, c, rbins, z, kind=KIND), noise_dex) for m, c in pairs])
log_sigmas = np.std(np.log10(nfws), axis=0)
nfw1_noisy = population.calculate_noise(nfw1, noise_dex)

outdir = os.path.join(SD, "../outputs/plots/obs_walkthrough.delta_sigma")
os.makedirs(outdir, exist_ok=True)

# ---- example nfw ----
med = np.median(nfws, axis=0)
upper = np.exp(np.log(med) + log_sigmas) - nfw1_noisy
lower = nfw1_noisy - np.exp(np.log(med) - log_sigmas)
plt.figure(figsize=(8, 8)); plt.loglog()
plt.xlabel("radius [kpc/h]", fontsize="xx-large")
plt.ylabel(YLAB, fontsize="xx-large")
plt.ylim(min(nfw1_noisy) * 0.8, max(nfw1_noisy) * 1.2)
plt.plot(rbins, nfw1, "--", color="tab:red", label="pt1, no noise")
plt.errorbar(rbins, nfw1_noisy, yerr=[np.abs(lower), np.abs(upper)], linestyle="-",
             color="tab:red", ecolor="k", capsize=3.0, label="pt1, with noise and error")
plt.legend()
plt.savefig(os.path.join(outdir, "example nfw.pdf")); plt.close()

# ---- noisy stacked nfws ----
median_mc = wlprofile.simulate_nfw(np.median(log10m), np.median(conc), rbins, z, kind=KIND)
mean_mc = wlprofile.simulate_nfw(np.mean(log10m), np.mean(conc), rbins, z, kind=KIND)
plt.figure(figsize=(8, 8)); plt.loglog()
plt.xlabel("radius [kpc/h]", fontsize="xx-large")
plt.ylabel(YLAB, fontsize="xx-large")
for nfw in nfws:
    plt.plot(rbins, nfw, "-", color="gray", alpha=0.05)
plt.plot(rbins, median_mc, color="k", linestyle="--", linewidth=3, label="NFW of median m-c pair")
plt.plot(rbins, mean_mc, color="tab:red", linestyle="--", label="NFW of mean m-c pair")
plt.plot(rbins, np.median(nfws, axis=0), color="k", label="median NFW profile")
plt.plot(rbins, np.mean(nfws, axis=0), color="tab:red", label="mean NFW profile")
corr = np.exp((noise_dex * np.log(10.0)) ** 2 / 2.0)
plt.plot(rbins, np.mean(nfws, axis=0) / corr, color="tab:green", linewidth=2,
         label="bias-corrected mean (adopted stack)")
plt.legend()
plt.xlim(min(rbins), max(rbins))
plt.ylim(min(min(np.median(nfws, 0)), min(np.mean(nfws, 0)), min(mean_mc)),
         max(max(np.median(nfws, 0)), max(np.mean(nfws, 0)), max(mean_mc)))
plt.savefig(os.path.join(outdir, "noisy stacked nfws.pdf")); plt.close()

# ---- fractional diff ----
plt.figure(figsize=(8, 5)); plt.xscale("log")
plt.xlabel("radius [kpc/h]", fontsize="xx-large")
plt.ylabel("Fractional Diff with Median NFW", fontsize="x-large")
plt.axhline(0, color="gray", linestyle="dotted", alpha=0.5)
plt.plot(rbins, (np.mean(nfws, 0) / np.median(nfws, 0)) - 1, color="tab:red", label="Mean NFW Profile")
plt.plot(rbins, (np.mean(nfws, 0) / corr / np.median(nfws, 0)) - 1, color="tab:green",
         label="Bias-corrected mean (adopted stack)")
plt.plot(rbins, (median_mc / np.median(nfws, 0)) - 1, color="k", linestyle="--", linewidth=3,
         label="NFW profile of median MC pair")
plt.plot(rbins, (mean_mc / np.median(nfws, 0)) - 1, color="tab:red", linestyle="-.",
         label="NFW profile of mean MC pair")
plt.legend()
plt.xlim(min(rbins), max(rbins))
plt.savefig(os.path.join(outdir, "fractional_diff.pdf")); plt.close()
print(f"saved 3 walkthrough figures -> {outdir}")
