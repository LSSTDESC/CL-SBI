"""
Appendix: Mass Function Comparison

Compare the distribution of mass-concentration pairs when drawing from a richness bin
using Tinker et al. (2008) mass function vs a flat (uniform) mass function.

This demonstrates that for a sufficiently narrow richness bin, the choice of mass
function has minimal impact on the resulting M-c distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
plt.style.use(ROOT / "plot" / "mplstyle.txt")

from weaklensclustersbi.simulations import population

# Configuration - match near-ideal experiment
RICHNESS_BIN = (30, 45)  # lambda range
NUM_SAMPLES = 10000      # large sample for smooth distributions
MC_RELATION = "child18"
RM_RELATION = "mcclintock18"
MC_SCATTER = 0.1
RM_SCATTER = 0.1
Z_RANGE = (0.2, 0.35)

np.random.seed(42)

print("Generating M-c pairs with Tinker08 mass function...")
mc_tinker = np.array(population.gen_mc_pairs_in_richness_bin(
    lambda_min=RICHNESS_BIN[0],
    lambda_max=RICHNESS_BIN[1],
    rm_relation=RM_RELATION,
    mc_relation=MC_RELATION,
    num_samples=NUM_SAMPLES,
    mc_scatter=MC_SCATTER,
    rm_scatter=RM_SCATTER,
    min_z=Z_RANGE[0],
    max_z=Z_RANGE[1],
    model="tinker08",
))

np.random.seed(42)  # Same seed for fair comparison

print("Generating M-c pairs with flat mass function...")
mc_flat = np.array(population.gen_mc_pairs_in_richness_bin(
    lambda_min=RICHNESS_BIN[0],
    lambda_max=RICHNESS_BIN[1],
    rm_relation=RM_RELATION,
    mc_relation=MC_RELATION,
    num_samples=NUM_SAMPLES,
    mc_scatter=MC_SCATTER,
    rm_scatter=RM_SCATTER,
    min_z=Z_RANGE[0],
    max_z=Z_RANGE[1],
    model="flat",
))

# Statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nTinker08 Mass Function:")
print(f"  Mass:  mean={np.mean(mc_tinker[:,0]):.3f}, std={np.std(mc_tinker[:,0]):.3f}")
print(f"  Conc:  mean={np.mean(mc_tinker[:,1]):.3f}, std={np.std(mc_tinker[:,1]):.3f}")

print(f"\nFlat Mass Function:")
print(f"  Mass:  mean={np.mean(mc_flat[:,0]):.3f}, std={np.std(mc_flat[:,0]):.3f}")
print(f"  Conc:  mean={np.mean(mc_flat[:,1]):.3f}, std={np.std(mc_flat[:,1]):.3f}")

# Practical effect sizes (difference in means / pooled std)
pooled_std_mass = np.sqrt((np.var(mc_tinker[:, 0]) + np.var(mc_flat[:, 0])) / 2)
pooled_std_conc = np.sqrt((np.var(mc_tinker[:, 1]) + np.var(mc_flat[:, 1])) / 2)
effect_mass = abs(np.mean(mc_tinker[:, 0]) - np.mean(mc_flat[:, 0])) / pooled_std_mass
effect_conc = abs(np.mean(mc_tinker[:, 1]) - np.mean(mc_flat[:, 1])) / pooled_std_conc

print(f"\nEffect Size (|Δmean| / pooled σ):")
print(f"  Mass:  {effect_mass:.3f} σ")
print(f"  Conc:  {effect_conc:.3f} σ")

# Fractional difference in means
frac_diff_mass = abs(np.mean(mc_tinker[:, 0]) - np.mean(mc_flat[:, 0])) / np.mean(mc_tinker[:, 0]) * 100
frac_diff_conc = abs(np.mean(mc_tinker[:, 1]) - np.mean(mc_flat[:, 1])) / np.mean(mc_tinker[:, 1]) * 100
print(f"\nFractional Difference in Means:")
print(f"  Mass:  {frac_diff_mass:.2f}%")
print(f"  Conc:  {frac_diff_conc:.2f}%")

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Colors
c_tinker = "#0077BB"
c_flat = "#EE7733"

# Panel (a): 2D scatter/KDE
ax = axes[0]

# KDE contours
def plot_kde_contours(ax, samples, color, label, ls="-"):
    x, y = samples[:, 0], samples[:, 1]
    xmin, xmax = x.min() - 0.1, x.max() + 0.1
    ymin, ymax = y.min() - 0.3, y.max() + 0.3
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = stats.gaussian_kde(samples.T)
    f = np.reshape(kernel(positions).T, xx.shape)

    sorted_f = np.sort(f.ravel())[::-1]
    cumsum = np.cumsum(sorted_f)
    cumsum /= cumsum[-1]
    levels = [sorted_f[np.searchsorted(cumsum, l)] for l in [0.68, 0.95]]
    levels = sorted(levels)

    ax.contour(xx, yy, f, levels=levels, colors=[color], linewidths=2, linestyles=ls)
    ax.plot([], [], color=color, lw=2, ls=ls, label=label)

plot_kde_contours(ax, mc_tinker, c_tinker, "Tinker08", ls="-")
plot_kde_contours(ax, mc_flat, c_flat, "Flat", ls="--")

ax.set_xlabel(r"$\log_{10}(M_{200m}/M_\odot)$", fontsize=12)
ax.set_ylabel("Concentration", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
ax.set_title("2D Distribution", fontsize=12, fontweight="bold")

# Panel (b): Mass histogram
ax = axes[1]
bins_mass = np.linspace(
    min(mc_tinker[:, 0].min(), mc_flat[:, 0].min()) - 0.05,
    max(mc_tinker[:, 0].max(), mc_flat[:, 0].max()) + 0.05,
    40
)
ax.hist(mc_tinker[:, 0], bins=bins_mass, density=True, alpha=0.5, color=c_tinker, label="Tinker08")
ax.hist(mc_flat[:, 0], bins=bins_mass, density=True, alpha=0.5, color=c_flat, label="Flat")
ax.axvline(np.mean(mc_tinker[:, 0]), color=c_tinker, ls="--", lw=2)
ax.axvline(np.mean(mc_flat[:, 0]), color=c_flat, ls="--", lw=2)
ax.set_xlabel(r"$\log_{10}(M_{200m}/M_\odot)$", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.text(0.05, 0.95, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
ax.set_title("Mass Distribution", fontsize=12, fontweight="bold")
ax.text(0.95, 0.75, f"$\\Delta\\mu = {effect_mass:.2f}\\sigma$", transform=ax.transAxes, fontsize=10,
        ha="right", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Panel (c): Concentration histogram
ax = axes[2]
bins_conc = np.linspace(
    min(mc_tinker[:, 1].min(), mc_flat[:, 1].min()) - 0.2,
    max(mc_tinker[:, 1].max(), mc_flat[:, 1].max()) + 0.2,
    40
)
ax.hist(mc_tinker[:, 1], bins=bins_conc, density=True, alpha=0.5, color=c_tinker, label="Tinker08")
ax.hist(mc_flat[:, 1], bins=bins_conc, density=True, alpha=0.5, color=c_flat, label="Flat")
ax.axvline(np.mean(mc_tinker[:, 1]), color=c_tinker, ls="--", lw=2)
ax.axvline(np.mean(mc_flat[:, 1]), color=c_flat, ls="--", lw=2)
ax.set_xlabel("Concentration", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.legend(fontsize=10)
ax.text(0.05, 0.95, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
ax.set_title("Concentration Distribution", fontsize=12, fontweight="bold")
ax.text(0.95, 0.75, f"$\\Delta\\mu = {effect_conc:.2f}\\sigma$", transform=ax.transAxes, fontsize=10,
        ha="right", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()

# Save
out_path = ROOT / "outputs"
fig.savefig(out_path / "appendix_mass_function_comparison.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out_path / "appendix_mass_function_comparison.png", bbox_inches="tight", dpi=150)
print(f"\nFigure saved to: {out_path / 'appendix_mass_function_comparison.pdf'}")

plt.show()

# Print LaTeX-ready summary
print("\n" + "=" * 60)
print("LATEX SUMMARY")
print("=" * 60)
print(f"""
\\subsection{{Mass Function Independence}}
\\label{{appendix:mass_function}}

When drawing mass-concentration pairs from a narrow richness bin, we must weight
samples by the halo mass function. Here we verify that for sufficiently narrow
richness bins, this weighting has negligible impact on the resulting distribution.

Figure~\\ref{{fig:mass_function_comparison}} compares the $M$-$c$ distribution for
$\\lambda \\in [{RICHNESS_BIN[0]}, {RICHNESS_BIN[1]}]$ using two approaches:
(1) weighting by the \\citet{{tinker2008}} mass function, and
(2) uniform (flat) weighting.
With {NUM_SAMPLES:,} samples each, the Tinker08 distribution has mean mass
$\\log_{{10}} M = {np.mean(mc_tinker[:,0]):.2f} \\pm {np.std(mc_tinker[:,0]):.2f}$
and mean concentration $c = {np.mean(mc_tinker[:,1]):.2f} \\pm {np.std(mc_tinker[:,1]):.2f}$,
while the flat distribution has mean mass
$\\log_{{10}} M = {np.mean(mc_flat[:,0]):.2f} \\pm {np.std(mc_flat[:,0]):.2f}$
and mean concentration $c = {np.mean(mc_flat[:,1]):.2f} \\pm {np.std(mc_flat[:,1]):.2f}$.

The effect sizes (difference in means divided by pooled standard deviation) are
${effect_mass:.2f}\\sigma$ for mass and ${effect_conc:.2f}\\sigma$ for concentration---both
well below the threshold of practical significance. The fractional differences
in the means are {frac_diff_mass:.1f}\\% for mass and {frac_diff_conc:.1f}\\% for concentration.
This confirms that within sufficiently narrow richness bins, the choice of mass
function weighting has negligible impact on the $M$-$c$ distribution used for inference.
""")
