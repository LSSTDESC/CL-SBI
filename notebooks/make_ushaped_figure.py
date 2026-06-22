"""Figure: U-shaped realistic-noise experiment. Left: recovered per-bin noise vs the true U-shaped
profile. Right: recovered population mass marginal vs truth (shows spread still recovered)."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":15,"axes.labelsize":17,"axes.titlesize":16,
                     "xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":13})
OUT="/Users/akumgill/Documents/GitHub/CL-SBI/notebooks/hierarchical_poc_outputs"
d=pickle.load(open(f"{OUT}/ushaped_noise.pkl","rb"))
RB=d["rbins"]; st=d["sigma_true"]; sp=d["sigma_noise_post"]
med=np.median(sp,0); lo,hi=np.percentile(sp,[16,84],axis=0)

fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5))
# left: per-bin noise recovery
axL.plot(RB, st, "k--", lw=2, marker="o", ms=4, label="true U-shaped noise")
axL.fill_between(RB, lo, hi, color="#2ca02c", alpha=0.3, label="inferred 16-84%")
axL.plot(RB, med, color="#2ca02c", lw=2, label="inferred (median)")
axL.set_xscale("log"); axL.set_xlabel(r"$R$ [kpc/$h$]"); axL.set_ylabel(r"per-bin noise $\sigma_i$ [dex]")
axL.set_title("Recovered radially-varying noise"); axL.legend(); axL.set_ylim(0,0.7)
axL.text(0.04,0.05,f"RMS residual = {np.sqrt(np.mean((med-st)**2)):.3f} dex",
         transform=axL.transAxes,fontsize=11,bbox=dict(boxstyle="round",fc="white",alpha=0.8))
# right: population mass marginal
gauss=lambda x,m,s: np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))
xs=np.linspace(13.9,14.85,300)
axR.plot(xs, gauss(xs, d["true_muM"], d["true_sigM"]), "k--", lw=2, label=f"TRUE ($\\sigma$={d['true_sigM']:.3f})")
axR.plot(xs, gauss(xs, d["mu_M"][0], d["sig_M"][0]), color="#2ca02c", lw=2,
         label=f"Hier. HMC ($\\sigma$={d['sig_M'][0]:.3f})")
axR.set_xlabel(r"$\log_{10} M$"); axR.set_ylabel("population density"); axR.set_yticks([])
axR.set_title("Recovered population spread"); axR.legend()
fig.suptitle("Hierarchical HMC under realistic (U-shaped) radial noise", fontsize=16)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{OUT}/ushaped_noise.png",dpi=150,bbox_inches="tight")
print("wrote ushaped_noise.png")
