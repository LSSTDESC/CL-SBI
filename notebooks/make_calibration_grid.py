"""2x4 small-multiples calibration grid: one panel per experiment, 3 methods + diagonal.
SBI+MCMC coverage from the pipeline pickles; HMC coverage computed from saved HBI posteriors."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, sys
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":13,"axes.labelsize":14,"axes.titlesize":13,
                     "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":12})
sys.path.insert(0,"/Users/akumgill/Documents/GitHub/CL-SBI/scripts")
from plot_calibration import compute_coverage, CONF_LEVELS
REPO="/Users/akumgill/Documents/GitHub/CL-SBI"; OUT=f"{REPO}/notebooks/hierarchical_poc_outputs"

EXPS=[("obs_z1_lambda5","Baseline"),("obs_z1_lambda5_high_mc_scatter","High M-c scatter"),
      ("obs_z1_lambda5_high_noise","High noise"),("obs_z1_lambda5_high_rm_scatter",r"High $\lambda$-M scatter"),
      ("obs_z1_lambda5_low_richness_contam","Low richness contam."),
      ("obs_z1_lambda5_high_richness_contam","High richness contam."),
      ("obs_z1_lambda5_prada","Prada M-c (OOD)"),("obs_z1_lambda5_ludlow","Ludlow M-c (OOD)")]
x=np.array(CONF_LEVELS)
def y(cov): return [cov[p] for p in CONF_LEVELS]
def dmax(cov): return max(abs(cov[p]-p) for p in CONF_LEVELS)

# Neural HBI per-experiment posterior samples (draw population, same compute_coverage as HMC)
nhbi=pickle.load(open(f"{OUT}/afull_neural_hbi.pkl","rb"))["experiments"]
def nhbi_cov(obs, true_mc, rng):
    if obs not in nhbi: return None
    s=nhbi[obs]["samples"]; i=rng.integers(0,len(s),20000)
    lM=s[i,0]+s[i,1]*rng.standard_normal(20000)
    cc=s[i,2]+s[i,3]*(lM-s[i,0])+s[i,4]*rng.standard_normal(20000)
    return compute_coverage(np.column_stack([lM,cc]),true_mc)

fig,axes=plt.subplots(2,4,figsize=(15,7.5)); axes=axes.ravel()
for ax,(obs,title) in zip(axes,EXPS):
    cal=pickle.load(open(f"{REPO}/outputs/plots/sim_z1.infer_z1.{obs}.10000.376/calibration/calibration_ftj.pickle","rb"))
    true_mc=np.load(f"{REPO}/outputs/observations/{obs}.376/drawn_mc_pairs.npy")
    h=pickle.load(open(f"{OUT}/hier_post_{obs}.pkl","rb"))["samples"]
    rng=np.random.default_rng(0); n=len(h["mu_M"]); idx=rng.integers(0,n,20000)
    lM=h["mu_M"][idx]+h["sig_M"][idx]*rng.standard_normal(20000)
    cc=h["c0"][idx]+h["beta"][idx]*(lM-h["mu_M"][idx])+h["sig_c"][idx]*rng.standard_normal(20000)
    cov_hmc=compute_coverage(np.column_stack([lM,cc]),true_mc)
    cov_mcmc, cov_sbi = cal[("mcmc","ftj")], cal[("sbi","ftj")]
    cov_nhbi=nhbi_cov(obs,true_mc,np.random.default_rng(1))
    ax.plot([0,1],[0,1],"k--",lw=1.2)
    ax.plot(x,y(cov_mcmc),color="#1f77b4",lw=2,label="MCMC joint")
    ax.plot(x,y(cov_sbi),color="#ff7f0e",lw=2,label="SBI FTJ")
    ax.plot(x,y(cov_hmc),color="#2ca02c",lw=2,label="Hier. HMC")
    txt=f"$\\Delta_{{\\max}}$: M={dmax(cov_mcmc):.2f} S={dmax(cov_sbi):.2f} H={dmax(cov_hmc):.2f}"
    if cov_nhbi is not None:
        ax.plot(x,y(cov_nhbi),color="#9467bd",lw=2,label="Neural HBI")
        txt+=f" N={dmax(cov_nhbi):.2f}"
    ax.set_title(f"{title}",fontsize=12)
    ax.text(0.04,0.96,txt,transform=ax.transAxes,va="top",fontsize=8.0)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
for i,ax in enumerate(axes):
    if i>=4: ax.set_xlabel("nominal $p$")
    if i%4==0: ax.set_ylabel("empirical coverage")
axes[0].legend(loc="lower right",fontsize=9)
fig.suptitle("Calibration across experiments: empirical coverage vs nominal credible level",fontsize=14)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(f"{OUT}/calibration_grid.png",dpi=150,bbox_inches="tight")
print("wrote calibration_grid.png")
