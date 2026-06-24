import pickle, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/akumgill/Documents/GitHub/CL-SBI"
os.chdir(ROOT)
OUT = "notebooks/hierarchical_poc_outputs"
OBS = ["obs_z1_lambda5", "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_rm_scatter",
       "obs_z1_lambda5_high_noise", "obs_z1_lambda5_ludlow", "obs_z1_lambda5_prada",
       "obs_z1_lambda5_low_richness_contam", "obs_z1_lambda5_high_richness_contam"]
HNAMES = ["mu_M", "sig_M", "c0", "beta", "sig_c"]

# noise per bin (from task): high_noise=0.70, high_rm=0.45, others ~0.31
NOISE = {"obs_z1_lambda5":0.31, "obs_z1_lambda5_high_mc_scatter":0.31,
         "obs_z1_lambda5_high_rm_scatter":0.45, "obs_z1_lambda5_high_noise":0.70,
         "obs_z1_lambda5_ludlow":0.31, "obs_z1_lambda5_prada":0.31,
         "obs_z1_lambda5_low_richness_contam":0.31, "obs_z1_lambda5_high_richness_contam":0.31}

# Priors
SBI_BETA_LO, SBI_BETA_HI = -1.8, 0.1
SBI_BETA_PRIOR_STD = (SBI_BETA_HI - SBI_BETA_LO)/np.sqrt(12)
HYBRID_BETA_PRIOR_STD = 0.8
HMC_BETA_PRIOR_STD = 2.0

# --- load posteriors ---
sbi = pickle.load(open(f"{OUT}/afull_neural_hbi.pkl","rb"))
hyb = pickle.load(open(f"{OUT}/hybrid_sbi_hmc_allexp.pkl","rb"))

def true_beta(obs):
    p = np.load(f"outputs/observations/{obs}.376/drawn_mc_pairs.npy")
    lm, c = p[:,0], p[:,1]
    coef = np.polyfit(lm - lm.mean(), c, 1)
    bt = float(coef[0])
    sigc = float((c - (c.mean() + bt*(lm - lm.mean()))).std())
    return bt, sigc, lm, c

rows = []
for obs in OBS:
    bt, sigc_true, lm, c = true_beta(obs)
    sigM_true = float(lm.std())
    sigc_marg_true = float(c.std())
    # SBI
    sb_est = sbi["experiments"][obs]["est"]
    sbi_beta = sb_est.get("beta")
    # hybrid
    hb_est = hyb["experiments"][obs]["est"]
    hyb_beta = hb_est.get("beta")
    # hier HMC
    hp = pickle.load(open(f"{OUT}/hier_post_{obs}.pkl","rb"))
    hsamp = hp["samples"]["beta"]
    hmc_beta = (float(np.mean(hsamp)), float(np.std(hsamp)))
    rows.append(dict(obs=obs, beta_true=bt, sigc_true=sigc_true, sigM_true=sigM_true,
                     sigc_marg_true=sigc_marg_true,
                     sbi_beta=sbi_beta, hyb_beta=hyb_beta, hmc_beta=hmc_beta,
                     noise=NOISE[obs]))

print("="*120)
print(f"{'obs':40s} {'noise':>5s} {'beta_true':>9s} {'sigM':>5s} | "
      f"{'SBI beta(std,frac)':>22s} | {'HYB beta(std,frac)':>22s} | {'HMC beta(std,frac)':>22s}")
print("="*120)
for r in rows:
    def fmt(b, prior_std):
        if b is None: return "    n/a"
        return f"{b[0]:+.3f}±{b[1]:.3f} ({b[1]/prior_std:.2f})"
    short = r["obs"].replace("obs_z1_lambda5","base")
    print(f"{short:40s} {r['noise']:5.2f} {r['beta_true']:+9.3f} {r['sigM_true']:5.3f} | "
          f"{fmt(r['sbi_beta'],SBI_BETA_PRIOR_STD):>22s} | {fmt(r['hyb_beta'],HYBRID_BETA_PRIOR_STD):>22s} | "
          f"{fmt(r['hmc_beta'],HMC_BETA_PRIOR_STD):>22s}")
print(f"\nprior stds: SBI={SBI_BETA_PRIOR_STD:.3f} (uniform[-1.8,0.1]), HYBRID={HYBRID_BETA_PRIOR_STD}, HMC={HMC_BETA_PRIOR_STD}")
print("frac = posterior_std / prior_std ; ~1 means unconstrained")

# --- DEGENERACY TEST using SBI samples ---
print("\n" + "="*120)
print("DEGENERACY TEST (Hierarchical SBI samples): corr(beta,sig_c), tightness of sigc_marg")
print("="*120)
deg = {}
for obs in OBS:
    S = sbi["experiments"][obs]["samples"]  # (Nsamp,5)
    S = np.asarray(S)
    beta_s = S[:,3]; sigM_s = S[:,1]; sigc_s = S[:,4]
    sigc_marg = np.sqrt(beta_s**2 * sigM_s**2 + sigc_s**2)
    corr_bs = np.corrcoef(beta_s, sigc_s)[0,1]
    # relative spreads (std/mean) to compare tightness fairly across params w/ different scales
    rel_beta = np.std(beta_s)/(abs(np.mean(beta_s))+1e-9)
    rel_sigc = np.std(sigc_s)/(abs(np.mean(sigc_s))+1e-9)
    rel_marg = np.std(sigc_marg)/(abs(np.mean(sigc_marg))+1e-9)
    deg[obs] = dict(corr_bs=corr_bs, beta_std=np.std(beta_s), sigc_std=np.std(sigc_s),
                    marg_mean=np.mean(sigc_marg), marg_std=np.std(sigc_marg),
                    rel_beta=rel_beta, rel_sigc=rel_sigc, rel_marg=rel_marg)
    short = obs.replace("obs_z1_lambda5","base")
    print(f"{short:40s} corr(beta,sigc)={corr_bs:+.3f}  "
          f"sigc_marg={np.mean(sigc_marg):.3f}±{np.std(sigc_marg):.3f}  "
          f"relstd[beta,sigc,marg]=[{rel_beta:.2f},{rel_sigc:.2f},{rel_marg:.3f}]")

# --- noise correlation: beta posterior fractional width vs noise ---
print("\n" + "="*120)
print("NOISE vs beta-recovery (SBI). frac width and |bias| vs true")
print("="*120)
noises=[]; sbi_fracs=[]; sbi_bias=[]
for r in rows:
    if r["sbi_beta"] is None: continue
    noises.append(r["noise"])
    sbi_fracs.append(r["sbi_beta"][1]/SBI_BETA_PRIOR_STD)
    sbi_bias.append(abs(r["sbi_beta"][0]-r["beta_true"]))
noises=np.array(noises); sbi_fracs=np.array(sbi_fracs); sbi_bias=np.array(sbi_bias)
print("corr(noise, beta_frac_width) =", np.corrcoef(noises, sbi_fracs)[0,1])
print("corr(noise, beta_bias)       =", np.corrcoef(noises, sbi_bias)[0,1])

# --- curvature check for ludlow/prada ---
print("\n" + "="*120)
print("CURVATURE check (ludlow/prada): linear vs quadratic fit residual")
print("="*120)
for obs in ["obs_z1_lambda5","obs_z1_lambda5_ludlow","obs_z1_lambda5_prada"]:
    p = np.load(f"outputs/observations/{obs}.376/drawn_mc_pairs.npy")
    lm,c = p[:,0],p[:,1]; x=lm-lm.mean()
    c1=np.polyfit(x,c,1); r1=np.std(c-np.polyval(c1,x))
    c2=np.polyfit(x,c,2); r2=np.std(c-np.polyval(c2,x))
    # slope stability: fit on low-mass vs high-mass half
    med=np.median(x)
    b_lo=np.polyfit(x[x<med],c[x<med],1)[0]; b_hi=np.polyfit(x[x>=med],c[x>=med],1)[0]
    short=obs.replace("obs_z1_lambda5","base")
    print(f"{short:30s} lin_slope={c1[0]:+.3f} quad_coef={c2[0]:+.3f} resid lin={r1:.3f} quad={r2:.3f} "
          f"| slope_lowM={b_lo:+.3f} slope_hiM={b_hi:+.3f}")

# --- diagnostic plot: beta vs sigc for baseline & high_rm_scatter ---
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,obs in zip(axes,["obs_z1_lambda5","obs_z1_lambda5_high_rm_scatter"]):
    S=np.asarray(sbi["experiments"][obs]["samples"])
    beta_s=S[:,3]; sigc_s=S[:,4]; sigM_s=S[:,1]
    sigc_marg=np.sqrt(beta_s**2*sigM_s**2+sigc_s**2)
    bt,sigc_true,lm,c=true_beta(obs)
    sc=ax.scatter(beta_s,sigc_s,s=4,alpha=0.25,c=sigc_marg,cmap="viridis")
    ax.axvline(bt,color="r",ls="--",lw=1.5,label=f"true $\\beta$={bt:.2f}")
    ax.axhline(sigc_true,color="orange",ls=":",lw=1.5,label=f"true $\\sigma_c$={sigc_true:.2f}")
    ax.set_xlabel(r"$\beta$ (posterior samples)")
    ax.set_ylabel(r"$\sigma_c$ (posterior samples)")
    corr=np.corrcoef(beta_s,sigc_s)[0,1]
    title=obs.replace("obs_z1_lambda5","baseline").replace("_"," ")
    ax.set_title(f"{title}\n corr($\\beta,\\sigma_c$)={corr:+.2f}; "
                 f"$\\sigma_c^{{marg}}$={sigc_marg.mean():.2f}±{sigc_marg.std():.2f}")
    ax.legend(fontsize=8,loc="best")
    cb=plt.colorbar(sc,ax=ax); cb.set_label(r"$\sigma_c^{marg}=\sqrt{\beta^2\sigma_M^2+\sigma_c^2}$")
fig.suptitle(r"$\beta$–$\sigma_c$ degeneracy at fixed marginals (Hierarchical SBI posterior)",fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/beta_sigc_degeneracy.png",dpi=130)
print("\nsaved", f"{OUT}/beta_sigc_degeneracy.png")

# stash for notes
pickle.dump(dict(rows=rows,deg=deg,
   noise_corr_width=float(np.corrcoef(noises,sbi_fracs)[0,1]),
   noise_corr_bias=float(np.corrcoef(noises,sbi_bias)[0,1]),
   priors=dict(sbi=SBI_BETA_PRIOR_STD,hyb=HYBRID_BETA_PRIOR_STD,hmc=HMC_BETA_PRIOR_STD)),
   open(f"{OUT}/_scratch_beta_results.pkl","wb"))
