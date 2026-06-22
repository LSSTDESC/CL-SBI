"""
Realistic radially-varying (U-shaped) noise experiment for the hierarchical HMC.
We generate baseline-population observations but with per-bin noise that is elevated at the
smallest radii (deblending/miscentering, few sources) and largest radii (low S/N), lower in the
middle -- a U-shaped sigma(R). We then run the hierarchical model with FREE per-bin noise and ask:
(1) does it still recover the population spread sigma_M, and (2) does it recover the U-shaped noise?
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary
import hierarchical_mcmc_poc as H
jax.config.update("jax_enable_x64", True)
OUT=H.OUT_DIR; REPO=H.REPO
N_c=H.N_c; RB=np.array(H.RBINS_KPC); NR=len(RB)

# true population: reuse the baseline drawn (M,c) so only the noise model changes
true_mc=np.load(f"{REPO}/outputs/observations/obs_z1_lambda5.376/drawn_mc_pairs.npy")
# noiseless profiles via the (validated) JAX forward model
noiseless=np.array(jax.vmap(H.nfw_logSigma)(jnp.array(true_mc[:,0]), jnp.array(true_mc[:,1])))  # (N_c,NR) log10

# U-shaped per-bin noise: high at inner+outer, ~0.2 floor in middle. Parabola in bin-index.
xi=np.linspace(-1,1,NR)
sigma_true = 0.20 + 0.35*xi**2          # 0.20 dex middle -> 0.55 dex edges
rng=np.random.default_rng(0)
obs = noiseless + rng.standard_normal((N_c,NR))*sigma_true[None,:]
obs=jnp.array(obs)
print("U-shaped true noise: middle=%.2f edges=%.2f dex"%(sigma_true[NR//2], sigma_true[0]), flush=True)

def model(y=None):
    mu_M=numpyro.sample("mu_M",dist.Normal(14.4,.3)); sig_M=numpyro.sample("sig_M",dist.HalfNormal(.3))
    c0=numpyro.sample("c0",dist.Normal(4.6,1.)); beta=numpyro.sample("beta",dist.Normal(0.,2.))
    sig_c=numpyro.sample("sig_c",dist.HalfNormal(.5))
    with numpyro.plate("rbins",NR):
        sigma_noise=numpyro.sample("sigma_noise",dist.HalfNormal(0.6))  # free per-bin noise
    with numpyro.plate("clusters",N_c):
        zM=numpyro.sample("zM",dist.Normal(0,1)); logM=mu_M+sig_M*zM
        zc=numpyro.sample("zc",dist.Normal(0,1)); c=c0+beta*(logM-mu_M)+sig_c*zc
    numpyro.sample("obs",dist.Normal(H.nfw_logSigma_vmap(logM,c), sigma_noise[None,:]), obs=y)

k=NUTS(model,target_accept_prob=0.9,init_strategy=numpyro.infer.init_to_median)
mc=MCMC(k,num_warmup=800,num_samples=1000,num_chains=2,chain_method="sequential",progress_bar=False)
import time; t0=time.time(); mc.run(jax.random.PRNGKey(0),y=obs); mc.get_samples()["mu_M"].block_until_ready()
t=time.time()-t0
p=mc.get_samples()
g=mc.get_samples(group_by_chain=True)
diag=nps_summary({k_:np.array(g[k_]) for k_ in ["mu_M","sig_M","c0","sig_c"]},prob=0.9)
rhat=max(float(diag[k_]["r_hat"]) for k_ in diag)
sig_noise_post=np.array(p["sigma_noise"])
res=dict(sigma_true=sigma_true, sigma_noise_post=sig_noise_post, rbins=RB,
         sig_M=(float(p["sig_M"].mean()),float(p["sig_M"].std())),
         mu_M=(float(p["mu_M"].mean()),float(p["mu_M"].std())),
         sig_c=(float(p["sig_c"].mean()),float(p["sig_c"].std())),
         true_sigM=float(true_mc[:,0].std()), true_muM=float(true_mc[:,0].mean()),
         true_sigc=float(true_mc[:,1].std()), rhat=rhat, t=t)
pickle.dump(res,open(f"{OUT}/ushaped_noise.pkl","wb"))
print("=== U-shaped noise HBI (%.0fs, rhat=%.3f) ==="%(t,rhat))
print("sig_M=%.4f+/-%.4f (true %.4f)"%(res['sig_M'][0],res['sig_M'][1],res['true_sigM']))
print("mu_M =%.4f (true %.4f)"%(res['mu_M'][0],res['true_muM']))
print("per-bin noise RMS err=%.4f (recovered mean %.3f vs true mean %.3f)"%(
    np.sqrt(np.mean((sig_noise_post.mean(0)-sigma_true)**2)), sig_noise_post.mean(), sigma_true.mean()))
