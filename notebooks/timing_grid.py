"""
Fill missing cells of the cost grid (task x method): time SBI-JTF, SBI-FTJ separately, and
run small HMC-JTF and HMC-FTJ (single-(M,c), differentiable) for a wall-time data point.
Writes timing_grid.pkl.
"""
import time, os, pickle, numpy as np, torch, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import sys; REPO="/Users/akumgill/Documents/GitHub/CL-SBI"; sys.path.insert(0,REPO)
import hierarchical_mcmc_poc as H
from weaklensclustersbi.inference import sbi_
jax.config.update("jax_enable_x64", True)
OUT=H.OUT_DIR
INF=f"{REPO}/outputs/inference/sim_z1.infer_z1.obs_z1_lambda5.10000.376"
OBS=f"{REPO}/outputs/observations/obs_z1_lambda5.376"
obs=np.array(H.obs_profiles); sig=np.array(H.sigmas); N_c=H.N_c
res={}

# ---- SBI JTF vs FTJ, timed separately ----
post=pickle.load(open(f"{INF}/sbi_chains.pickle","rb"))  # not used for timing; need posteriors
import pickle as pk
posterior=pk.load(open(f"{REPO}/outputs/posteriors/sim_z1.infer_z1.10000.376/posterior.pickle","rb"))
posterior_jtf=pk.load(open(f"{REPO}/outputs/posteriors/sim_z1.infer_z1.10000.376/posterior_jtf.pickle","rb"))
mc=np.load(f"{OBS}/drawn_mc_pairs.npy"); prof=np.load(f"{OBS}/drawn_nfw_profiles.npy")
from weaklensclustersbi.inference.sbiutils import create_join_fit_observation_nfw, create_fit_join_observation_nfw
_,x_jtf=create_join_fit_observation_nfw(mc,prof); _,x_ftj=create_fit_join_observation_nfw(mc,prof)
# warm + time (median of 5)
def time_sbi(p,x,reps=5):
    p.sample((10000,),x=x)  # warm
    ts=[]
    for _ in range(reps):
        t=time.time(); s=p.sample((10000,),x=x); p.log_prob(s,x=x); ts.append(time.time()-t)
    return float(np.median(ts))
res['sbi_jtf']=time_sbi(posterior_jtf,x_jtf)
res['sbi_ftj']=time_sbi(posterior,x_ftj)
print(f"SBI JTF={res['sbi_jtf']:.3f}s  SBI FTJ={res['sbi_ftj']:.3f}s",flush=True)

# ---- HMC JTF (fit median profile, single (M,c)) ----
fwd=jax.jit(H.nfw_logSigma)
median_prof=jnp.array(np.median(obs,axis=0))
jtf_sig=jnp.array(np.sqrt(np.pi/2)*sig/np.sqrt(N_c))   # JTF noise reduction, matches paper
def model_jtf(y=None):
    logM=numpyro.sample("logM",dist.Uniform(13.,16.)); c=numpyro.sample("c",dist.Uniform(2.,9.))
    lnf=numpyro.sample("lnf",dist.Uniform(-10.,10.))
    mu=H.nfw_logSigma(logM,c); s2=jtf_sig**2+(jnp.exp(lnf))**2
    numpyro.sample("y",dist.Normal(mu,jnp.sqrt(s2)),obs=y)
def run_hmc(model,y,warm=600,samp=800):
    k=NUTS(model,target_accept_prob=0.9,init_strategy=numpyro.infer.init_to_median)
    m=MCMC(k,num_warmup=warm,num_samples=samp,num_chains=1,progress_bar=False)
    t=time.time(); m.run(jax.random.PRNGKey(0),y=y); m.get_samples()["logM"].block_until_ready()
    return time.time()-t,m
t_jtf,_=run_hmc(model_jtf,median_prof); res['hmc_jtf']=t_jtf
print(f"HMC JTF={t_jtf:.1f}s",flush=True)

# ---- HMC FTJ (joint likelihood over all clusters, single (M,c)) ----
obs_j=jnp.array(obs); sig_j=jnp.array(sig)
def model_ftj(y=None):
    logM=numpyro.sample("logM",dist.Uniform(13.,16.)); c=numpyro.sample("c",dist.Uniform(2.,9.))
    lnf=numpyro.sample("lnf",dist.Uniform(-10.,10.))
    mu=H.nfw_logSigma(logM,c); s2=sig_j**2+(jnp.exp(lnf))**2
    # joint: sum over all N_c clusters (each compared to the single (M,c) model)
    numpyro.sample("y",dist.Normal(mu[None,:],jnp.sqrt(s2)[None,:]),obs=y)
t_ftj,_=run_hmc(model_ftj,obs_j); res['hmc_ftj']=t_ftj
print(f"HMC FTJ={t_ftj:.1f}s",flush=True)

pickle.dump(res,open(f"{OUT}/timing_grid.pkl","wb"))
print("saved timing_grid.pkl:",res,flush=True)
