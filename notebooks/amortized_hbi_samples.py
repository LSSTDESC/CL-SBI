"""
Amortized hierarchical SBI via SAMPLE REWEIGHTING (non-Gaussian-safe).

Each cluster's SBI posterior is represented by its raw samples theta_{j,s} (carrying the full,
possibly non-Gaussian shape). Because the SBI training prior is uniform (BoxUniform), the
per-cluster marginal likelihood under population hyperparameters is, up to a constant,
    L_j(hyper) = (1/S) sum_s p_pop(theta_{j,s} | hyper),
and the population log-likelihood is sum_j log L_j. No Gaussian summary, no per-cluster latents.
We compute it with a logsumexp over each cluster's samples inside the NumPyro model.
"""
import warnings; warnings.filterwarnings("ignore")
import os, pickle, time, numpy as np, torch, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary
import sys; REPO="/Users/akumgill/Documents/GitHub/CL-SBI"; sys.path.insert(0,REPO)
import hierarchical_mcmc_poc as H
from weaklensclustersbi.inference import sbi_
jax.config.update("jax_enable_x64", True)
OUT=H.OUT_DIR
PRIORS={"min_log10mass":13.5,"max_log10mass":15.2,"min_concentration":2.0,"max_concentration":8.0}
Z=0.275; RB=10**np.arange(0,3.0,0.1); N_TRAIN=20000; NOISE=0.3; N_SAMP=300
CHILD18_BETA=-0.855

def simulate_single(theta):
    p=H.np.log10(__import__("weaklensclustersbi.simulations.wlprofile",fromlist=["simulate_nfw"]).simulate_nfw(float(theta[0]),float(theta[1]),rbins=RB,z=Z))
    return p+np.random.normal(0,NOISE,size=p.shape)

def get_samples():
    cache=f"{OUT}/per_cluster_sbi_samples.npy"
    if os.path.exists(cache):
        print("## loading cached per-cluster samples"); return np.load(cache)
    np.random.seed(0); torch.manual_seed(0)
    lo=np.array([PRIORS["min_log10mass"],PRIORS["min_concentration"]]); hi=np.array([PRIORS["max_log10mass"],PRIORS["max_concentration"]])
    theta=np.random.uniform(lo,hi,size=(N_TRAIN,2)); x=np.array([simulate_single(t) for t in theta])
    inf=sbi_.gen_inferrer(PRIORS,param_dim=2)
    de=inf.append_simulations(torch.as_tensor(theta,dtype=torch.float32),torch.as_tensor(x,dtype=torch.float32)).train(training_batch_size=100,show_train_summary=False)
    post=inf.build_posterior(de)
    obs=np.load(f"{REPO}/outputs/observations/obs_z1_lambda5.376/drawn_nfw_profiles.npy")
    S=np.array([post.sample((N_SAMP,),x=torch.as_tensor(p,dtype=torch.float32),show_progress_bars=False).numpy() for p in obs])
    np.save(cache,S); print(f"## trained+sampled, shape {S.shape}"); return S

def run(samples, beta_width=0.5, sigc_scale=0.5, warm=1000, samp=1500, chains=4, seed=0):
    S=jnp.array(samples)  # (N_c, N_SAMP, 2)
    logM_s=S[:,:,0]; c_s=S[:,:,1]
    def model():
        mu_M=numpyro.sample("mu_M",dist.Normal(14.4,0.4)); sig_M=numpyro.sample("sig_M",dist.HalfNormal(0.4))
        c0=numpyro.sample("c0",dist.Normal(4.6,1.0)); beta=numpyro.sample("beta",dist.Normal(CHILD18_BETA,beta_width))
        sig_c=numpyro.sample("sig_c",dist.HalfNormal(sigc_scale))
        # p_pop(theta_s | hyper) for each cluster-sample: logM ~ N(mu_M,sig_M); c ~ N(c0+beta(logM-mu_M),sig_c)
        lp_M = -0.5*((logM_s-mu_M)/sig_M)**2 - jnp.log(sig_M)
        lp_c = -0.5*((c_s-(c0+beta*(logM_s-mu_M)))/sig_c)**2 - jnp.log(sig_c)
        lp = lp_M + lp_c                                   # (N_c, N_SAMP) log p_pop per sample
        # L_j = mean_s p_pop  ->  log L_j = logsumexp_s(lp) - log(S); sum over clusters
        logLj = jax.scipy.special.logsumexp(lp, axis=1) - jnp.log(lp.shape[1])
        numpyro.factor("loglik", jnp.sum(logLj))
    k=NUTS(model,target_accept_prob=0.9,init_strategy=numpyro.infer.init_to_median)
    mc=MCMC(k,num_warmup=warm,num_samples=samp,num_chains=chains,chain_method="sequential",progress_bar=False)
    t=time.time(); mc.run(jax.random.PRNGKey(seed)); mc.get_samples()["mu_M"].block_until_ready(); t=time.time()-t
    p=mc.get_samples(); g=mc.get_samples(group_by_chain=True)
    diag=nps_summary({k_:np.array(g[k_]) for k_ in ["mu_M","sig_M","c0","beta","sig_c"]},prob=0.9)
    rhat=max(float(diag[k_]["r_hat"]) for k_ in diag)
    return {k_:(float(p[k_].mean()),float(p[k_].std())) for k_ in ["mu_M","sig_M","c0","beta","sig_c"]}, rhat, t

if __name__=="__main__":
    S=get_samples()
    true=np.load(f"{REPO}/outputs/observations/obs_z1_lambda5.376/drawn_mc_pairs.npy")
    summ,rhat,t=run(S)
    print("=== sample-reweighting amortized HBI (%.0fs, rhat=%.3f) ==="%(t,rhat))
    for k_,(m,s) in summ.items(): print("  %-6s = %.4f +/- %.4f"%(k_,m,s))
    print("TRUE: mu_M=%.4f sig_M=%.4f mu_c=%.4f sig_c=%.4f"%(true[:,0].mean(),true[:,0].std(),true[:,1].mean(),true[:,1].std()))
    pickle.dump({"summary":summ,"rhat":rhat,"t":t,"true_mc":true},open(f"{OUT}/amortized_hbi_samples.pkl","wb"))
