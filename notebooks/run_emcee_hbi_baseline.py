"""
Real emcee-HBI run on the baseline config, as an empirical cost/feasibility data point vs NUTS-HBI.
Runs in chunks, saving the chain + diagnostics after each chunk so partial results survive a kill.
This is expected to be SLOW and likely NOT fully converged (758-dim, affine-invariant sampler) --
that is itself the point: it quantifies why a differentiable/gradient sampler is needed.
"""
import time, os, pickle, numpy as np, jax, jax.numpy as jnp, emcee
import hierarchical_mcmc_poc as H

OUT = H.OUT_DIR
N_c = H.N_c
ndim = 2*N_c + 6
nwalkers = 2*ndim + 2
obs = np.array(H.obs_profiles); sig = np.array(H.sigmas)
true_mc = np.array(H.true_mc)
fwd = jax.jit(jax.vmap(H.nfw_logSigma))

def logprob(theta):
    mu_M, sig_M, c0, beta, sig_c, sig_extra = theta[:6]
    if sig_M<=0 or sig_c<=0 or sig_extra<=0: return -np.inf
    logM = theta[6:6+N_c]; c = theta[6+N_c:6+2*N_c]
    if np.any(c<2) or np.any(c>9) or np.any(logM<13) or np.any(logM>16): return -np.inf
    model = np.array(fwd(jnp.array(logM), jnp.array(c)))
    s2 = sig[None,:]**2 + sig_extra**2
    ll = -0.5*np.sum((obs-model)**2/s2 + np.log(s2))
    lp = -0.5*np.sum(((logM-mu_M)/sig_M)**2) - N_c*np.log(sig_M)
    lp += -0.5*np.sum(((c-(c0+beta*(logM-mu_M)))/sig_c)**2) - N_c*np.log(sig_c)
    return ll+lp

rng = np.random.default_rng(0)
base = np.concatenate([[14.4,0.12,4.6,-0.85,0.18,0.05], rng.normal(14.4,0.12,N_c), rng.normal(4.6,0.18,N_c)])
p0 = base + 1e-3*rng.standard_normal((nwalkers, ndim))

CHUNK = 2000
MAX_STEPS = 60000   # ~16-17 hr at ~1 s/step; saves every chunk
HYPER = ['mu_M','sig_M','c0','beta','sig_c','sig_extra']
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
pos = p0; total = 0; t_start = time.time()
print(f"emcee-HBI baseline: ndim={ndim}, nwalkers={nwalkers}, chunk={CHUNK}, max={MAX_STEPS}", flush=True)
while total < MAX_STEPS:
    t0 = time.time()
    pos, _, _ = sampler.run_mcmc(pos, CHUNK, progress=False)
    total += CHUNK
    dt = time.time()-t0
    ch = sampler.get_chain()
    try: tau = float(np.nanmax(sampler.get_autocorr_time(tol=0)))
    except Exception: tau = float('nan')
    # burn 30%, flatten, summarize hyperparams
    b = int(total*0.3); flat = ch[b:].reshape(-1, ndim)
    summ = {nm:(float(flat[:,i].mean()), float(flat[:,i].std())) for i,nm in enumerate(HYPER)}
    rec = dict(total_steps=total, nwalkers=nwalkers, ndim=ndim, s_per_step=dt/CHUNK,
               wall_s=time.time()-t_start, tau_max=tau, accept=float(np.mean(sampler.acceptance_fraction)),
               summary=summ, true_sigM=float(true_mc[:,0].std()), true_muM=float(true_mc[:,0].mean()),
               true_sigc=float(true_mc[:,1].std()), true_muc=float(true_mc[:,1].mean()))
    pickle.dump(rec, open(f"{OUT}/emcee_hbi_baseline_progress.pkl","wb"))
    np.save(f"{OUT}/emcee_hbi_baseline_chain.npy", ch[-1])  # last positions only (small)
    print(f"[{total:6d} steps | {rec['wall_s']/3600:.2f} hr] accept={rec['accept']:.3f} tau~{tau:.0f} "
          f"sig_M={summ['sig_M'][0]:.3f}(true {rec['true_sigM']:.3f}) mu_M={summ['mu_M'][0]:.3f}", flush=True)
print("DONE", flush=True)
