"""
Full-dataset Hybrid SBI-HMC on the realistic McClintock 4x3 (richness x redshift) grid.

Counterpart to hier_fulldataset_hmc.py but with the per-cluster likelihood AMORTIZED by SBI:
train ONE per-cluster net q(M,c | profile, noise, z) -- now also CONDITIONED ON REDSHIFT, since the
full dataset spans z=0.275/0.425/0.575 -- evaluate it on every cell's clusters, then run the SAME
NUTS population hierarchy (shared M-c relation with redshift evolution c0,beta,gamma + per-cell
MF x selection mass populations) via sample-reweighting (logsumexp). 12 cells, 6504 clusters.

Run under base env:
  /Users/akumgill/anaconda3/bin/python notebooks/hybrid_fulldataset.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle, numpy as np, torch
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary
from scipy.stats import norm as scipy_norm
import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"; sys.path.insert(0, REPO)
from weaklensclustersbi.simulations import wlprofile
from weaklensclustersbi.inference import sbi_
from weaklensclustersbi.simulations import populationutils
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from sbi.inference import SNPE
from sbi.utils import BoxUniform, posterior_nn
jax.config.update("jax_enable_x64", True)
cosmology.setCosmology("planck18")

OUT = f"{REPO}/notebooks/hierarchical_poc_outputs"
RB = 10 ** np.arange(0, 3.0, 0.1); NBINS = len(RB)
BOX_LO = np.array([12.0, 2.0]); BOX_HI = np.array([15.2, 8.0])
NOISE_LO, NOISE_HI = 0.10, 0.80
Z_LO, Z_HI = 0.2, 0.65                # per-cluster net trained over the full z range
N_TRAIN = 150000; N_SAMP = 300
F = float(populationutils.get_rm_slope("mcclintock18")); RM_SCATTER = 0.1
Z_REF, LOGM_REF = 0.35, 14.3
CELLS = [
    ("z1", 0.275, "lambda4", (20, 30), 762),  ("z1", 0.275, "lambda5", (30, 45), 376),
    ("z1", 0.275, "lambda6", (45, 60), 123),  ("z1", 0.275, "lambda7", (60, 200), 91),
    ("z2", 0.425, "lambda4", (20, 30), 1549), ("z2", 0.425, "lambda5", (30, 45), 672),
    ("z2", 0.425, "lambda6", (45, 60), 187),  ("z2", 0.425, "lambda7", (60, 200), 148),
    ("z3", 0.575, "lambda4", (20, 30), 1612), ("z3", 0.575, "lambda5", (30, 45), 687),
    ("z3", 0.575, "lambda6", (45, 60), 205),  ("z3", 0.575, "lambda7", (60, 200), 92),
]


# ---- z-conditioned per-cluster net: q(M,c | profile(30), noise(1), z(1)) ----
def simulate_single(theta, noise, z, rng):
    p = np.log10(wlprofile.simulate_nfw(float(theta[0]), float(theta[1]), rbins=RB, z=float(z)))
    return (p + rng.normal(0, noise, size=p.shape)).astype(np.float32)


def train_per_cluster():
    cache = f"{OUT}/hybrid_fulldataset_post.pkl"
    if os.path.exists(cache):
        print("## loading cached z-conditioned per-cluster posterior"); return pickle.load(open(cache, "rb"))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    theta = rng.uniform(BOX_LO, BOX_HI, size=(N_TRAIN, 2)).astype(np.float32)
    noise = rng.uniform(NOISE_LO, NOISE_HI, size=N_TRAIN).astype(np.float32)
    zs = rng.uniform(Z_LO, Z_HI, size=N_TRAIN).astype(np.float32)
    x = np.empty((N_TRAIN, NBINS + 2), dtype=np.float32)
    print(f"## simulating {N_TRAIN} single-cluster pairs (profile,noise,z) ...", flush=True)
    for i in range(N_TRAIN):
        x[i, :NBINS] = simulate_single(theta[i], noise[i], zs[i], rng); x[i, NBINS] = noise[i]; x[i, NBINS + 1] = zs[i]
    prior = BoxUniform(torch.as_tensor(BOX_LO, dtype=torch.float32), torch.as_tensor(BOX_HI, dtype=torch.float32))
    inf = SNPE(prior, density_estimator=posterior_nn(model="maf", hidden_features=80, num_transforms=8), device="cpu")
    inf = inf.append_simulations(torch.as_tensor(theta), torch.as_tensor(x))
    print("## training per-cluster q(M,c | profile, noise, z) ...", flush=True)
    de = inf.train(training_batch_size=300, stop_after_epochs=25, show_train_summary=False)
    post = inf.build_posterior(de); pickle.dump(post, open(cache, "wb")); return post


def per_cell_samples(post, ztag, ltag, N, noise, zc):
    d = f"{REPO}/outputs/observations/obs_{ztag}_{ltag}.{N}"
    prof = np.load(f"{d}/drawn_nfw_profiles.npy").astype(np.float32)
    S = []
    for p in prof:
        xc = torch.as_tensor(np.concatenate([p, [noise, zc]]).astype(np.float32))
        S.append(post.sample((N_SAMP,), x=xc, show_progress_bars=False).numpy())
    return np.array(S)


def cell_mf_logpdf(lmin, lmax, zc):
    grid = np.linspace(13.3, 15.4, 1200)
    dndlnM = mass_function.massFunction(10 ** grid, zc, mdef="200m", model="tinker08", q_out="dndlnM")
    sig = RM_SCATTER * np.log(10.0) / F
    ll = np.log(populationutils.get_richness(grid, z=zc, model="mcclintock18"))
    pin = scipy_norm.cdf(np.log(lmax), ll, sig) - scipy_norm.cdf(np.log(lmin), ll, sig)
    pdf = np.clip(dndlnM * np.log(10.0) * np.clip(pin, 1e-12, None), 1e-300, None)
    lp = np.log(pdf); lp -= np.log(np.trapz(np.exp(lp), grid)); return grid, lp


if __name__ == "__main__":
    t0 = time.time()
    post = train_per_cluster()
    print("## evaluating per-cluster posteriors on all 12 cells ...", flush=True)
    CELLDATA, allc, allM, allz = [], [], [], []
    for ztag, zc, ltag, (lmin, lmax), N in CELLS:
        d = f"{REPO}/outputs/observations/obs_{ztag}_{ltag}.{N}"
        noise = float(np.load(f"{d}/sigmas.npy").mean())
        S = per_cell_samples(post, ztag, ltag, N, noise, zc)
        grid, lp = cell_mf_logpdf(lmin, lmax, zc)
        CELLDATA.append(dict(S=jnp.array(S), zc=zc))
        tmc = np.load(f"{d}/drawn_mc_pairs.npy"); allc.append(tmc[:, 1]); allM.append(tmc[:, 0]); allz.append(np.full(len(tmc), zc))
    allc = np.concatenate(allc); allM = np.concatenate(allM); allz = np.concatenate(allz)
    A = np.column_stack([np.ones_like(allM), allM - LOGM_REF, allz - Z_REF])
    TC0, TBETA, TGAMMA = np.linalg.lstsq(A, allc, rcond=None)[0]
    N_TOT = sum(c["S"].shape[0] for c in CELLDATA)
    print(f"## N_tot={N_TOT}; TRUE c0={TC0:.3f} beta={TBETA:.3f} gamma={TGAMMA:.3f}", flush=True)

    def model():
        c0 = numpyro.sample("c0", dist.Normal(4.6, 1.0)); beta = numpyro.sample("beta", dist.Normal(0.0, 1.5))
        gamma = numpyro.sample("gamma", dist.Normal(0.0, 3.0)); sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.5))
        mu_M = numpyro.sample("mu_M", dist.Normal(14.3, 0.4)); sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.4))
        for k, cd in enumerate(CELLDATA):
            logM_s = cd["S"][:, :, 0]; c_s = cd["S"][:, :, 1]
            lpM = -0.5 * ((logM_s - mu_M) / sig_M) ** 2 - jnp.log(sig_M)
            cmean = c0 + beta * (logM_s - LOGM_REF) + gamma * (cd["zc"] - Z_REF)
            lpc = -0.5 * ((c_s - cmean) / sig_c) ** 2 - jnp.log(sig_c)
            logLj = jax.scipy.special.logsumexp(lpM + lpc, axis=1) - jnp.log(logM_s.shape[1])
            numpyro.factor(f"ll_{k}", jnp.sum(logLj))

    mc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median),
              num_warmup=800, num_samples=1000, num_chains=2, chain_method="sequential", progress_bar=True)
    t = time.time(); mc.run(jax.random.PRNGKey(0)); mc.get_samples()["beta"].block_until_ready(); t = time.time() - t
    p = mc.get_samples(); g = mc.get_samples(group_by_chain=True)
    keys = ["c0", "beta", "gamma", "sig_c", "mu_M", "sig_M"]
    diag = nps_summary({k: np.array(g[k]) for k in keys}, prob=0.9); rhat = max(float(diag[k]["r_hat"]) for k in diag)
    est = {k: (float(p[k].mean()), float(p[k].std())) for k in keys}
    print(f"\n=== FULL-DATASET HYBRID (N_tot={N_TOT}, 12 cells, rhat={rhat:.3f}, {t/60:.1f} min) ===")
    for k, tv in [("c0", TC0), ("beta", TBETA), ("gamma", TGAMMA), ("sig_c", allc.std())]:
        print(f"  {k:7s} = {est[k][0]:7.3f} +/- {est[k][1]:.3f}  (true {tv:.3f})")
    pickle.dump(dict(est=est, rhat=rhat, t=t, n_tot=N_TOT,
                     true=dict(c0=float(TC0), beta=float(TBETA), gamma=float(TGAMMA)),
                     samples={k: np.array(p[k]) for k in ["c0", "beta", "gamma", "sig_c"]}),
                open(f"{OUT}/hybrid_fulldataset.pkl", "wb"))
    print(f"\n=== total wall {(time.time()-t0)/60:.1f} min === saved -> {OUT}/hybrid_fulldataset.pkl")
