"""
Full-dataset hierarchical HMC: the realistic McClintock18 (richness x redshift) grid.

The single-bin experiments cannot constrain the M-c slope beta (no mass lever arm) or its redshift
evolution. Here we fit the FULL realistic dataset: the 4 richness bins (lambda>=20, McClintock18's
mass-calibration floor) x 3 redshift bins = 12 cells, with the actual McClintock18 per-cell cluster
counts (6504 clusters total; the lambda<20 bins are below McClintock's floor and excluded). This
spans ~1.3 dex in mass and z=0.2-0.65, giving a real lever arm for beta AND a baseline to measure the
concentration's redshift evolution.

Model: ONE shared M-c relation across all 12 cells with redshift evolution,
    c = c0 + beta*(logM - logM_ref) + gamma*(z - z_ref) + N(0, sig_c),   z_ref=0.35, logM_ref=14.3
each cell has its own mass-function x richness-selection population (the validated machinery) at the
cell's redshift, and the cell's redshift sets the forward model (Delta_vir(z) + NFW geometry).
Recovers (mu_M per cell implicitly, c0, beta, gamma, sig_c). This is the Figure-3 analog for the
full dataset, HMC only (first); repeat for other methods if it works.

Run under base env (heavy: 6504 latents x NUTS):
  /Users/akumgill/anaconda3/bin/python notebooks/hier_fulldataset_hmc.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, pickle, numpy as np
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as nps_summary
from scipy.stats import norm as scipy_norm
import sys
REPO = "/Users/akumgill/Documents/GitHub/CL-SBI"; sys.path.insert(0, REPO)
import hierarchical_mcmc_poc as H
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.halo import mass_so
from weaklensclustersbi.simulations import populationutils
from halox.halo import NFWHalo
jax.config.update("jax_enable_x64", True)

OUT = H.OUT_DIR; P18 = H.P18; RBINS_KPC = H.RBINS_KPC
cosmology.setCosmology("planck18")
F = float(populationutils.get_rm_slope("mcclintock18")); RM_SCATTER = 0.1
Z_REF, LOGM_REF = 0.35, 14.3

# 12 cells: (z_tag, z_mid, lambda_tag, (lmin,lmax), N) -- McClintock18 counts, lambda>=20
CELLS = [
    ("z1", 0.275, "lambda4", (20, 30), 762),  ("z1", 0.275, "lambda5", (30, 45), 376),
    ("z1", 0.275, "lambda6", (45, 60), 123),  ("z1", 0.275, "lambda7", (60, 200), 91),
    ("z2", 0.425, "lambda4", (20, 30), 1549), ("z2", 0.425, "lambda5", (30, 45), 672),
    ("z2", 0.425, "lambda6", (45, 60), 187),  ("z2", 0.425, "lambda7", (60, 200), 148),
    ("z3", 0.575, "lambda4", (20, 30), 1612), ("z3", 0.575, "lambda5", (30, 45), 687),
    ("z3", 0.575, "lambda6", (45, 60), 205),  ("z3", 0.575, "lambda7", (60, 200), 92),
]


def cell_mf_logpdf(lmin, lmax, zc):
    grid = np.linspace(13.3, 15.4, 1200)
    dndlnM = mass_function.massFunction(10 ** grid, zc, mdef="200m", model="tinker08", q_out="dndlnM")
    sig_lnlam = RM_SCATTER * np.log(10.0) / F
    lam = populationutils.get_richness(grid, z=zc, model="mcclintock18"); ll = np.log(lam)
    pin = scipy_norm.cdf(np.log(lmax), ll, sig_lnlam) - scipy_norm.cdf(np.log(lmin), ll, sig_lnlam)
    pdf = np.clip(dndlnM * np.log(10.0) * np.clip(pin, 1e-12, None), 1e-300, None)
    lp = np.log(pdf); lp -= np.log(np.trapz(np.exp(lp), grid))
    return grid, lp


# z-aware differentiable NFW forward model (Delta_vir(z) per cell)
def make_fwd(z, delta_vir):
    def nfw_logSigma(logM, c):
        halo = NFWHalo(m_delta=10.0 ** logM, c_delta=c, z=z, cosmo=P18, delta=delta_vir)
        Rs = halo.Rs * 1000.0; rho0 = halo.rho0 / 1e9
        x = RBINS_KPC / Rs
        x_lo = jnp.where(x < 1, x, 0.5); x_hi = jnp.where(x > 1, x, 2.0)
        f_lo = (1 - 2 * jnp.arctanh(jnp.sqrt((1 - x_lo) / (1 + x_lo))) / jnp.sqrt(1 - x_lo**2)) / (x_lo**2 - 1)
        f_hi = (1 - 2 * jnp.arctan(jnp.sqrt((x_hi - 1) / (1 + x_hi))) / jnp.sqrt(x_hi**2 - 1)) / (x_hi**2 - 1)
        f = jnp.where(x < 1, f_lo, jnp.where(x > 1, f_hi, 1.0 / 3.0))
        return jnp.log10(2 * rho0 * Rs * f)
    return jax.vmap(nfw_logSigma)


print("## loading 12 McClintock cells + per-cell MF priors + z-aware forward models ...", flush=True)
LOADED = []
allM, allc, allz = [], [], []
for ztag, zc, ltag, (lmin, lmax), N in CELLS:
    d = f"{REPO}/outputs/observations/{ztag.replace('z','obs_z')}_{ltag}.{N}"
    d = f"{REPO}/outputs/observations/obs_{ztag}_{ltag}.{N}"
    prof = jnp.array(np.load(f"{d}/drawn_nfw_profiles.npy")); sig = jnp.array(np.load(f"{d}/sigmas.npy"))
    tmc = np.load(f"{d}/drawn_mc_pairs.npy")
    grid, lp = cell_mf_logpdf(lmin, lmax, zc)
    dvir = float(mass_so.deltaVir(zc))
    LOADED.append(dict(prof=prof, sig=sig, n=prof.shape[0], zc=zc, fwd=make_fwd(zc, dvir),
                       grid=jnp.array(grid), lp=jnp.array(lp), mode=float(grid[np.argmax(lp)])))
    allM.append(tmc[:, 0]); allc.append(tmc[:, 1]); allz.append(np.full(len(tmc), zc))
N_TOT = sum(c["n"] for c in LOADED)
allM = np.concatenate(allM); allc = np.concatenate(allc); allz = np.concatenate(allz)
A = np.column_stack([np.ones_like(allM), allM - LOGM_REF, allz - Z_REF])
TRUE_C0, TRUE_BETA, TRUE_GAMMA = np.linalg.lstsq(A, allc, rcond=None)[0]
print(f"## N_tot={N_TOT}; TRUE c0={TRUE_C0:.3f} beta={TRUE_BETA:.3f} gamma={TRUE_GAMMA:.3f}", flush=True)


def model():
    c0 = numpyro.sample("c0", dist.Normal(4.6, 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.5))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 3.0))     # redshift evolution of concentration
    sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.5))
    sig_extra = numpyro.sample("sig_extra", dist.HalfNormal(0.15))
    for k, b in enumerate(LOADED):
        dmu = numpyro.sample(f"dmu_{k}", dist.Normal(0.0, 0.15))
        with numpyro.plate(f"cl_{k}", b["n"]):
            zM = numpyro.sample(f"zM_{k}", dist.Normal(0, 1))
            logM = b["mode"] + dmu + 0.15 * zM
            numpyro.factor(f"mf_{k}", jnp.interp(logM, b["grid"], b["lp"]))
            zc_ = numpyro.sample(f"zc_{k}", dist.Normal(0, 1))
            c = c0 + beta * (logM - LOGM_REF) + gamma * (b["zc"] - Z_REF) + sig_c * zc_
        sig_tot = jnp.sqrt(b["sig"][None, :] ** 2 + sig_extra ** 2)
        numpyro.sample(f"obs_{k}", dist.Normal(b["fwd"](logM, c), sig_tot), obs=b["prof"])


def run(warm=600, samp=800, chains=2, seed=0):
    mc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=numpyro.infer.init_to_median),
              num_warmup=warm, num_samples=samp, num_chains=chains, chain_method="sequential", progress_bar=True)
    t = time.time(); mc.run(jax.random.PRNGKey(seed)); mc.get_samples()["beta"].block_until_ready(); t = time.time() - t
    p = mc.get_samples(); g = mc.get_samples(group_by_chain=True)
    keys = ["c0", "beta", "gamma", "sig_c", "sig_extra"]
    diag = nps_summary({k: np.array(g[k]) for k in keys}, prob=0.9)
    rhat = max(float(diag[k]["r_hat"]) for k in keys)
    return {k: (float(p[k].mean()), float(p[k].std())) for k in keys}, rhat, t, p


if __name__ == "__main__":
    t0 = time.time()
    est, rhat, t, post = run()
    print(f"\n=== FULL-DATASET HMC (N_tot={N_TOT}, 12 cells, rhat={rhat:.3f}, {t/60:.1f} min) ===")
    for k in ["c0", "beta", "gamma", "sig_c"]:
        tv = {"c0": TRUE_C0, "beta": TRUE_BETA, "gamma": TRUE_GAMMA, "sig_c": float(allc.std())}[k]
        print(f"  {k:7s} = {est[k][0]:7.3f} +/- {est[k][1]:.3f}   (true {tv:.3f})")
    pickle.dump(dict(est=est, rhat=rhat, t=t, n_tot=N_TOT,
                     true=dict(c0=float(TRUE_C0), beta=float(TRUE_BETA), gamma=float(TRUE_GAMMA)),
                     samples={k: np.array(post[k]) for k in ["c0", "beta", "gamma", "sig_c"]}),
                open(f"{OUT}/fulldataset_hmc.pkl", "wb"))
    print(f"\n=== total wall {(time.time()-t0)/60:.1f} min === saved -> {OUT}/fulldataset_hmc.pkl")
