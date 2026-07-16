# %% [markdown]
# # Tutorial 02 — Constraining the M–c *relation* across richness bins (cheap)
# Reuses Tutorial 01's **amortized per-cluster NPE** (no new training, no per-cluster MCMC):
# the same network evaluates posteriors for clusters from *any* richness bin in milliseconds.
# With 4 bins spanning ~1 dex in mass we get the lever arm a single bin lacks, and recycle the
# cached posteriors into a fit of the relation  c = c0 + beta (log10 M - 14.4)  + scatter.
# %%
import os, sys, pickle, warnings, numpy as np, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, ROOT)
from weaklensclustersbi.simulations import population, wlprofile, populationutils
import torch, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
Z, NOISE, RBINS, SEED = 0.275, 0.3, 10 ** np.arange(0, 3, 0.1), 7
BINS = [(20, 30), (30, 45), (45, 60), (60, 100)]; PER_BIN = 12
npe = pickle.load(open("../tutorial_01_population_inference/cache/percluster_npe.pkl", "rb"))

def cached(name, fn):
    p = f"cache/{name}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    v = fn(); pickle.dump(v, open(p, "wb")); return v

def make():
    np.random.seed(SEED); out = []
    for lo, hi in BINS:
        mc = np.asarray(population.gen_mc_pairs_in_richness_bin(lo, hi,
            rm_relation="mcclintock18", mc_relation="child18", num_samples=PER_BIN,
            mc_scatter=0.1, rm_scatter=0.1, min_z=0.2, max_z=0.35))
        prof = np.log10(population.calculate_noise(np.array([
            wlprofile.simulate_nfw(m, c, RBINS, Z, kind="delta_sigma") for m, c in mc]), NOISE))
        out.append((mc, prof))
    return out
DATA = cached("obs_bins", make)

def npe_all():
    return np.array([npe.sample((3000,), x=torch.as_tensor(p, dtype=torch.float32),
                     show_progress_bars=False).numpy() for mc, prof in DATA for p in prof])
PC = cached("pc_samples", npe_all)              # (48, 3000, 2)
print("clusters:", PC.shape[0], "| mass range of truths:",
      round(min(d[0][:,0].min() for d in DATA),2), "-", round(max(d[0][:,0].max() for d in DATA),2))
# %%
S = jnp.array(PC); NB = len(BINS)
bin_idx = np.repeat(np.arange(NB), PER_BIN)
def model():
    c0   = numpyro.sample("c0",   dist.Normal(4.6, 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 2.0))
    sigc = numpyro.sample("sig_c", dist.HalfNormal(0.5))
    muM  = numpyro.sample("muM",  dist.Normal(14.2, 0.5).expand([NB]))
    sigM = numpyro.sample("sigM", dist.HalfNormal(0.3).expand([NB]))
    lm, cc = S[..., 0], S[..., 1]
    logp = (dist.Normal(muM[bin_idx][:, None], sigM[bin_idx][:, None]).log_prob(lm)
            + dist.Normal(c0 + beta * (lm - 14.4), sigc).log_prob(cc))
    numpyro.factor("L", jnp.sum(jax.scipy.special.logsumexp(logp, axis=1) - jnp.log(S.shape[1])))
def run():
    m = MCMC(NUTS(model), num_warmup=800, num_samples=1500, progress_bar=False)
    m.run(jax.random.PRNGKey(0)); s = m.get_samples()
    return {k: np.array(v) for k, v in s.items()}
FIT = cached("relation_fit", run)
gc = populationutils.get_concentration
beta_true = (gc(15.0, model="child18", z=Z) - gc(13.8, model="child18", z=Z)) / 1.2
c0_true = gc(14.4, model="child18", z=Z)
print(f"beta = {FIT['beta'].mean():+.3f} +/- {FIT['beta'].std():.3f}   (child18 local slope ~ {beta_true:+.3f})")
print(f"c0   = {FIT['c0'].mean():.3f} +/- {FIT['c0'].std():.3f}    (child18 c(14.4) = {c0_true:.3f})")
# %%
fig, ax = plt.subplots(figsize=(8, 5.5))
for (mc, _), col in zip(DATA, ["#009E73", "#E69F00", "#0072B2", "#D55E00"]):
    ax.scatter(mc[:, 0], mc[:, 1], s=18, color=col)
g = np.linspace(13.8, 15.0, 50)
for i in np.random.default_rng(0).integers(0, len(FIT["beta"]), 60):
    ax.plot(g, FIT["c0"][i] + FIT["beta"][i] * (g - 14.4), color="gray", alpha=0.12, lw=1)
ax.plot(g, [gc(m, model="child18", z=Z) for m in g], "k--", lw=2.5, label="true relation (child18)")
ax.plot([], [], color="gray", label="posterior draws of fitted relation")
ax.set_xlabel(r"$\log_{10} M$"); ax.set_ylabel("c"); ax.legend()
ax.set_title(f"M-c relation from {PC.shape[0]} clusters across {NB} richness bins\n(zero new inference: Tutorial 01's cached NPE + recycling)")
plt.show()
# %% [markdown]
# **Takeaway:** the multi-bin mass lever arm turns the unconstrained single-bin slope into a
# real measurement of beta — with *no new training and no per-cluster MCMC*: one amortized
# network + importance-sampled recycling. This is the cheap path to relation-level science.
