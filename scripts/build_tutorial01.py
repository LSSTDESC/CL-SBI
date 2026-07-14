"""Emit notebooks/tutorial_01_population_inference/tutorial_01_population_inference.ipynb"""
import json, os

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}

C = []

C.append(md("""# Tutorial 01 — Population inference on 5 clusters, three ways

**The core idea of the paper in one notebook.** We take **five** simulated cluster weak-lensing
observations and infer the *population* mass–concentration distribution with three methods:

| color | method | per-cluster step | population step |
|---|---|---|---|
| 🟢 green | **Hierarchical HMC** | sampled jointly (latents) | NUTS on hyperparameters (differentiable model) |
| 🟤 brown | **Hybrid: SBI + recycling** | amortized SBI (NPE) posteriors, cached | reweight cached samples (importance sampling) |
| 🟣 purple | **Full hierarchical SBI** | — (end-to-end) | neural posterior trained to map 5 profiles → hyperparameters |

All three answer the same question with the same priors — the punchline is that they **agree**.

*Runtime:* first fresh run ≈ 10–20 min (SBI training dominates); everything is cached in `cache/`,
so re-runs take seconds. Delete `cache/` to recompute from scratch. The 5 observations live in
`data/` (committed, fixed seed) so everyone runs on identical inputs."""))

C.append(code("""# --- setup: imports, paths, cache helper -------------------------------------
import os, sys, time, pickle, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

# repo root = two levels up from this notebook
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
assert os.path.isdir(os.path.join(ROOT, "weaklensclustersbi")), f"repo root not found from {os.getcwd()}"
sys.path.insert(0, ROOT)

DATA, CACHE = "data", "cache"
os.makedirs(DATA, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
TIMINGS = {}

def cached(name, fn, where=CACHE):
    \"\"\"Load `where`/name.pkl if present, else compute fn(), save, and record timing.\"\"\"
    path = os.path.join(where, f"{name}.pkl")
    if os.path.isfile(path):
        TIMINGS[name] = ("cached", None)
        with open(path, "rb") as f: return pickle.load(f)
    t0 = time.perf_counter(); val = fn(); dt = time.perf_counter() - t0
    TIMINGS[name] = ("computed", dt)
    with open(path, "wb") as f: pickle.dump(val, f)
    print(f"[{name}] computed in {dt:.1f}s (cached for next run)")
    return val

# package machinery (imported, not re-implemented -- see the paper / repo)
from weaklensclustersbi.simulations import population, wlprofile

N_CLUSTERS = 5          # scale this up later (e.g. 10, 50) -- everything else adapts
NOISE_DEX  = 0.3        # per-radial-bin log-normal noise (paper baseline)
Z          = 0.275      # mid of the 0.2 < z < 0.35 bin
RBINS      = 10 ** np.arange(0, 3, 0.1)   # 30 log-spaced radii [kpc/h]
SEED       = 11"""))

C.append(md("""## 1. The data: five clusters from one richness bin

We draw 5 clusters from the paper's baseline population — the $30<\\lambda<45$ richness bin
(McClintock+18 richness–mass relation, Child+18 concentration–mass relation, Tinker08 mass
function weighting) — and observe each as a noisy **excess surface density** $\\Delta\\Sigma(R)$
profile (0.3 dex log-normal noise per radial bin), exactly as in the paper."""))

C.append(code("""def make_observations():
    np.random.seed(SEED)
    for _ in range(5):  # retry: the richness-bin sampler very occasionally hiccups
        try:
            pairs = np.asarray(population.gen_mc_pairs_in_richness_bin(
                30, 45, rm_relation="mcclintock18", mc_relation="child18",
                num_samples=N_CLUSTERS, mc_scatter=0.1, rm_scatter=0.1,
                min_z=0.2, max_z=0.35))
            break
        except TypeError:
            continue
    profs = np.array([wlprofile.simulate_nfw(m, c, RBINS, Z, kind="delta_sigma") for m, c in pairs])
    noisy = np.log10(population.calculate_noise(profs, NOISE_DEX))
    return {"true_mc": pairs, "log_profiles": noisy}

def make_truth_population():
    np.random.seed(4242)
    pop = np.asarray(population.gen_mc_pairs_in_richness_bin(
        30, 45, rm_relation="mcclintock18", mc_relation="child18",
        num_samples=20000, mc_scatter=0.1, rm_scatter=0.1, min_z=0.2, max_z=0.35))
    return pop

OBS  = cached("observations", make_observations, where=DATA)
POP  = cached("truth_population", make_truth_population, where=DATA)
TRUE = dict(mu_M=POP[:,0].mean(), sig_M=POP[:,0].std(), mu_c=POP[:,1].mean(), sig_c=POP[:,1].std())

print("The five clusters (truth):")
for i, (m, c) in enumerate(OBS["true_mc"]):
    print(f"  cluster {i}:  log10 M = {m:.3f}   c = {c:.3f}")
print(f"TRUE population:  mu_M={TRUE['mu_M']:.3f}  sig_M={TRUE['sig_M']:.3f}  mu_c={TRUE['mu_c']:.3f}  sig_c={TRUE['sig_c']:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
for i in range(N_CLUSTERS):
    ax.plot(RBINS, 10 ** OBS["log_profiles"][i], alpha=0.8, label=f"cluster {i}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("radius [kpc/h]"); ax.set_ylabel(r"$\\Delta\\Sigma$ [$M_\\odot h/kpc^2$]")
ax.set_title("The data: 5 noisy $\\\\Delta\\\\Sigma$ profiles"); ax.legend(fontsize=8); plt.show()"""))

C.append(md("""## 2. The per-cluster problem

One noisy profile constrains $(\\log_{10}M, c)$ only weakly, with a strong mass–concentration
degeneracy. We train a small **amortized SBI posterior** (NPE, flat prior over a wide box) once —
it then evaluates the posterior for *any* cluster in milliseconds. (This same network is reused by
the 🟤 hybrid method below.)"""))

C.append(code("""import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

BOX_LO, BOX_HI = [13.4, 2.0], [15.4, 8.0]      # flat NPE prior (simplifies recycling: it cancels)

def train_percluster_npe(n_train=4000):
    torch.manual_seed(SEED); np.random.seed(SEED)
    prior = BoxUniform(low=torch.tensor(BOX_LO), high=torch.tensor(BOX_HI))
    theta = prior.sample((n_train,)).numpy()
    x = np.array([np.log10(population.calculate_noise(
            wlprofile.simulate_nfw(m, c, RBINS, Z, kind="delta_sigma"), NOISE_DEX))
        for m, c in theta])
    inf = SNPE(prior=prior, density_estimator="maf")
    de = inf.append_simulations(torch.as_tensor(theta, dtype=torch.float32),
                                torch.as_tensor(x, dtype=torch.float32)).train(show_train_summary=False)
    return inf.build_posterior(de)

npe = cached("percluster_npe", train_percluster_npe)

def percluster_samples(n=2000):
    out = []
    for i in range(N_CLUSTERS):
        s = npe.sample((n,), x=torch.as_tensor(OBS["log_profiles"][i], dtype=torch.float32),
                       show_progress_bars=False).numpy()
        out.append(s)
    return np.array(out)                        # (N_CLUSTERS, n, 2)

PC = cached("percluster_samples", percluster_samples)
np.savez(os.path.join(DATA, "percluster_posteriors.npz"), samples=PC, true_mc=OBS["true_mc"])  # shared deliverable

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(PC[0][:, 0], PC[0][:, 1], s=3, alpha=0.15, color="gray", label="cluster-0 posterior (SBI)")
ax.scatter(*OBS["true_mc"][0], marker="*", s=250, color="red", edgecolor="k", zorder=5, label="cluster-0 truth")
ax.scatter(POP[::40, 0], POP[::40, 1], s=2, alpha=0.15, color="C0", label="true population")
ax.set_xlabel(r"$\\log_{10} M$"); ax.set_ylabel("concentration"); ax.legend()
ax.set_title("One cluster tells you little -- hence: population inference"); plt.show()"""))

C.append(md("""## 3. The population question

We model the population as independent Gaussians,
$\\log_{10}M \\sim \\mathcal{N}(\\mu_M, \\sigma_M)$ and $c \\sim \\mathcal{N}(\\mu_c, \\sigma_c)$,
and infer the four **hyperparameters** $(\\mu_M, \\sigma_M, \\mu_c, \\sigma_c)$.
All three methods use the **same hyperprior**:
$\\mu_M \\sim \\mathcal{N}(14.4, 0.3)$, $\\sigma_M \\sim \\mathrm{HalfNormal}(0.25)$,
$\\mu_c \\sim \\mathcal{N}(4.6, 0.8)$, $\\sigma_c \\sim \\mathrm{HalfNormal}(0.5)$."""))

C.append(code("""# --- method 1 (green): hierarchical HMC -- the joint generative model ---------
import jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from weaklensclustersbi.simulations.wlprofile_jax import nfw_logDeltaSigma_vmap, nfw_logDeltaSigma

# guardrail: the differentiable forward model must match the generator (colossus)
ref = np.log10(wlprofile.simulate_nfw(14.4, 4.6, RBINS, Z, kind="delta_sigma"))
assert np.max(np.abs(10 ** np.array(nfw_logDeltaSigma(14.4, 4.6)) / 10 ** ref - 1)) < 1e-3
print("JAX forward model validated against the data-generating pipeline (<0.1%)")

def green_model(log_profiles):
    mu_M  = numpyro.sample("mu_M",  dist.Normal(14.4, 0.3))
    sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.25))
    mu_c  = numpyro.sample("mu_c",  dist.Normal(4.6, 0.8))
    sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.5))
    # per-cluster latents (non-centered)
    zM = numpyro.sample("zM", dist.Normal(0, 1).expand([N_CLUSTERS]))
    zc = numpyro.sample("zc", dist.Normal(0, 1).expand([N_CLUSTERS]))
    logM = mu_M + sig_M * zM
    c    = jnp.clip(mu_c + sig_c * zc, 0.5, 12.0)
    mu_prof = nfw_logDeltaSigma_vmap(logM, c)                  # (N_CLUSTERS, 30)
    numpyro.sample("obs", dist.Normal(mu_prof, NOISE_DEX), obs=log_profiles)

def run_green():
    mcmc = MCMC(NUTS(green_model), num_warmup=800, num_samples=1500, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), jnp.array(OBS["log_profiles"]))
    s = mcmc.get_samples()
    return {k: np.array(s[k]) for k in ("mu_M", "sig_M", "mu_c", "sig_c")}

GREEN = cached("green_hmc", run_green)
print({k: f"{v.mean():.3f}+/-{v.std():.3f}" for k, v in GREEN.items()})"""))

C.append(code("""# --- method 2 (brown): recycle the cached per-cluster SBI posteriors ----------
# GW-style importance sampling: with a FLAT per-cluster training prior, the population
# likelihood is just the average population density over each cluster's posterior samples:
#   L_j(hyper) = (1/S) sum_s  N(logM_js | mu_M, sig_M) * N(c_js | mu_c, sig_c)
SAMPS = jnp.array(PC)                                            # (N_CLUSTERS, S, 2)

def brown_model():
    mu_M  = numpyro.sample("mu_M",  dist.Normal(14.4, 0.3))
    sig_M = numpyro.sample("sig_M", dist.HalfNormal(0.25))
    mu_c  = numpyro.sample("mu_c",  dist.Normal(4.6, 0.8))
    sig_c = numpyro.sample("sig_c", dist.HalfNormal(0.5))
    logp = (dist.Normal(mu_M, sig_M).log_prob(SAMPS[..., 0])
            + dist.Normal(mu_c, sig_c).log_prob(SAMPS[..., 1]))   # (N_CLUSTERS, S)
    logL = jnp.sum(jax.scipy.special.logsumexp(logp, axis=1) - jnp.log(SAMPS.shape[1]))
    numpyro.factor("recycled_likelihood", logL)

def run_brown():
    mcmc = MCMC(NUTS(brown_model), num_warmup=800, num_samples=1500, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1))
    s = mcmc.get_samples()
    return {k: np.array(s[k]) for k in ("mu_M", "sig_M", "mu_c", "sig_c")}

BROWN = cached("brown_recycled", run_brown)
print({k: f"{v.mean():.3f}+/-{v.std():.3f}" for k, v in BROWN.items()})"""))

C.append(code("""# --- method 3 (purple): full hierarchical SBI -- profiles in, hyperparameters out
def train_population_npe(n_train=6000):
    torch.manual_seed(SEED + 1); rng = np.random.default_rng(SEED + 1)
    thetas, xs = [], []
    while len(thetas) < n_train:
        hyper = [rng.normal(14.4, 0.3), abs(rng.normal(0, 0.25)) + 1e-3,
                 rng.normal(4.6, 0.8), abs(rng.normal(0, 0.5)) + 1e-3]
        logM = rng.normal(hyper[0], hyper[1], N_CLUSTERS)
        c    = np.clip(rng.normal(hyper[2], hyper[3], N_CLUSTERS), 0.5, 12.0)
        if logM.min() < 12.5 or logM.max() > 16.0:   # keep the forward model in sane range
            continue
        profs = np.array([wlprofile.simulate_nfw(m, cc, RBINS, Z, kind="delta_sigma")
                          for m, cc in zip(logM, c)])
        x = np.log10(population.calculate_noise(profs, NOISE_DEX))
        x = x[np.argsort(x.mean(axis=1))].ravel()    # sort rows: permutation-stable input
        thetas.append(hyper); xs.append(x)
    prior = BoxUniform(low=torch.tensor([13.0, 0.0, 2.0, 0.0]),
                       high=torch.tensor([15.5, 1.0, 8.0, 2.0]))
    inf = SNPE(prior=prior, density_estimator="maf")
    de = inf.append_simulations(torch.as_tensor(np.array(thetas), dtype=torch.float32),
                                torch.as_tensor(np.array(xs), dtype=torch.float32)).train(show_train_summary=False)
    return inf.build_posterior(de)

pop_npe = cached("population_npe", train_population_npe)

def run_purple(n=4000):
    x_obs = OBS["log_profiles"][np.argsort(OBS["log_profiles"].mean(axis=1))].ravel()
    s = pop_npe.sample((n,), x=torch.as_tensor(x_obs, dtype=torch.float32),
                       show_progress_bars=False).numpy()
    return {"mu_M": s[:, 0], "sig_M": s[:, 1], "mu_c": s[:, 2], "sig_c": s[:, 3]}

PURPLE = cached("purple_samples", run_purple)
print({k: f"{v.mean():.3f}+/-{v.std():.3f}" for k, v in PURPLE.items()})"""))

C.append(md("""## 4. Compare: three posteriors, one truth"""))

C.append(code("""METHODS = [("Hierarchical HMC", GREEN, "green"), ("Hybrid SBI + recycling", BROWN, "saddlebrown"),
           ("Full hierarchical SBI", PURPLE, "purple")]
PARAMS  = [("mu_M", r"$\\mu_{\\log M}$"), ("sig_M", r"$\\sigma_{\\log M}$"),
           ("mu_c", r"$\\mu_c$"), ("sig_c", r"$\\sigma_c$")]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (key, label) in zip(axes.ravel(), PARAMS):
    for name, s, color in METHODS:
        ax.hist(s[key], bins=40, density=True, histtype="step", lw=2, color=color, label=name)
    ax.axvline(TRUE[key], color="k", ls="--", lw=2, label="truth (population)")
    ax.set_xlabel(label); ax.set_yticks([])
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"Population posteriors from {N_CLUSTERS} clusters -- three methods, same priors", y=1.0)
plt.tight_layout(); plt.show()

print("Timings (this run):")
for k, (status, dt) in TIMINGS.items():
    print(f"  {k:<22} {status}" + (f"  {dt:6.1f}s" if dt else ""))"""))

C.append(md("""## 5. What you just saw (and how it maps to the paper)

- **The three methods agree** (within sampling noise) — that is the paper's central check: the
  joint hierarchical likelihood (🟢), the cached-posterior recycling shortcut (🟤), and the
  end-to-end neural posterior (🟣) are three routes to the *same* population posterior.
- **With only 5 clusters the posteriors are wide** — honestly so. The paper's analyses use
  $N_c = 376$ per bin (and thousands across bins), where $\\sigma_M, \\sigma_c$ tighten dramatically.
  Change `N_CLUSTERS` at the top and re-run (delete `cache/`) to watch that happen.
- **Why the hybrid matters:** the per-cluster NPE posteriors were computed *once* and cached
  (`data/percluster_posteriors.npz`). Re-fitting a *different* population model reuses them for
  free — no per-cluster inference is ever repeated. That is the workflow that scales.
- **Next tutorials:** importance-sampling details (undoing non-flat priors), out-of-distribution
  robustness, and the full richness–redshift-binned analysis."""))

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "tutorial_01_population_inference")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "tutorial_01_population_inference.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", path)
