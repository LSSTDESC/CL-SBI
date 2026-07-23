"""
Hierarchical Bayesian population model for cluster (M, c) inference (NumPyro).

Lifts and parameterizes the two proof-of-concept models into one config-driven package
model:
  - notebooks/hierarchical_mcmc_poc.py   -- single cell, free (mu_M, sig_M), no z-evolution
  - notebooks/hier_fulldataset_hmc.py    -- 12 cells, mass-function x selection prior, + gamma

Both share ONE M-c relation with (optional) redshift evolution across all cells:

    c = c0 + beta * (logM - logM_ref) + gamma * (z - z_ref) + Normal(0, sig_c)

and each cell contributes its own per-cluster mass population via one of two modes:

  - "gaussian": logM ~ Normal(mu_M, sig_M) with FREE per-cell (mu_M, sig_M).  Here the
    relation is centered on the cell's own mu_M (matches the single-cell POC).
  - "massfn":   logM has an informative mass-function x richness-selection prior supplied
    as a (grid, logpdf) table, with a free per-cell mean shift dmu and fixed width.  Here
    the relation is centered on the shared logM_ref (matches the 12-cell POC).

Non-centered parameterization throughout (avoids Neal's funnel).  The differentiable
forward model is weaklensclustersbi.simulations.wlprofile_jax.make_forward (Sigma or
DeltaSigma, z- and Delta_vir-aware), bound per cell and stored as ``cell["fwd"]``.

This module is pure model code: it builds no data and does no IO.  Cells (including the
MF x selection tables and the bound forward) are assembled by
weaklensclustersbi.simulations.population_hbi and the run_hbi_* pipeline scripts.

Fully-unbinned inference (per-cluster continuous lambda_i, z_i) is a Stage-1 extension:
it degenerates to a single "cell" with per-cluster z in the forward and a per-cluster
mass prior; the cell-grouped model here covers the single-cell and 12-cell cases (Stage 0).
"""
from __future__ import annotations

import json

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

# Relation reference points (match the 12-cell POC: hier_fulldataset_hmc.py).
LOGM_REF = 14.3
Z_REF = 0.35

# Default hyperparameter priors -- match the 12-cell POC exactly.  Override via the
# ``priors`` argument (e.g. the single-cell POC used beta ~ Normal(0, 2.0),
# sig_extra ~ HalfNormal(0.1), and mass_mode="gaussian" with evolve_z=False).
DEFAULT_PRIORS = {
    "c0": ("Normal", 4.6, 1.0),
    "beta": ("Normal", 0.0, 1.5),
    "gamma": ("Normal", 0.0, 3.0),
    "sig_c": ("HalfNormal", 0.5),
    "sig_extra": ("HalfNormal", 0.15),
    # gaussian mass mode
    "mu_M": ("Normal", 14.4, 0.3),
    "sig_M": ("HalfNormal", 0.3),
    # massfn mass mode
    "dmu": ("Normal", 0.0, 0.15),
}

_DISTS = {
    "Normal": dist.Normal,
    "HalfNormal": dist.HalfNormal,
    "Uniform": dist.Uniform,
    "LogNormal": dist.LogNormal,
}


def _make_dist(spec):
    """Build a numpyro distribution from a JSON-friendly ``(name, *params)`` tuple."""
    name, *params = spec
    if name not in _DISTS:
        raise ValueError(f"unknown distribution {name!r}; known: {sorted(_DISTS)}")
    return _DISTS[name](*params)


def _resolve_priors(priors):
    if priors is None:
        return dict(DEFAULT_PRIORS)
    merged = dict(DEFAULT_PRIORS)
    merged.update(priors)
    return merged


def hier_model(
    cells,
    priors=None,
    *,
    mass_mode="massfn",
    evolve_z=True,
    logM_ref=LOGM_REF,
    z_ref=Z_REF,
    observed=True,
):
    """One shared M-c relation across ``cells``; per-cell mass populations.

    Parameters
    ----------
    cells : list of dict
        Each cell provides:
          ``n``     : int, number of clusters
          ``z``     : float, cell redshift (sets the relation's z-term)
          ``sig``   : (nbins,) per-radial-bin log-space observational error
          ``prof``  : (n, nbins) observed log10 profiles (used iff ``observed``)
          ``fwd``   : callable (logM[n], c[n]) -> log10 model profiles[n, nbins]
                      (bind with wlprofile_jax.make_forward(rbins, z, delta_vir, kind))
        For ``mass_mode="massfn"`` also:
          ``grid``  : (G,) log10-mass grid, ``lp`` : (G,) log P(logM) (MF x selection),
          ``mode``  : float, grid argmax (mean-shift anchor); ``width`` : float (default 0.15)
    priors : dict, optional
        Overrides for ``DEFAULT_PRIORS`` (JSON-friendly ``(dist_name, *params)`` tuples).
    mass_mode : {"massfn", "gaussian"}
    evolve_z : bool
        Include the ``gamma * (z - z_ref)`` redshift-evolution term.
    observed : bool
        If False, run the model in prior-predictive mode (obs=None) -- used for tests.
    """
    p = _resolve_priors(priors)

    c0 = numpyro.sample("c0", _make_dist(p["c0"]))
    beta = numpyro.sample("beta", _make_dist(p["beta"]))
    gamma = numpyro.sample("gamma", _make_dist(p["gamma"])) if evolve_z else 0.0
    sig_c = numpyro.sample("sig_c", _make_dist(p["sig_c"]))
    sig_extra = numpyro.sample("sig_extra", _make_dist(p["sig_extra"]))

    for k, cell in enumerate(cells):
        n = int(cell["n"])
        z = float(cell["z"])
        sig = cell["sig"]

        if mass_mode == "gaussian":
            mu_M = numpyro.sample(f"mu_M_{k}", _make_dist(p["mu_M"]))
            sig_M = numpyro.sample(f"sig_M_{k}", _make_dist(p["sig_M"]))
            with numpyro.plate(f"cl_{k}", n):
                zM = numpyro.sample(f"zM_{k}", dist.Normal(0.0, 1.0))
                logM = mu_M + sig_M * zM
                zc = numpyro.sample(f"zc_{k}", dist.Normal(0.0, 1.0))
                # relation centered on the cell's own mean (single-cell POC convention)
                c = c0 + beta * (logM - mu_M) + gamma * (z - z_ref) + sig_c * zc

        elif mass_mode == "massfn":
            width = float(cell.get("width", 0.15))
            dmu = numpyro.sample(f"dmu_{k}", _make_dist(p["dmu"]))
            with numpyro.plate(f"cl_{k}", n):
                zM = numpyro.sample(f"zM_{k}", dist.Normal(0.0, 1.0))
                logM = cell["mode"] + dmu + width * zM
                # informative mass-function x richness-selection prior
                numpyro.factor(f"mf_{k}", jnp.interp(logM, cell["grid"], cell["lp"]))
                zc = numpyro.sample(f"zc_{k}", dist.Normal(0.0, 1.0))
                # relation centered on the shared reference (12-cell POC convention)
                c = c0 + beta * (logM - logM_ref) + gamma * (z - z_ref) + sig_c * zc

        else:
            raise ValueError(f"mass_mode must be 'massfn' or 'gaussian', got {mass_mode!r}")

        model_prof = cell["fwd"](logM, c)
        sig_tot = jnp.sqrt(sig[None, :] ** 2 + sig_extra ** 2)
        numpyro.sample(
            f"obs_{k}",
            dist.Normal(model_prof, sig_tot),
            obs=(cell["prof"] if observed else None),
        )


# Hyperparameters reported/compared across methods (per-cluster latents excluded).
POP_PARAMS = ("c0", "beta", "gamma", "sig_c", "sig_extra")


def read_config(path):
    """Load a configs/population/*.json population-model config."""
    with open(path) as f:
        return json.load(f)


def model_kwargs_from_config(cfg):
    """Map a population config to hier_model/run_nuts model kwargs."""
    rel = cfg.get("relation", {})
    return dict(
        priors=cfg.get("priors"),
        mass_mode=cfg.get("mass_mode", "massfn"),
        evolve_z=rel.get("evolve_z", True),
        logM_ref=rel.get("logM_ref", LOGM_REF),
        z_ref=rel.get("z_ref", Z_REF),
    )


def nuts_kwargs_from_config(cfg):
    """NUTS sampler kwargs from the config's ``nuts`` block (empty -> run_nuts defaults)."""
    return dict(cfg.get("nuts", {}))


def run_nuts(
    cells,
    priors=None,
    *,
    mass_mode="massfn",
    evolve_z=True,
    logM_ref=LOGM_REF,
    z_ref=Z_REF,
    num_warmup=600,
    num_samples=800,
    num_chains=2,
    target_accept_prob=0.9,
    seed=0,
):
    """Sample the hierarchical model with NUTS; return (samples, grouped_samples).

    ``grouped_samples`` is by-chain (for r-hat via numpyro.diagnostics.summary).
    """
    import jax
    from numpyro.infer import MCMC, NUTS

    def _model():
        hier_model(
            cells,
            priors=priors,
            mass_mode=mass_mode,
            evolve_z=evolve_z,
            logM_ref=logM_ref,
            z_ref=z_ref,
            observed=True,
        )

    kernel = NUTS(
        _model,
        target_accept_prob=target_accept_prob,
        init_strategy=numpyro.infer.init_to_median,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples(), mcmc.get_samples(group_by_chain=True)
