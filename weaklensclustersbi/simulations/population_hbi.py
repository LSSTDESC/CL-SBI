"""
Generative population machinery for the hierarchical (M, c) models.

Provides the mass-function x richness-selection mass prior and the cell assembly used by
weaklensclustersbi.inference.hbi.  Lifted and parameterized from the POC scripts
(notebooks/hier_fulldataset_hmc.py ``cell_mf_logpdf`` + cell loading), removing the
hardcoded model/relation names so the choices live in configs.

A "cell" is a (redshift, richness-window) group of clusters.  Each cell carries its
observed profiles, per-bin errors, redshift, the differentiable forward bound to that
redshift/geometry, and -- for the ``massfn`` mass mode -- an informative log P(logM)
table = mass function (Tinker08 200m) x richness-selection window (McClintock18), the
same "validated machinery" as the paper.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import norm as _norm

from . import populationutils
from . import wlprofile_jax as J

# numpy 2.0 renamed trapz -> trapezoid (trapz still present but deprecated)
_trapz = getattr(np, "trapezoid", np.trapz)

DEFAULT_MF_MODEL = "tinker08"
DEFAULT_MF_MDEF = "200m"
DEFAULT_RM_RELATION = "mcclintock18"
DEFAULT_RM_SCATTER = 0.1
DEFAULT_MASS_GRID = (13.3, 15.4, 1200)  # (log10 M lo, hi, n)


def massfn_selection_logpdf(
    lmin,
    lmax,
    z,
    *,
    mf_model=DEFAULT_MF_MODEL,
    mf_mdef=DEFAULT_MF_MDEF,
    rm_relation=DEFAULT_RM_RELATION,
    rm_scatter=DEFAULT_RM_SCATTER,
    mass_grid=DEFAULT_MASS_GRID,
):
    """Normalized log P(log10 M) for a richness bin [lmin, lmax] at redshift ``z``.

    P(logM) proportional to  (dn/dlnM)(logM, z)  x  P(lmin < lambda < lmax | logM, z),
    with the richness-selection window from a log-normal scatter about the richness-mass
    relation.  Returns (grid, logpdf) with logpdf normalized by trapezoid over the grid.
    Mirrors ``cell_mf_logpdf`` in notebooks/hier_fulldataset_hmc.py.
    """
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function

    cosmology.setCosmology("planck18")
    lo, hi, ng = mass_grid
    grid = np.linspace(lo, hi, ng)

    dndlnM = mass_function.massFunction(10 ** grid, z, mdef=mf_mdef, model=mf_model, q_out="dndlnM")

    # richness-selection window: log-normal scatter in ln(lambda) about the R-M relation
    F = float(populationutils.get_rm_slope(rm_relation))
    sig_lnlam = rm_scatter * np.log(10.0) / F
    lam = populationutils.get_richness(grid, z=z, model=rm_relation)
    ll = np.log(lam)
    p_in = _norm.cdf(np.log(lmax), ll, sig_lnlam) - _norm.cdf(np.log(lmin), ll, sig_lnlam)

    pdf = np.clip(dndlnM * np.log(10.0) * np.clip(p_in, 1e-12, None), 1e-300, None)
    lp = np.log(pdf)
    lp -= np.log(_trapz(np.exp(lp), grid))
    return grid, lp


def load_cell(
    obs_dir,
    z,
    *,
    mass_mode="massfn",
    lmin=None,
    lmax=None,
    kind="surface_density",
    width=0.15,
    delta_vir=None,
    **mf_kwargs,
):
    """Assemble one cell dict (for hbi.hier_model) from an observation output dir.

    Reads ``drawn_nfw_profiles.npy`` (log10 profiles) and ``sigmas.npy`` from ``obs_dir``,
    binds the z/Δ_vir/observable-aware forward, and -- for ``mass_mode="massfn"`` -- attaches
    the MF x selection prior table (requires ``lmin``/``lmax``).
    """
    import jax.numpy as jnp

    prof = np.load(os.path.join(obs_dir, "drawn_nfw_profiles.npy"))
    sig = np.load(os.path.join(obs_dir, "sigmas.npy"))
    if delta_vir is None:
        delta_vir = J.delta_vir_of_z(z)

    cell = dict(
        n=int(prof.shape[0]),
        z=float(z),
        sig=jnp.asarray(sig),
        prof=jnp.asarray(prof),
        fwd=J.make_forward(z=z, delta_vir=delta_vir, kind=kind),
    )

    if mass_mode == "massfn":
        if lmin is None or lmax is None:
            raise ValueError("mass_mode='massfn' requires lmin and lmax")
        grid, lp = massfn_selection_logpdf(lmin, lmax, z, **mf_kwargs)
        cell.update(
            grid=jnp.asarray(grid),
            lp=jnp.asarray(lp),
            mode=float(grid[int(np.argmax(lp))]),
            width=float(width),
        )
    elif mass_mode != "gaussian":
        raise ValueError(f"mass_mode must be 'massfn' or 'gaussian', got {mass_mode!r}")

    # keep the truth for acceptance/diagnostics if present (not used by the model)
    tmc_path = os.path.join(obs_dir, "drawn_mc_pairs.npy")
    if os.path.exists(tmc_path):
        cell["true_mc"] = np.load(tmc_path)

    return cell


def load_cells(specs, *, mass_mode="massfn", kind="surface_density", width=0.15, **mf_kwargs):
    """Load a list of cells.

    Each spec is a dict with at least ``obs_dir`` and ``z``; for ``massfn`` also ``lmin``,
    ``lmax`` (per-spec keys override the shared kwargs).
    """
    cells = []
    for s in specs:
        s = dict(s)
        obs_dir = s.pop("obs_dir")
        z = s.pop("z")
        cells.append(
            load_cell(
                obs_dir,
                z,
                mass_mode=s.pop("mass_mode", mass_mode),
                kind=s.pop("kind", kind),
                width=s.pop("width", width),
                **{**mf_kwargs, **s},
            )
        )
    return cells


def build_unbinned_massprior(
    richness,
    z,
    *,
    rm_relation=DEFAULT_RM_RELATION,
    rm_scatter=DEFAULT_RM_SCATTER,
    mf_model=DEFAULT_MF_MODEL,
    mf_mdef=DEFAULT_MF_MDEF,
    mass_grid=DEFAULT_MASS_GRID,
    n_z_grid=40,
    width_floor=0.03,
):
    """Per-cluster mass prior tables for the fully-unbinned model.

    For each cluster i with (lambda_i, z_i):
        log P(logM | lambda_i, z_i) = log[(dn/dlnM)(logM, z_i)]
                                      - 0.5 * ((ln lambda_i - ln lambda_mean(logM, z_i)) / sig_lnlam)^2
    i.e. the mass function shape times the richness likelihood (the continuous, per-cluster
    analogue of the cell MF x selection window).  The z-dependence of the mass function and
    the R-M relation is tabulated on a z-grid and linearly interpolated to each z_i.

    Returns (grid[G], lp_table[N, G], anchor[N], width[N]).  ``anchor``/``width`` are the
    per-cluster prior mean and std (width floored), used as the non-centered logM scale.
    """
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function
    from scipy.interpolate import interp1d

    cosmology.setCosmology("planck18")
    richness = np.asarray(richness, dtype=float)
    z = np.asarray(z, dtype=float)
    lo, hi, ng = mass_grid
    grid = np.linspace(lo, hi, ng)

    F = float(populationutils.get_rm_slope(rm_relation))
    sig_lnlam = rm_scatter * np.log(10.0) / F

    zg = np.linspace(z.min(), z.max(), n_z_grid) if z.max() > z.min() else np.array([z[0]])
    log_mf_g = np.empty((len(zg), ng))
    ln_lam_g = np.empty((len(zg), ng))
    for j, zz in enumerate(zg):
        dndlnM = mass_function.massFunction(10 ** grid, zz, mdef=mf_mdef, model=mf_model, q_out="dndlnM")
        log_mf_g[j] = np.log(np.clip(dndlnM * np.log(10.0), 1e-300, None))
        ln_lam_g[j] = np.log(populationutils.get_richness(grid, z=zz, model=rm_relation))

    if len(zg) > 1:
        log_mf_i = interp1d(zg, log_mf_g, axis=0)(z)   # (N, G)
        ln_lam_i = interp1d(zg, ln_lam_g, axis=0)(z)   # (N, G)
    else:
        log_mf_i = np.repeat(log_mf_g, len(z), axis=0)
        ln_lam_i = np.repeat(ln_lam_g, len(z), axis=0)

    lp = log_mf_i - 0.5 * ((np.log(richness)[:, None] - ln_lam_i) / sig_lnlam) ** 2
    lp -= lp.max(axis=1, keepdims=True)  # per-cluster constant; numerical stability only

    w = np.exp(lp)
    w /= w.sum(axis=1, keepdims=True)
    anchor = (w * grid).sum(axis=1)
    var = (w * (grid - anchor[:, None]) ** 2).sum(axis=1)
    width = np.maximum(np.sqrt(np.clip(var, 0.0, None)), width_floor)
    return grid, lp, anchor, width


def load_unbinned_dataset(dataset_dir, cfg):
    """Load a gen_hbi_dataset.py output into an hbi.hier_model_unbinned ``data`` dict."""
    import json

    import jax.numpy as jnp

    prof = np.load(os.path.join(dataset_dir, "profiles.npy"))
    sig = np.load(os.path.join(dataset_dir, "sigmas.npy"))
    lam = np.load(os.path.join(dataset_dir, "richness.npy"))
    z = np.load(os.path.join(dataset_dir, "redshift.npy"))
    rbins = np.load(os.path.join(dataset_dir, "rbins.npy"))

    skw = selection_kwargs_from_config(cfg)
    kind = skw.pop("kind", "surface_density")
    skw.pop("mass_mode", None)
    skw.pop("width", None)  # cell-model fixed width; unbinned prior derives per-cluster width
    grid, lp, anchor, width = build_unbinned_massprior(lam, z, **skw)

    # delta_vir(z_i) via z-grid interpolation (cheap, avoids N colossus calls)
    zg = np.linspace(z.min(), z.max(), 40) if z.max() > z.min() else np.array([z[0]])
    dvir_g = np.array([J.delta_vir_of_z(zz) for zz in zg])
    delta_vir = np.interp(z, zg, dvir_g)

    data = dict(
        n=len(z),
        prof=jnp.asarray(prof),
        sig=jnp.asarray(sig),
        z=jnp.asarray(z),
        richness=jnp.asarray(lam),
        delta_vir=jnp.asarray(delta_vir),
        grid=jnp.asarray(grid),
        lp_table=jnp.asarray(lp),
        anchor=jnp.asarray(anchor),
        width=jnp.asarray(width),
        fwd=J.make_forward_percluster(rbins=jnp.asarray(rbins), kind=kind),
    )
    tmc = os.path.join(dataset_dir, "true_mc.npy")
    if os.path.exists(tmc):
        data["true_mc"] = np.load(tmc)
    tj = os.path.join(dataset_dir, "truth.json")
    if os.path.exists(tj):
        with open(tj) as f:
            data["truth"] = json.load(f)
    return data


def selection_kwargs_from_config(cfg):
    """Map a population config to load_cells kwargs (mass mode, observable, MF x selection)."""
    out = {"mass_mode": cfg.get("mass_mode", "massfn"), "kind": cfg.get("observable", "surface_density")}
    ms = cfg.get("massfn_selection", {})
    for k in ("mf_model", "mf_mdef", "rm_relation", "rm_scatter", "width"):
        if k in ms:
            out[k] = ms[k]
    if "mass_grid" in ms:
        out["mass_grid"] = tuple(ms["mass_grid"])
    return out


def fit_true_relation(cells, *, logM_ref, z_ref, evolve_z=True):
    """Least-squares (c0, beta, gamma) of the pooled true (M,c) across cells.

    The ground-truth relation the hierarchical fit should recover; requires each cell to
    carry ``true_mc``.  With ``evolve_z=False`` the gamma column is dropped (gamma=0).
    """
    logM, conc, zs = [], [], []
    for cell in cells:
        if "true_mc" not in cell:
            raise ValueError("cell missing 'true_mc'; cannot compute true relation")
        tmc = cell["true_mc"]
        logM.append(tmc[:, 0])
        conc.append(tmc[:, 1])
        zs.append(np.full(len(tmc), cell["z"]))
    logM = np.concatenate(logM)
    conc = np.concatenate(conc)
    zs = np.concatenate(zs)

    cols = [np.ones_like(logM), logM - logM_ref]
    if evolve_z:
        cols.append(zs - z_ref)
    A = np.column_stack(cols)
    coef = np.linalg.lstsq(A, conc, rcond=None)[0]
    out = {"c0": float(coef[0]), "beta": float(coef[1]), "sig_c": float(conc.std())}
    out["gamma"] = float(coef[2]) if evolve_z else 0.0
    return out
