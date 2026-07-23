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
