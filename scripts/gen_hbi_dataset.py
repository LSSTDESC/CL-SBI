#!/usr/bin/env python3
"""
Generate a fully-unbinned dataset for hierarchical (M, c) population inference (Paper II).

Unlike the binned pipeline (gen_simulations/gen_observations, which loop over discrete
richness bins), this draws ONE population of clusters each carrying a CONTINUOUS
(richness lambda_i, redshift z_i):

  1. (logM_i, z_i) ~ mass function (Tinker08) x richness-selection above a floor, over the
     survey redshift range  -- the detected-cluster (M, z) distribution.
  2. lambda_i         ~ richness-mass relation (McClintock18) + log-normal scatter, truncated
     at the floor (a per-cluster covariate for the inference mass prior).
  3. c_i              = c0 + beta*(logM_i - logM_ref) + gamma*(z_i - z_ref) + N(0, sig_c)
     -- an EXPLICIT parametric truth so the recovered (c0, beta, gamma, sig_c) is testable.
  4. profile_i        = log10 forward(logM_i, c_i, z_i, observable) + N(0, sigma)   (log-space)

Outputs (gitignored) -> outputs/hbi/<dataset_id>/:
    profiles.npy   (N, nbins) log10 observed profiles
    sigmas.npy     (nbins,)   per-radial-bin log-space error
    richness.npy   (N,)       lambda_i
    redshift.npy   (N,)       z_i
    true_mc.npy    (N, 2)     [logM_i, c_i]
    rbins.npy      (nbins,)   kpc/h
    truth.json     truth relation + selection + observable metadata

Selection / relation model (rm_relation, mf model, scatter, refs, observable) is read from a
configs/population/*.json; the truth relation and survey selection come from CLI flags.
Reuses the validated machinery (populationutils, colossus MF) -- the continuous analogue of
population_hbi.massfn_selection_logpdf.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from weaklensclustersbi.simulations import populationutils, wlprofile
from weaklensclustersbi.inference import hbi


def sample_population(
    n,
    *,
    z_min,
    z_max,
    lambda_floor,
    lambda_max,
    rm_relation,
    rm_scatter,
    mf_model,
    mf_mdef,
    mass_grid,
    n_z_grid,
    rng,
):
    """Draw (logM_i, z_i, lambda_i) for ``n`` detected clusters.

    Samples the joint (logM, z) distribution of detected clusters,
    p(logM, z) proportional to  (dn/dlnM)(logM, z) * P(lambda >= floor | logM, z),
    on a (z x logM) grid, then assigns each a scattered richness truncated at the floor.
    """
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function
    from scipy.stats import norm

    cosmology.setCosmology("planck18")
    lo, hi, ng = mass_grid
    logM_grid = np.linspace(lo, hi, ng)
    z_grid = np.linspace(z_min, z_max, n_z_grid)

    F = float(populationutils.get_rm_slope(rm_relation))
    sig_lnlam = rm_scatter * np.log(10.0) / F

    # joint weight over (z, logM)
    W = np.empty((n_z_grid, ng))
    for j, z in enumerate(z_grid):
        dndlnM = mass_function.massFunction(
            10 ** logM_grid, z, mdef=mf_mdef, model=mf_model, q_out="dndlnM"
        )
        ll = np.log(populationutils.get_richness(logM_grid, z=z, model=rm_relation))
        p_sel = 1.0 - norm.cdf(np.log(lambda_floor), ll, sig_lnlam)  # P(lambda >= floor)
        W[j] = dndlnM * np.log(10.0) * np.clip(p_sel, 0.0, None)
    W = np.clip(W, 0.0, None)
    W /= W.sum()

    idx = rng.choice(W.size, size=n, replace=True, p=W.ravel())
    zj, mi = np.unravel_index(idx, W.shape)
    # jitter within grid cells so masses/redshifts are continuous, not gridded
    dlogM = (hi - lo) / (ng - 1)
    dz = (z_max - z_min) / (n_z_grid - 1) if n_z_grid > 1 else 0.0
    logM = np.clip(logM_grid[mi] + rng.uniform(-0.5, 0.5, n) * dlogM, lo, hi)
    z = np.clip(z_grid[zj] + rng.uniform(-0.5, 0.5, n) * dz, z_min, z_max)

    # assign richness from the R-M relation + scatter, truncated at the floor
    ln_lam_mean = np.log(populationutils.get_richness(logM, z=z, model=rm_relation))
    lam = np.exp(ln_lam_mean + rng.normal(0.0, sig_lnlam, n))
    below = lam < lambda_floor
    n_iter = 0
    while below.any() and n_iter < 100:  # resample scatter for clusters that fell below
        lam[below] = np.exp(ln_lam_mean[below] + rng.normal(0.0, sig_lnlam, below.sum()))
        below = lam < lambda_floor
        n_iter += 1
    lam = np.clip(lam, lambda_floor, lambda_max)
    return logM, z, lam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_id", required=True, help="output dir name under outputs/hbi/")
    ap.add_argument("--population_config", default="configs/population/fiducial.json")
    ap.add_argument("--n_clusters", type=int, default=2000)
    # survey selection
    ap.add_argument("--z_min", type=float, default=0.2)
    ap.add_argument("--z_max", type=float, default=0.65)
    ap.add_argument("--lambda_floor", type=float, default=20.0)
    ap.add_argument("--lambda_max", type=float, default=200.0)
    ap.add_argument("--n_z_grid", type=int, default=24)
    # truth M-c relation (what the fit should recover)
    ap.add_argument("--c0", type=float, default=4.55)
    ap.add_argument("--beta", type=float, default=-0.68)
    ap.add_argument("--gamma", type=float, default=-2.07)
    ap.add_argument("--sig_c", type=float, default=0.15)
    # noise template + rng
    ap.add_argument("--sigmas_from", default="outputs/observations/obs_z1_lambda5.376",
                    help="dir with a sigmas.npy noise template (per-radial-bin log-space error)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()

    script_dir = os.path.dirname(__file__)
    cfg = hbi.read_config(os.path.join(script_dir, "..", args.population_config))
    rel = cfg.get("relation", {})
    logM_ref = rel.get("logM_ref", hbi.LOGM_REF)
    z_ref = rel.get("z_ref", hbi.Z_REF)
    kind = cfg.get("observable", "surface_density")
    ms = cfg.get("massfn_selection", {})
    rm_relation = ms.get("rm_relation", "mcclintock18")
    rm_scatter = ms.get("rm_scatter", 0.1)
    mf_model = ms.get("mf_model", "tinker08")
    mf_mdef = ms.get("mf_mdef", "200m")
    mass_grid = tuple(ms.get("mass_grid", [13.3, 15.4, 1200]))

    out_dir = os.path.join(script_dir, "..", "outputs", "hbi", args.dataset_id)
    if os.path.exists(os.path.join(out_dir, "profiles.npy")) and not args.regenerate:
        print(f"dataset exists at {out_dir}; use --regenerate to overwrite")
        return
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # noise template + radial bins (match the paper's 30 log-spaced kpc/h bins)
    sig_path = os.path.join(script_dir, "..", args.sigmas_from, "sigmas.npy")
    sigmas = np.load(sig_path)
    nbins = len(sigmas)
    rbins = 10 ** np.arange(0, nbins / 10, 0.1)

    print(f"drawing {args.n_clusters} clusters: z in [{args.z_min},{args.z_max}], "
          f"lambda>={args.lambda_floor}, {rm_relation} x {mf_model}, observable={kind}", flush=True)
    logM, z, lam = sample_population(
        args.n_clusters, z_min=args.z_min, z_max=args.z_max,
        lambda_floor=args.lambda_floor, lambda_max=args.lambda_max,
        rm_relation=rm_relation, rm_scatter=rm_scatter,
        mf_model=mf_model, mf_mdef=mf_mdef, mass_grid=mass_grid, n_z_grid=args.n_z_grid, rng=rng,
    )

    # concentration from the explicit truth relation
    c = args.c0 + args.beta * (logM - logM_ref) + args.gamma * (z - z_ref) + rng.normal(0.0, args.sig_c, args.n_clusters)

    # forward model each cluster at its own redshift, then add log-space noise
    noiseless = np.array([
        np.log10(wlprofile.simulate_nfw(m, ci, rbins, zi, kind=kind))
        for m, ci, zi in zip(logM, c, z)
    ])
    profiles = noiseless + rng.normal(0.0, sigmas[None, :], size=noiseless.shape)

    true_mc = np.column_stack([logM, c])
    np.save(os.path.join(out_dir, "profiles.npy"), profiles)
    np.save(os.path.join(out_dir, "sigmas.npy"), sigmas)
    np.save(os.path.join(out_dir, "richness.npy"), lam)
    np.save(os.path.join(out_dir, "redshift.npy"), z)
    np.save(os.path.join(out_dir, "true_mc.npy"), true_mc)
    np.save(os.path.join(out_dir, "rbins.npy"), rbins)

    truth = dict(
        dataset_id=args.dataset_id, n_clusters=args.n_clusters, observable=kind,
        relation=dict(c0=args.c0, beta=args.beta, gamma=args.gamma, sig_c=args.sig_c,
                      logM_ref=logM_ref, z_ref=z_ref),
        selection=dict(z_min=args.z_min, z_max=args.z_max, lambda_floor=args.lambda_floor,
                       lambda_max=args.lambda_max, rm_relation=rm_relation, rm_scatter=rm_scatter,
                       mf_model=mf_model, mf_mdef=mf_mdef),
        population_config=args.population_config, sigmas_from=args.sigmas_from, seed=args.seed,
        logM_range=[float(logM.min()), float(logM.max())],
        lambda_range=[float(lam.min()), float(lam.max())],
    )
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)

    # self-consistency: lstsq of the generated (M,c) should recover the input truth relation
    A = np.column_stack([np.ones_like(logM), logM - logM_ref, z - z_ref])
    fit = np.linalg.lstsq(A, c, rcond=None)[0]
    print(f"wrote {args.n_clusters} clusters -> {out_dir}")
    print(f"  logM in [{logM.min():.2f},{logM.max():.2f}]  ~{logM.max()-logM.min():.2f} dex lever arm; "
          f"lambda in [{lam.min():.0f},{lam.max():.0f}]")
    print(f"  input truth:     c0={args.c0:.3f} beta={args.beta:.3f} gamma={args.gamma:.3f}")
    print(f"  lstsq of sample: c0={fit[0]:.3f} beta={fit[1]:.3f} gamma={fit[2]:.3f}  (self-consistency)")


if __name__ == "__main__":
    main()
