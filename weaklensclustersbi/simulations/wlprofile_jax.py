"""
Differentiable (JAX) NFW excess surface density, for HMC/NUTS hierarchical models.

Analytic Wright & Brainerd (2000) DeltaSigma(R) = mean-Sigma(<R) - Sigma(R), with halox
providing the differentiable (M, c) -> (Rs, rho0) conversion. Mirrors
wlprofile.simulate_nfw(kind="delta_sigma") (colossus) to <0.1% -- validated by
tests in the tutorial notebook; keep the two in sync via that assert.

Uses the "safe-x" trick from notebooks/hierarchical_mcmc_poc.py so the untaken
jnp.where branch never sees an invalid sqrt (which would poison gradients).
"""
import numpy as np
import jax
import jax.numpy as jnp
from halox.halo import NFWHalo
import halox.cosmology as hcos

Z = 0.275                  # mid of (min_z=0.2, max_z=0.35), matches the paper configs
DELTA_VIR = 124.89         # Bryan & Norman Delta_vir(z=0.275) wrt rho_crit (colossus 'vir')
NUM_RBINS = 30
RBINS_KPC = jnp.array(10 ** np.arange(0, NUM_RBINS / 10, 0.1))  # kpc/h
P18 = hcos.Planck18()


def nfw_logDeltaSigma(logM, c, rbins=RBINS_KPC, z=Z):
    """log10 DeltaSigma [Msun h / kpc^2] at rbins. Differentiable in (logM, c)."""
    halo = NFWHalo(m_delta=10.0 ** logM, c_delta=c, z=z, cosmo=P18, delta=DELTA_VIR)
    Rs = halo.Rs * 1000.0    # Mpc/h -> kpc/h
    rho0 = halo.rho0 / 1e9   # Msun h^2/Mpc^3 -> /kpc^3
    x = rbins / Rs
    x_lo = jnp.where(x < 1, x, 0.5)   # safe-x: untaken branch never sees bad sqrt
    x_hi = jnp.where(x > 1, x, 2.0)
    # Sigma(x) = 2 rho0 Rs f(x)
    f_lo = (1 - 2 * jnp.arctanh(jnp.sqrt((1 - x_lo) / (1 + x_lo))) / jnp.sqrt(1 - x_lo**2)) / (x_lo**2 - 1)
    f_hi = (1 - 2 * jnp.arctan(jnp.sqrt((x_hi - 1) / (1 + x_hi))) / jnp.sqrt(x_hi**2 - 1)) / (x_hi**2 - 1)
    f = jnp.where(x < 1, f_lo, jnp.where(x > 1, f_hi, 1.0 / 3.0))
    # mean-Sigma(<x) = 4 rho0 Rs g(x)/x^2
    g_lo = 2 * jnp.arctanh(jnp.sqrt((1 - x_lo) / (1 + x_lo))) / jnp.sqrt(1 - x_lo**2) + jnp.log(x_lo / 2)
    g_hi = 2 * jnp.arctan(jnp.sqrt((x_hi - 1) / (1 + x_hi))) / jnp.sqrt(x_hi**2 - 1) + jnp.log(x_hi / 2)
    g = jnp.where(x < 1, g_lo, jnp.where(x > 1, g_hi, 1.0 + jnp.log(0.5)))
    delta_sigma = 4 * rho0 * Rs * g / x**2 - 2 * rho0 * Rs * f
    return jnp.log10(delta_sigma)


# vectorized over a population of clusters: (N,), (N,) -> (N, NUM_RBINS)
nfw_logDeltaSigma_vmap = jax.vmap(nfw_logDeltaSigma, in_axes=(0, 0))


def delta_vir_of_z(z):
    """Bryan & Norman Delta_vir(z) wrt rho_crit (colossus 'vir' mdef).

    Not differentiable and not needed to be: the cell/cluster redshift is data, not a
    sampled parameter, so this is evaluated once per redshift and passed into the JAX
    forward as a constant. DELTA_VIR (=124.89) is the z=0.275 value.
    """
    from colossus.cosmology import cosmology
    from colossus.halo import mass_so

    cosmology.setCosmology("planck18")
    return float(mass_so.deltaVir(z))


def nfw_log_profile(logM, c, rbins=RBINS_KPC, z=Z, delta_vir=DELTA_VIR, kind="surface_density"):
    """log10 NFW profile at ``rbins``, differentiable in (logM, c).

    Unified z-aware forward for the hierarchical models: generalizes the single-z
    ``nfw_logDeltaSigma`` above to (i) an arbitrary redshift via ``delta_vir`` (pass
    ``delta_vir_of_z(z)`` for z != 0.275) and (ii) either observable.

    ``kind``:
      - ``"surface_density"``  -> log10 Sigma(R)                 (matches wlprofile Sigma)
      - ``"delta_sigma"``      -> log10 [meanSigma(<R) - Sigma(R)] (tangential-shear observable)

    ``kind`` is a static Python string (chosen at trace time), so the branch is safe
    under jit/vmap. Mirrors ``wlprofile.simulate_nfw(kind=...)`` (colossus) to <0.1%.
    """
    halo = NFWHalo(m_delta=10.0 ** logM, c_delta=c, z=z, cosmo=P18, delta=delta_vir)
    Rs = halo.Rs * 1000.0    # Mpc/h -> kpc/h
    rho0 = halo.rho0 / 1e9   # Msun h^2/Mpc^3 -> /kpc^3
    x = rbins / Rs
    x_lo = jnp.where(x < 1, x, 0.5)   # safe-x: untaken branch never sees a bad sqrt
    x_hi = jnp.where(x > 1, x, 2.0)

    # Sigma(x) = 2 rho0 Rs f(x)
    f_lo = (1 - 2 * jnp.arctanh(jnp.sqrt((1 - x_lo) / (1 + x_lo))) / jnp.sqrt(1 - x_lo**2)) / (x_lo**2 - 1)
    f_hi = (1 - 2 * jnp.arctan(jnp.sqrt((x_hi - 1) / (1 + x_hi))) / jnp.sqrt(x_hi**2 - 1)) / (x_hi**2 - 1)
    f = jnp.where(x < 1, f_lo, jnp.where(x > 1, f_hi, 1.0 / 3.0))
    sigma = 2 * rho0 * Rs * f
    if kind == "surface_density":
        return jnp.log10(sigma)
    if kind != "delta_sigma":
        raise ValueError(f"kind must be 'surface_density' or 'delta_sigma', got {kind!r}")

    # mean-Sigma(<x) = 4 rho0 Rs g(x)/x^2
    g_lo = 2 * jnp.arctanh(jnp.sqrt((1 - x_lo) / (1 + x_lo))) / jnp.sqrt(1 - x_lo**2) + jnp.log(x_lo / 2)
    g_hi = 2 * jnp.arctan(jnp.sqrt((x_hi - 1) / (1 + x_hi))) / jnp.sqrt(x_hi**2 - 1) + jnp.log(x_hi / 2)
    g = jnp.where(x < 1, g_lo, jnp.where(x > 1, g_hi, 1.0 + jnp.log(0.5)))
    mean_sigma = 4 * rho0 * Rs * g / x**2
    return jnp.log10(mean_sigma - sigma)


def make_forward(rbins=RBINS_KPC, z=Z, delta_vir=DELTA_VIR, kind="surface_density"):
    """Return a population-vmapped forward ``(logM[N], c[N]) -> log10 profile[N, nbins]``.

    Mirrors the ``make_fwd`` pattern in the POC scripts: bind the per-cell geometry
    (rbins, z, delta_vir, observable) once, vmap over the per-cluster (logM, c).
    """
    def _one(logM, c):
        return nfw_log_profile(logM, c, rbins=rbins, z=z, delta_vir=delta_vir, kind=kind)

    return jax.vmap(_one, in_axes=(0, 0))


def make_forward_percluster(rbins=RBINS_KPC, kind="surface_density"):
    """Return a forward vmapped over per-cluster (logM, c, z, delta_vir).

    For the fully-unbinned model, each cluster carries its OWN redshift.  z and delta_vir
    are data (not sampled), so delta_vir is precomputed via delta_vir_of_z and passed as an
    array; only (logM, c) are traced parameters.  Returns
    ``(logM[N], c[N], z[N], delta_vir[N]) -> log10 profile[N, nbins]``.
    """
    def _one(logM, c, z, delta_vir):
        return nfw_log_profile(logM, c, rbins=rbins, z=z, delta_vir=delta_vir, kind=kind)

    return jax.vmap(_one, in_axes=(0, 0, 0, 0))
