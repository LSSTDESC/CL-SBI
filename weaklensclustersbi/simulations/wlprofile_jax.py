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
