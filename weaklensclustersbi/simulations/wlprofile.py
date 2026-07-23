"""
Core structures and tools for dealing with a simulated weak lensing profile

Notes::

    [profile format] weaklensingclustersbi currently supports radial profiles
"""

from typing import Literal
import numpy as np
from numpy.typing import NDArray
from colossus.halo import profile_nfw


def _nfw_delta_sigma(nfw, rbins):
    """
    Analytic NFW excess surface density ΔΣ(R) = Σ̄(<R) − Σ(R) following
    Wright & Brainerd (2000).

    Colossus's numerical ``deltaSigma`` rebuilds an integration table per
    profile (~85x slower than ``surfaceDensity``), which is prohibitive for the
    millions of forward-model evaluations in simulation generation and MCMC.
    This closed-form evaluation is as fast as ``surfaceDensity`` and agrees with
    Colossus's ``deltaSigma`` to ~1e-8 relative. The scale radius ``rs`` and
    characteristic density ``rhos`` are read from the Colossus profile so the
    normalization is identical to ``surfaceDensity`` (same mass definition).
    """
    rs = nfw.par["rs"]
    rhos = nfw.par["rhos"]
    x = np.asarray(rbins, dtype=float) / rs

    lo = x < 1.0
    hi = x > 1.0
    eq = ~(lo | hi)
    xl = x[lo]
    xh = x[hi]

    # Σ(x) = 2 rhos rs f(x)
    f = np.empty_like(x)
    f[lo] = (1.0 - 2.0 / np.sqrt(1.0 - xl**2) * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))) / (xl**2 - 1.0)
    f[hi] = (1.0 - 2.0 / np.sqrt(xh**2 - 1.0) * np.arctan(np.sqrt((xh - 1.0) / (1.0 + xh)))) / (xh**2 - 1.0)
    f[eq] = 1.0 / 3.0

    # Σ̄(<x) = 4 rhos rs g(x) / x^2
    g = np.empty_like(x)
    g[lo] = 2.0 / np.sqrt(1.0 - xl**2) * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl))) + np.log(xl / 2.0)
    g[hi] = 2.0 / np.sqrt(xh**2 - 1.0) * np.arctan(np.sqrt((xh - 1.0) / (1.0 + xh))) + np.log(xh / 2.0)
    g[eq] = 1.0 + np.log(0.5)

    sigma = 2.0 * rhos * rs * f
    mean_sigma = 4.0 * rhos * rs * g / x**2
    return mean_sigma - sigma


def simulate_nfw(
    log10mass: float,
    concentration: float,
    rbins: NDArray[np.floating] = 10 ** np.arange(0, 3, 0.1),
    z: float = 0.0,
    mdef: str = "vir",
    kind: Literal["surface_density", "delta_sigma", "density"] = "surface_density",
) -> NDArray[np.floating]:
    """
    Simulate an NFW profile with mass and concentration defined with respect to a given mass definition.

    Parameters
    ----------
    log10mass : float
        Log10 of the halo mass in solar masses.
    concentration : float
        Halo concentration parameter.
    rbins : NDArray[np.floating], optional
        Radial bins in kpc/h. Default is logarithmically spaced from 1 to 1000 kpc/h.
    z : float, optional
        Redshift. Default is 0.0.
    mdef : str, optional
        Mass definition for the halo. Default is "vir" (virial).
    kind : {"surface_density", "delta_sigma", "density"}, optional
        Type of profile to return. Default is "surface_density".
        "delta_sigma" returns the excess surface density
        ΔΣ(R) = Σ̄(<R) − Σ(R), the quantity probed by tangential shear in
        stacked weak-lensing analyses.

    Returns
    -------
    NDArray[np.floating]
        The NFW profile evaluated at the specified radial bins.
    """
    nfw = profile_nfw.NFWProfile(M=10**log10mass, c=concentration, z=z, mdef=mdef)

    if kind == "surface_density":
        return nfw.surfaceDensity(rbins)

    elif kind == "delta_sigma":
        return _nfw_delta_sigma(nfw, rbins)

    elif kind == "density":
        return nfw.density(rbins)


model_profiles = {"nfw": simulate_nfw}
