"""
Core structures and tools for dealing with a simulated weak lensing profile

Notes::

    [profile format] weaklensingclustersbi currently supports radial profiles
"""

from typing import Literal
import numpy as np
from numpy.typing import NDArray
from colossus.halo import profile_nfw


def simulate_nfw(
    log10mass: float,
    concentration: float,
    rbins: NDArray[np.floating] = 10 ** np.arange(0, 3, 0.1),
    z: float = 0.0,
    mdef: str = "vir",
    kind: Literal["surface_density", "density"] = "surface_density",
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
    kind : {"surface_density", "density"}, optional
        Type of profile to return. Default is "surface_density".

    Returns
    -------
    NDArray[np.floating]
        The NFW profile evaluated at the specified radial bins.
    """
    nfw = profile_nfw.NFWProfile(M=10**log10mass, c=concentration, z=z, mdef=mdef)

    if kind == "surface_density":
        return nfw.surfaceDensity(rbins)

    elif kind == "density":
        return nfw.density(rbins)


model_profiles = {"nfw": simulate_nfw}
