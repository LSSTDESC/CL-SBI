"""
Utilities for defining and working with a sample of simulated clusters

Copyright 2022-2023, LSST-DESC
"""

from typing import Literal
import warnings
import numpy as np
from colossus.halo import concentration


def get_concentration(
    log10mass: float,
    mdef: str = "vir",
    z: float = 0.0,
    model: str = "child18",
) -> float:
    """
    Return the concentration for a halo of a given mass.

    Parameters
    ----------
    log10mass : float
        Log10 of the halo mass in solar masses.
    mdef : str, optional
        Mass definition. Default is "vir" (virial).
    z : float, optional
        Redshift. Default is 0.0.
    model : str, optional
        Mass-concentration relation model from colossus. Default is "child18".

    Returns
    -------
    float
        Halo concentration.
    """
    return concentration.concentration(10**log10mass, mdef, z, model=model)


def get_richness(
    log10mass: float,
    z: float = 0.0,
    model: Literal["murata17", "mcclintock18"] = "murata17",
    strict: bool = False,
) -> float:
    """
    Return the optical richness for a halo of a given mass.

    Parameters
    ----------
    log10mass : float
        Log10 of the halo mass in solar masses.
    z : float, optional
        Redshift. Default is 0.0.
    model : {"murata17", "mcclintock18"}, optional
        Richness-mass relation model. Default is "murata17".
    strict : bool, optional
        If True, warn when richness is outside the valid range for the model.
        Default is False.

    Returns
    -------
    float
        Optical richness (lambda).

    Notes
    -----
    The murata17 relation is only valid for lambda between 20 and 100.
    """
    if model == "murata17":
        a, b, mass_pivot = _get_murata2017_parameters()
        lambda_ = np.exp(a + b * np.log((10**log10mass) / mass_pivot))

        if strict and (lambda_ < 20 or lambda_ > 100):
            warnings.warn(
                f"calculated lambda value ({lambda_}) from the given log10mass ({log10mass}) is outside the range supported by the murata17 relation (20-100)"
            )

        return lambda_
    if model == "mcclintock18":
        m0, G, F = _get_mcclintock18_parameters()
        return ((10**log10mass / m0) * (((1 + z) / 1.35) ** -G)) ** (1 / F) * 40
    return 0


def get_log10mass_from_richness(
    lambda_: float,
    z: float = 0.0,
    model: Literal["murata17", "mcclintock18"] = "murata17",
    strict: bool = False,
) -> float:
    """
    Return the mass for a halo of a given optical richness.

    Parameters
    ----------
    lambda_ : float
        Optical richness.
    z : float, optional
        Redshift. Default is 0.0.
    model : {"murata17", "mcclintock18"}, optional
        Richness-mass relation model. Default is "murata17".
    strict : bool, optional
        If True, warn when richness is outside the valid range for the model.
        Default is False.

    Returns
    -------
    float
        Log10 of the halo mass in solar masses.

    Notes
    -----
    The murata17 relation is only valid for lambda between 20 and 100.
    """
    if model == "murata17":
        if strict and (lambda_ < 20 or lambda_ > 100):
            warnings.warn(
                f"provided lambda value ({lambda_}) is outside the range supported by the murata17 relation (20-100)"
            )

        a, b, mass_pivot = _get_murata2017_parameters()
        return np.log10(np.exp((np.log(lambda_) - a) / b) * mass_pivot)
    if model == "mcclintock18":
        m0, G, F = _get_mcclintock18_parameters()
        return np.log10(m0 * (lambda_ / 40) ** F * ((1 + z) / 1.35) ** G)
    return 0


def _get_murata2017_parameters() -> tuple[float, float, float]:
    """
    Return parameters for the Murata et al. 2017 richness-mass relation.

    Returns
    -------
    tuple[float, float, float]
        (a, b, mass_pivot) parameters.
    """
    a = 3.207
    b = 0.993
    mass_pivot = 3.0 * 10**14
    return a, b, mass_pivot


def _get_mcclintock18_parameters() -> tuple[float, float, float]:
    """
    Return parameters for the McClintock et al. 2018 richness-mass relation.

    Returns
    -------
    tuple[float, float, float]
        (m0, G, F) parameters.
    """
    m0 = 3.081e14
    G = -0.3
    F = 1.356
    return m0, G, F
