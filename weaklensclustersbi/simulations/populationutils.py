"""
Utilities for defining and working with a sample of simulated clusters

Copyright 2022-2023, LSST-DESC
"""
from colossus.halo import concentration

def get_concentration(log10mass, mdef='vir', z=0.0, model='child18') :
  '''
  Return the concentration for a halo of a given mass assuming a model of child18
  at the present day (redshift 0.0), defined with respect to the virial radius. 
  '''

  return concentration.concentration(10**log10mass, mdef, z, model=model)
