'''
Core structures and tools for dealing with a simulated weak lensing profile

Notes::
 
    [profile format] weaklensingclustersbi currently supports radial profiles
'''

from colossus.halo import profile_nfw
 
def simulate_nfw(log10mass, concentration, rbins, z=0.0, mdef = 'vir', kind='surface_density') :
  '''                                                                             
  Simulate an NFW profile with mass and concentration  defined with respect to a given mass definition.
  '''
  nfw = profile_nfw.NFWProfile(M = 10**log10mass, c = concentration,
                                z = z, mdef = mdef)

  if kind == 'surface_density': return nfw.surfaceDensity(rbins)

  elif kind == 'density' : return nfw.density(rbins)





model_profiles = {'nfw': simulate_nfw}
