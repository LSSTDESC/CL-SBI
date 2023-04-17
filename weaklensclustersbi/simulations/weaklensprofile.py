'''
Core structures and tools for dealing with a simulated weak lensing profile

Notes::
 
    [profile format] weaklensingclustersbi currently supports radial profiles
'''


def simulate_nfw(log10mass, concentration, z=0.0, mdef = 'vir') :
  '''                                                                             
  Simulate an NFW profile at the present day (z=0), with mass and concentration  defined with respect to the virial radius                                        
  '''
  return profile_nfw.NFWProfile(M = 10**log10mass, c = concentration,
                                z = z, mdef = mdef)

model_profiles = {'nfw': simulate_nfw}
