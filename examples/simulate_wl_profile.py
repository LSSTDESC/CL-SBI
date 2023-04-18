"""
Example script to simulate a weak lensing profile using modules
"""


# Define a mass sample with some noise, and concentration sample that scatters
# about the theoretical prediction

log10mass_sample = np.random.uniform(13,15,size=10000)
non_noisy_concentration_sample = get_concentration(log10mass_sample)
noisy_concentration = np.random.normal(non_noisy_concentration_sample,0.2,10000)


