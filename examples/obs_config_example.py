import json
import os

obs_config = {
    'rm_relation': 'murata17',
    'mc_relation': 'child18',
    'min_richness': 30,
    'max_richness': 40,
    'num_obs': 10,
    # TODO: how do we want to add other noise "profiles" other than dex
    'drawn_noise_dex': 0,
    'profile_noise_dex': 0.1,
    'num_radial_bins': 30,
    'output_dir': 'example',
}

# Figuring out directory of where to output the obs_config
script_dir = os.path.dirname(__file__)
obs_rel_path = '../configs/observations/' + obs_config['output_dir']
obs_dir = os.path.join(script_dir, obs_rel_path)
if not os.path.exists(obs_dir):
    os.makedirs(obs_dir)
obs_filename = os.path.join(obs_dir, 'obs_config.json')

with open(obs_filename, "w") as outfile:
    json.dump(obs_config, outfile)
