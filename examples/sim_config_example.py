''' 
Experiment setup
* One output_dir for each sim_config
* One directory for each infer config (including the sim_output_dir that it reads from)

Experiment:
* Generate simulations using sim_config (including the output dir)
* Run inference using infer_config, writing back to the same directory that infer_config is in


'''
import json
import os

sim_config = {
    'rm_relation': 'murata17',
    'mc_relation': 'child18',

    # number of simulations
    'num_parameter_samples': 10000,
    'sample_noise_dex': 0,

    # richness band and number of masses drawn from it
    'min_richness': 30,
    'max_richness': 40,
    'num_sims': 10,
    # TODO: how do we want to add other noise "profiles" other than dex
    'drawn_noise_dex': 0.2,
    'num_radial_bins': 30,
    # TODO: should this be passed in from command line for consistency?
    'sim_output_dir': 'example'
}

# Figuring out directory of where to output the sim_config
script_dir = os.path.dirname(__file__)
sim_rel_path = '../simulations/' + sim_config['sim_output_dir']
sim_dir = os.path.join(script_dir, sim_rel_path)
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)
sim_filename = os.path.join(sim_dir, 'sim_config.json')

with open(sim_filename, "w") as outfile:
    json.dump(sim_config, outfile)
