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
    'mc_relation': 'child18',
    'num_sims': 10000,
    'mc_scatter': 0,
    'min_log10mass': 13,
    'max_log10mass': 15,
    'num_radial_bins': 30,
    'mc_pair_subselect': 'all',
    'output_dir': 'example',
}

# Figuring out directory of where to output the sim_config
script_dir = os.path.dirname(__file__)
sim_rel_path = '../configs/simulations/' + sim_config['output_dir']
sim_dir = os.path.join(script_dir, sim_rel_path)
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)
sim_filename = os.path.join(sim_dir, 'sim_config.json')

# Write the sim_config to the appropriate directory
with open(sim_filename, "w") as outfile:
    json.dump(sim_config, outfile)
