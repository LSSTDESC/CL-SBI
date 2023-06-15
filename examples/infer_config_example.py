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

infer_config = {
    'inference_type': 'sbi',
    'profile_noise_dex': 0.2,
    'config_dir': '0.2',
    'mc_pair_subselect': 'all',
}

# Figuring out directory of where to output the infer_config
script_dir = os.path.dirname(__file__)
config_rel_path = '../configs/' + infer_config['config_dir']
config_dir = os.path.join(script_dir, config_rel_path)
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
infer_filename = os.path.join(config_dir, 'infer_config.json')

with open(infer_filename, "w") as outfile:
    json.dump(infer_config, outfile)

# plot_config = {
#     'plot_type': 'cc',  # chainconsumer
# }
