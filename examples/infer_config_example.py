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
    'config_dir': 'example',
    'priors': {
        'min_log10mass': 0,
        'max_log10mass': 20,
        'min_concentration': 0,
        'max_concentration': 20,
    }
}

# Figuring out directory of where to output the infer_config
script_dir = os.path.dirname(__file__)
config_rel_path = '../configs/inference/' + infer_config['config_dir']
config_dir = os.path.join(script_dir, config_rel_path)
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
infer_filename = os.path.join(config_dir, 'infer_config.json')

with open(infer_filename, "w") as outfile:
    json.dump(infer_config, outfile)
