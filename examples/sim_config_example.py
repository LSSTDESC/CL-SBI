''' 
Experiment setup
* One output_dir for each sim_config
* One directory for each infer config (including the sim_output_dir that it reads from)

Experiment:
* Generate simulations using sim_config (including the output dir)
* Run inference using infer_config, writing back to the same directory that infer_config is in


'''
import json

sim_config = {
    'rm_relation': 'murata17',
    'mc_relation': 'child18',

    # number of simulations
    'num_samples': 10000,
    'sample_noise_dex': 0,

    # richness band and number of masses drawn from it
    'min_richness': 30,
    'max_richness': 40,
    'num_drawn': 10,
    # TODO: how do we want to add other noise "profiles" other than dex
    'drawn_noise_dex': 0.2,
    'num_radial_bins': 30,
    'sim_output_dir': 'examples'
}

with open(f"{sim_config['sim_output_dir']}/sim_config.json", "w") as outfile:
    json.dump(sim_config, outfile)
