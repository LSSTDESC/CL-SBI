''' 
Experiment setup
* One output_dir for each sim_config
* One directory for each infer config (including the sim_output_dir that it reads from)

Experiment:
* Generate simulations using sim_config (including the output dir)
* Run inference using infer_config, writing back to the same directory that infer_config is in


'''
import json

infer_config = {
    'sim_output_dir': 'examples',
    'inference_type': 'sbi',
    'profile_noise_dex': 0.3,
}

with open(f"{infer_config['sim_output_dir']}/infer_config.json",
          "w") as outfile:
    json.dump(infer_config, outfile)

plot_config = {
    'plot_type': 'cc',  # chainconsumer
}
