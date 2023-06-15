weaklensingclustersbi
========================

This repository contains tools for experiments for Weak Lensing Galaxy Clusters using Simulation Based Inference.



---------------

If you want to learn more about ...


---------------

Running instructions:

First, generate the sim_config file in the directory of interest:
python examples/sim_config_example.py

Next, generate the infer_config file in the directory of interest:
python examples/infer_config_example.py

Then, use this sim_config to run the numerical experiment in the sim_dir folder:
python examples/simulate_wl_profile.py --sim_dir example

Next, run inference using the sim_config in sim_dir and the infer_config in config_dir. This will output to the infer_config (so if you want to run the same inference with different simulations, you should specify multiple infer_configs for that).
python examples/mcmc_example.py --config_dir example --sim_dir example 
python examples/sbi_example.py --config_dir example --sim_dir example

Finally, we can plot the results using pygtc and chainconsumer
python examples/plot_example.py --config_dir example
