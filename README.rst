weaklensingclustersbi
========================

This repository contains tools for experiments for Weak Lensing Galaxy Clusters using Simulation Based Inference.



---------------

If you want to learn more about ...


---------------

Running instructions:

First, generate the sim_config file in the directory of interest:
python examples/sim_config_example.py

Then, use this sim_config to run the numerical experiment in the folder.

python examples/simulate_wl_profile.py
python examples/mcmc_example.py
python examples/sbi_example.py
python examples/plot_example.py