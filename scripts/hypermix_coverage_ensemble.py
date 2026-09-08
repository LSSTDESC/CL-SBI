"""
Realization-marginalized coverage for the hyperparameter-marginalized (hypermix) SBI net.

The paper's coverage curves are computed from a single observation realization per test;
this script repeats the SBI FTJ coverage over M fresh realizations (fixed seeds) to separate
persistent effects (e.g. edge-of-training-prior under-coverage) from realization noise.
Truth reference = the fixed population sample from the obs config (population-truth
convention); only the observed 376-cluster draw varies. SBI-only: fully amortized, minutes.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plot"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pickle

from weaklensclustersbi.inference import sbi_
from run_ensemble import draw_observation
from plot_calibration import compute_coverage, sample_truth_ftj, draw_posterior_samples

M = 30
SD = os.path.dirname(__file__)
PD = os.path.join(SD, "../outputs/posteriors/sim_z1_hypermix.infer_z1.10000.376.delta_sigma.corrected_mean")
post = pickle.load(open(f"{PD}/posterior.pickle", "rb"))
post_jtf = pickle.load(open(f"{PD}/posterior_jtf.pickle", "rb"))

TESTS = ["obs_z1_lambda5", "obs_z1_lambda5_prada", "obs_z1_lambda5_ludlow",
         "obs_z1_lambda5_high_mc_scatter", "obs_z1_lambda5_high_rm_scatter"]
out = {}
for obs_id in TESTS:
    cfg = json.load(open(os.path.join(SD, f"../configs/observations/{obs_id}.json")))
    rbins = 10 ** np.arange(0, cfg["num_radial_bins"] / 10, 0.1)
    truth = sample_truth_ftj(cfg, n_samples=1000)
    c68, c95 = [], []
    for r in range(M):
        np.random.seed(7000 + r)
        pairs, prof, log_sig = draw_observation(cfg, 376, rbins, "delta_sigma")
        ftj_c = sbi_.apply_observations(post, post_jtf, pairs, prof,
                                        stack_estimator="corrected_mean", sigmas=log_sig)[1]
        samples = draw_posterior_samples(np.asarray(ftj_c), 4000)
        cov = compute_coverage(samples, truth)
        ks = sorted(cov.keys())
        c68.append(cov[min(ks, key=lambda x: abs(x - 0.68))])
        c95.append(cov[min(ks, key=lambda x: abs(x - 0.95))])
    out[obs_id] = dict(c68_mean=float(np.mean(c68)), c68_std=float(np.std(c68)),
                       c95_mean=float(np.mean(c95)), c95_std=float(np.std(c95)),
                       c68_all=[float(x) for x in c68])
    print(f"{obs_id:42s} 68%: {np.mean(c68):.3f} +/- {np.std(c68):.3f}   "
          f"95%: {np.mean(c95):.3f} +/- {np.std(c95):.3f}", flush=True)
json.dump(out, open("/tmp/hypermix_cov_ensemble.json", "w"), indent=1)
print("DONE")
