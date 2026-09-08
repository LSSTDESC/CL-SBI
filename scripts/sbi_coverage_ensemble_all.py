"""
Realization-marginalized SBI FTJ coverage for ALL experiment rows (fixed-training nets).

Companion to hypermix_coverage_ensemble.py: repeats the coverage measurement over M fresh
observation realizations for every (trained posterior, obs config) combination in the paper's
coverage/KS analysis. SBI-only (amortized); MCMC rows are excluded deliberately — their
coverage departure is structural (posterior ~10x narrower than the population by
construction), so realization noise cannot affect the conclusion.
Results -> /tmp/sbi_cov_ensemble_all.json. DOES NOT touch the paper.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plot"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pickle
from weaklensclustersbi.inference import sbi_
from run_ensemble import draw_observation
from plot_calibration import compute_coverage, sample_truth_ftj, draw_posterior_samples

M = 30
CM = ".delta_sigma.corrected_mean"
SD = os.path.dirname(__file__)
COMBOS = [  # (sim_id, infer_id, obs_id) — the 16 driver rows
    ("sim_z1","infer_z1","obs_z1_lambda5"),
    ("sim_z1","infer_z1","obs_z1_lambda5_high_mc_scatter"),
    ("sim_z1","infer_z1","obs_z1_lambda5_high_noise"),
    ("sim_z1","infer_z1","obs_z1_lambda5_high_richness_contam"),
    ("sim_z1","infer_z1","obs_z1_lambda5_high_rm_scatter"),
    ("sim_z1","infer_z1","obs_z1_lambda5_low_richness_contam"),
    ("sim_z1","infer_z1","obs_z1_lambda5_ludlow"),
    ("sim_z1","infer_z1","obs_z1_lambda5_prada"),
    ("sim_z1_high_mc_scatter","infer_z1_high_mc_scatter","obs_z1_lambda5_high_mc_scatter"),
    ("sim_z1_high_mc_scatter","infer_z1_high_mc_scatter","obs_z1_lambda5"),
    ("sim_z1_high_noise","infer_z1","obs_z1_lambda5_high_noise"),
    ("sim_z1_high_noise","infer_z1","obs_z1_lambda5"),
    ("sim_z1_high_rm_scatter","infer_z1_high_rm_scatter","obs_z1_lambda5_high_rm_scatter"),
    ("sim_z1_high_rm_scatter","infer_z1_high_rm_scatter","obs_z1_lambda5"),
    ("sim_z1_high_rm_scatter","infer_z1","obs_z1_lambda5_high_rm_scatter"),
    ("sim_z1_high_rm_scatter","infer_z1","obs_z1_lambda5"),
]
out = {}
cache = {}
for sim, inf, obs in COMBOS:
    pd = os.path.join(SD, f"../outputs/posteriors/{sim}.{inf}.10000.376{CM}")
    if (sim, inf) not in cache:
        cache[(sim, inf)] = (pickle.load(open(f"{pd}/posterior.pickle","rb")),
                             pickle.load(open(f"{pd}/posterior_jtf.pickle","rb")))
    post, post_jtf = cache[(sim, inf)]
    cfg = json.load(open(os.path.join(SD, f"../configs/observations/{obs}.json")))
    rbins = 10 ** np.arange(0, cfg["num_radial_bins"] / 10, 0.1)
    truth = sample_truth_ftj(cfg, n_samples=1000)
    c68, c95 = [], []
    for r in range(M):
        np.random.seed(8000 + r)
        pairs, prof, log_sig = draw_observation(cfg, 376, rbins, "delta_sigma")
        ftj_c = sbi_.apply_observations(post, post_jtf, pairs, prof,
                                        stack_estimator="corrected_mean", sigmas=log_sig)[1]
        samples = draw_posterior_samples(np.asarray(ftj_c), 4000)
        cov = compute_coverage(samples, truth)
        ks = sorted(cov.keys())
        c68.append(cov[min(ks, key=lambda x: abs(x-0.68))])
        c95.append(cov[min(ks, key=lambda x: abs(x-0.95))])
    key = f"{sim}.{inf}.{obs}"
    out[key] = dict(c68_mean=float(np.mean(c68)), c68_std=float(np.std(c68)),
                    c95_mean=float(np.mean(c95)), c95_std=float(np.std(c95)))
    print(f"{key:75s} 68%: {np.mean(c68):.3f}±{np.std(c68):.3f}  95%: {np.mean(c95):.3f}±{np.std(c95):.3f}", flush=True)
    json.dump(out, open("/tmp/sbi_cov_ensemble_all.json","w"), indent=1)
print("DONE")
