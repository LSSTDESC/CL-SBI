"""
Evaluate the training-seed ensemble (IR2/Zhou L586-594).

For each seed-retrained baseline net (and the production net as an extra sample), apply the
SBI FTJ inferrer to M=20 fresh observation realizations (fixed seeds, shared across nets) and
record the realization-mean population-summary bias and coverage. The across-net spread of the
per-net means is the SEED FLOOR: any per-net offset within it is network-init noise, not a
systematic of the method.
Results -> /tmp/seed_ensemble_eval.json (paper text only after discussion).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plot"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from weaklensclustersbi.inference import sbi_
from run_ensemble import draw_observation
from plot_calibration import compute_coverage, sample_truth_ftj, draw_posterior_samples
from plotutils import build_gaussian_summary_from_chain

M = 20
SD = os.path.dirname(__file__)
BASE = "sim_z1.infer_z1.10000.376.delta_sigma.corrected_mean"
NETS = [("production", f"../outputs/posteriors/{BASE}")] + \
       [(f"seed{n}", f"../outputs/posteriors/{BASE}.seed{n}") for n in (101, 102, 103, 104, 105)]

cfg = json.load(open(os.path.join(SD, "../configs/observations/obs_z1_lambda5.json")))
rbins = 10 ** np.arange(0, cfg["num_radial_bins"] / 10, 0.1)
truth_pop = sample_truth_ftj(cfg, n_samples=1000)

# pre-draw the shared realizations once (same seeds for every net)
reals = []
for r in range(M):
    np.random.seed(9000 + r)
    reals.append(draw_observation(cfg, 376, rbins, "delta_sigma"))

out = {}
for name, pdir in NETS:
    pdir = os.path.join(SD, pdir)
    if not os.path.isfile(os.path.join(pdir, "posterior.pickle")):
        print(f"SKIP {name}: missing {pdir}", flush=True); continue
    post = pickle.load(open(os.path.join(pdir, "posterior.pickle"), "rb"))
    post_jtf = pickle.load(open(os.path.join(pdir, "posterior_jtf.pickle"), "rb"))
    bM, bc, sM, sc, c68 = [], [], [], [], []
    for pairs, prof, log_sig in reals:
        ftj_c = np.asarray(sbi_.apply_observations(post, post_jtf, pairs, prof,
                           stack_estimator="corrected_mean", sigmas=log_sig)[1])
        summ = build_gaussian_summary_from_chain(ftj_c)
        bM.append(summ["mass"]["mu"] - np.median(pairs[:, 0]))
        bc.append(summ["concentration"]["mu"] - np.median(pairs[:, 1]))
        sM.append(summ["mass"]["sigma"]); sc.append(summ["concentration"]["sigma"])
        cov = compute_coverage(draw_posterior_samples(ftj_c, 4000), truth_pop)
        ks = sorted(cov.keys()); c68.append(cov[min(ks, key=lambda x: abs(x - 0.68))])
    out[name] = dict(bias_M=float(np.mean(bM)), bias_M_realsig=float(np.std(bM)),
                     bias_c=float(np.mean(bc)), bias_c_realsig=float(np.std(bc)),
                     sigma_M=float(np.mean(sM)), sigma_c=float(np.mean(sc)),
                     cov68=float(np.mean(c68)), cov68_realsig=float(np.std(c68)))
    print(f"{name:11s} bias_M={np.mean(bM):+.4f}±{np.std(bM):.4f}  bias_c={np.mean(bc):+.4f}±{np.std(bc):.4f}  "
          f"sigma_M={np.mean(sM):.4f}  sigma_c={np.mean(sc):.4f}  cov68={np.mean(c68):.3f}", flush=True)

nets = [v for k, v in out.items()]
if len(nets) > 2:
    for q in ("bias_M", "bias_c", "sigma_M", "sigma_c", "cov68"):
        vals = [n[q] for n in nets]
        print(f"SEED FLOOR {q:8s}: across-net mean {np.mean(vals):+.4f}, spread (std) {np.std(vals):.4f}", flush=True)
json.dump(out, open("/tmp/seed_ensemble_eval.json", "w"), indent=1)
print("DONE")
