"""
Paper-2 apples-to-apples analysis: broadened-training (mc-mix) SBI FTJ vs the
Paper-I child18-only SBI FTJ on the ludlow/prada OOD observation sets.

For each obs set, runs the trained mc-mix FTJ posterior on the observation,
collapses the 15-dim percentile vector into a Gaussian population estimate
(mu_M, sig_M, mu_c, sig_c), and compares to:
  (a) the OLD child18-only SBI FTJ values from _all_methods_rows.pkl
  (b) truth from outputs/observations/{obs}.376/drawn_mc_pairs.npy

Writes results to notebooks/hierarchical_poc_outputs/ftj_mcmix_results.pkl.

Run AFTER train_inferrer.py has produced the mc-mix posterior, e.g.:
  python analyze_ftj_mcmix.py --sim_id sim_z1_mcmix --infer_id infer_z1_mcmix \
      --num_sims 10000 --num_obs 376
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pickle
import numpy as np
import torch

from weaklensclustersbi.inference.sbiutils import (
    create_fit_join_observation_nfw,
    PERCENTILE_LEVELS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--sim_id", default="sim_z1_mcmix")
parser.add_argument("--infer_id", default="infer_z1_mcmix")
parser.add_argument("--num_sims", default="10000")
parser.add_argument("--num_obs", default="376")
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
repo = os.path.join(script_dir, "..")

OBS_SETS = ["obs_z1_lambda5", "obs_z1_lambda5_ludlow", "obs_z1_lambda5_prada"]

# ---- Load the broadened-training FTJ posterior ----
post_path = os.path.join(
    repo,
    "outputs",
    "posteriors",
    f"{args.sim_id}.{args.infer_id}.{args.num_sims}.{args.num_obs}",
    "posterior.pickle",
)
with open(post_path, "rb") as fh:
    posterior = pickle.load(fh)

# ---- Load OLD child18-only FTJ rows ----
old_rows_path = os.path.join(
    repo, "notebooks", "hierarchical_poc_outputs", "_all_methods_rows.pkl"
)
with open(old_rows_path, "rb") as fh:
    old_rows = pickle.load(fh)


def lookup_old(obs):
    """Find the child18-only SBI FTJ row for this obs set."""
    if isinstance(old_rows, list):
        for r in old_rows:
            if r.get("obs") == obs:
                return {
                    "sbi_muM": r.get("sbi_muM"),
                    "sbi_sigM": r.get("sbi_sigM"),
                    "sbi_muc": r.get("sbi_muc"),
                    "sbi_sigc": r.get("sbi_sigc"),
                }
    elif isinstance(old_rows, dict):
        r = old_rows.get(obs)
        if r is not None:
            return {
                "sbi_muM": r.get("sbi_muM"),
                "sbi_sigM": r.get("sbi_sigM"),
                "sbi_muc": r.get("sbi_muc"),
                "sbi_sigc": r.get("sbi_sigc"),
            }
    return None


def percentile_vec_to_population(vec):
    """
    Collapse a 15-dim percentile summary (7 M-percentiles, 7 c-percentiles,
    1 correlation) into a Gaussian population estimate the way Paper I does:
      mu  = the 50th percentile (median)
      sig = (P84 - P16) / 2  (symmetric 1-sigma half-width from percentiles)
    """
    vec = np.asarray(vec)
    n = len(PERCENTILE_LEVELS)
    m_perc = vec[:n]
    c_perc = vec[n : 2 * n]
    i50 = PERCENTILE_LEVELS.index(50)
    i16 = PERCENTILE_LEVELS.index(16)
    i84 = PERCENTILE_LEVELS.index(84)
    mu_M = float(m_perc[i50])
    sig_M = float((m_perc[i84] - m_perc[i16]) / 2.0)
    mu_c = float(c_perc[i50])
    sig_c = float((c_perc[i84] - c_perc[i16]) / 2.0)
    return mu_M, sig_M, mu_c, sig_c


results = {}
for obs in OBS_SETS:
    obs_dir = os.path.join(repo, "outputs", "observations", f"{obs}.{args.num_obs}")
    drawn_mc_pairs = np.load(os.path.join(obs_dir, "drawn_mc_pairs.npy"))
    drawn_nfw = np.load(os.path.join(obs_dir, "drawn_nfw_profiles.npy"))

    # FTJ observation = all profiles concatenated; posterior samples a 15-dim vec
    _theta_o, x_o = create_fit_join_observation_nfw(drawn_mc_pairs, drawn_nfw)
    samples = posterior.sample((10000,), x=x_o)
    logp = posterior.log_prob(samples, x=x_o)
    map_vec = samples[torch.argmax(logp)].numpy()

    mu_M, sig_M, mu_c, sig_c = percentile_vec_to_population(map_vec)

    truth_muM = float(np.mean(drawn_mc_pairs[:, 0]))
    truth_sigM = float(np.std(drawn_mc_pairs[:, 0]))
    truth_muc = float(np.mean(drawn_mc_pairs[:, 1]))
    truth_sigc = float(np.std(drawn_mc_pairs[:, 1]))

    results[obs] = {
        "mcmix_muM": mu_M,
        "mcmix_sigM": sig_M,
        "mcmix_muc": mu_c,
        "mcmix_sigc": sig_c,
        "old_child18": lookup_old(obs),
        "truth_muM": truth_muM,
        "truth_sigM": truth_sigM,
        "truth_muc": truth_muc,
        "truth_sigc": truth_sigc,
    }

out_path = os.path.join(
    repo, "notebooks", "hierarchical_poc_outputs", "ftj_mcmix_results.pkl"
)
with open(out_path, "wb") as fh:
    pickle.dump(results, fh)

# ---- Print comparison table ----
hdr = f"{'obs':28s} {'src':8s} {'muM':>7s} {'sigM':>7s} {'muc':>7s} {'sigc':>7s}"
print(hdr)
print("-" * len(hdr))
for obs, r in results.items():
    print(
        f"{obs:28s} {'mcmix':8s} {r['mcmix_muM']:7.3f} {r['mcmix_sigM']:7.3f} "
        f"{r['mcmix_muc']:7.3f} {r['mcmix_sigc']:7.3f}"
    )
    old = r["old_child18"]
    if old is not None:
        def g(k):
            v = old.get(k)
            return float(v) if v is not None else float("nan")
        print(
            f"{'':28s} {'child18':8s} {g('sbi_muM'):7.3f} {g('sbi_sigM'):7.3f} "
            f"{g('sbi_muc'):7.3f} {g('sbi_sigc'):7.3f}"
        )
    print(
        f"{'':28s} {'truth':8s} {r['truth_muM']:7.3f} {r['truth_sigM']:7.3f} "
        f"{r['truth_muc']:7.3f} {r['truth_sigc']:7.3f}"
    )
print(f"\nSaved -> {out_path}")
