"""
Sampling-robustness ensemble (Payerne P2-d): draw M seed-matched fresh observation
realizations for an experiment and record the observed (truth) distribution + recovered
posterior summary per realization, so headline quantities can be reported as
mean +/- run-to-run scatter rather than depending on a single N_c-cluster draw.

INCREMENTAL & RESUMABLE: each realization r has a fixed seed (--seed + r) and is stored in
its own file (realizations/r###.json). SBI (cheap, all realizations) and MCMC (slow, gated
by --run_mcmc/--n_mcmc) are tracked independently, so you can run e.g. 12 MCMC now and 38
later (they become r=0..11 then r=12..49), and a killed run resumes where it left off.

SBI-only (fast):   python3 run_ensemble.py ... --n_realizations 50
Add MCMC (slow):   python3 run_ensemble.py ... --n_realizations 50 --run_mcmc --n_mcmc 12
Extend MCMC later: python3 run_ensemble.py ... --n_realizations 50 --run_mcmc --n_mcmc 50
"""
import sys, os, json, pickle, argparse, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plot"))
warnings.filterwarnings("ignore")
import numpy as np
import multiprocessing
from weaklensclustersbi.simulations import population, wlprofile
from weaklensclustersbi.inference import sbi_, mcmc
from plot.plotutils import build_gaussian_summary_from_chain


def draw_observation(cfg, num_obs, rbins, observable):
    pairs = np.array(population.gen_mc_pairs_in_richness_bin(
        cfg["min_richness"], cfg["max_richness"], rm_relation=cfg["rm_relation"],
        mc_relation=cfg["mc_relation"], num_samples=num_obs, mc_scatter=cfg["mc_scatter"],
        rm_scatter=cfg["rm_scatter"], min_z=cfg["min_z"], max_z=cfg["max_z"],
        richness_contam_frac=cfg.get("richness_contam_frac", 0.0),
        lambda_min_contam=cfg.get("min_richness_contam", None)))
    z = np.random.uniform(cfg["min_z"], cfg["max_z"], size=len(pairs))
    noiseless = np.array([wlprofile.simulate_nfw(m, c, rbins, zz, kind=observable)
                          for (m, c), zz in zip(pairs, z)])
    noisy = population.calculate_noise(noiseless, cfg["profile_noise_dex"])
    return pairs, np.log10(noisy), np.std(np.log10(noisy), axis=0)


def sbi_summary(chain):
    s = build_gaussian_summary_from_chain(np.asarray(chain))
    return [s["mass"]["mu"], s["mass"]["sigma"], s["concentration"]["mu"],
            s["concentration"]["sigma"], s["correlation"]["rho"]]


def chain_summary(ch):
    return [float(np.median(ch[:, 0])), float(np.std(ch[:, 0])),
            float(np.median(ch[:, 1])), float(np.std(ch[:, 1]))]


def main():
    ap = argparse.ArgumentParser()
    for a in ("sim_id", "infer_id", "obs_id", "num_sims", "num_obs"):
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--observable", default="surface_density")
    ap.add_argument("--n_realizations", type=int, default=50)
    ap.add_argument("--run_mcmc", action="store_true")
    ap.add_argument("--n_mcmc", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    sd = os.path.dirname(__file__)
    suf = "" if args.observable == "surface_density" else f".{args.observable}"
    cfg = json.load(open(os.path.join(sd, f"../configs/observations/{args.obs_id}.json")))
    infer = json.load(open(os.path.join(sd, f"../configs/inference/{args.infer_id}.json")))
    infer["priors"]["observable"] = args.observable
    rbins = 10 ** np.arange(0, cfg["num_radial_bins"] / 10, 0.1)
    pdir = os.path.join(sd, f"../outputs/posteriors/{args.sim_id}.{args.infer_id}.{args.num_sims}.{args.num_obs}{suf}")
    posterior = pickle.load(open(os.path.join(pdir, "posterior.pickle"), "rb"))
    posterior_jtf = pickle.load(open(os.path.join(pdir, "posterior_jtf.pickle"), "rb"))

    key = f"{args.sim_id}.{args.infer_id}.{args.obs_id}.{args.num_sims}.{args.num_obs}{suf}"
    base = os.path.join(sd, f"../outputs/ensembles/{key}")
    rdir = os.path.join(base, "realizations")
    os.makedirs(rdir, exist_ok=True)

    no = int(args.num_obs)
    n_mcmc = args.n_mcmc if args.run_mcmc else 0
    N = max(args.n_realizations, n_mcmc)
    t0 = time.perf_counter()
    for r in range(N):
        fp = os.path.join(rdir, f"r{r:03d}.json")
        rec = json.load(open(fp)) if os.path.isfile(fp) else {}
        need_sbi = (r < args.n_realizations) and ("sbi_ftj" not in rec)
        need_mcmc = (r < n_mcmc) and (rec.get("mcmc_ftj") is None)
        if not (need_sbi or need_mcmc):
            continue
        np.random.seed(args.seed + r)
        pairs, prof, log_sig = draw_observation(cfg, no, rbins, args.observable)
        if need_sbi:
            jtf_c, ftj_c = sbi_.apply_observations(posterior, posterior_jtf, pairs, prof)[:2]
            rec["truth"] = [float(np.median(pairs[:, 0])), float(np.std(pairs[:, 0])),
                            float(np.median(pairs[:, 1])), float(np.std(pairs[:, 1]))]
            rec["sbi_jtf"] = sbi_summary(jtf_c); rec["sbi_ftj"] = sbi_summary(ftj_c)
        if need_mcmc:
            with multiprocessing.Pool() as pool:
                jtf_sig = np.sqrt(np.pi / 2) * log_sig / np.sqrt(no)
                mj = mcmc.join_then_fit(prof, jtf_sig, infer["priors"], pool=pool)[0]
                mf = mcmc.fit_then_join(prof, log_sig, infer["priors"], pool=pool)[0]
            rec["mcmc_jtf"] = chain_summary(mj); rec["mcmc_ftj"] = chain_summary(mf)
        json.dump(rec, open(fp, "w"))
        print(f"  r{r:03d} {'S' if need_sbi else ' '}{'M' if need_mcmc else ' '} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # aggregate all realizations present
    recs = [json.load(open(os.path.join(rdir, f))) for f in sorted(os.listdir(rdir)) if f.endswith(".json")]
    def arr(k): return np.array([r[k] for r in recs if k in r and r[k] is not None])
    np.savez(os.path.join(base, "ensemble.npz"),
             truth=arr("truth"), sbi_ftj=arr("sbi_ftj"), sbi_jtf=arr("sbi_jtf"),
             mcmc_ftj=arr("mcmc_ftj"), mcmc_jtf=arr("mcmc_jtf"))
    t, sf, mf = arr("truth"), arr("sbi_ftj"), arr("mcmc_ftj")
    with open(os.path.join(base, "summary.txt"), "w") as f:
        f.write(f"# {key}  M_sbi={len(sf)} M_mcmc={len(mf)}\n\n")
        def line(nm, x): f.write(f"{nm:<20} {np.mean(x):+.4f} +/- {np.std(x):.4f}\n")
        f.write("OBSERVED (truth):\n")
        for i, nm in enumerate(["median logM", "sigma logM", "median c", "sigma c"]): line(nm, t[:, i])
        f.write("\nSBI FTJ recovered:\n")
        for i, nm in enumerate(["mu logM", "sigma logM", "mu c", "sigma c"]): line(nm, sf[:, i])
        f.write("\nSBI FTJ bias (recovered-truth):\n")
        line("logM median", sf[:, 0] - t[:, 0]); line("c median", sf[:, 2] - t[:, 2])
        if len(mf):
            f.write(f"\nMCMC FTJ ({len(mf)} realizations):\n")
            for i, nm in enumerate(["median logM", "sigma logM", "median c", "sigma c"]): line(nm, mf[:, i])
    print(f"saved -> {base}  (SBI={len(sf)}, MCMC={len(mf)}, {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
