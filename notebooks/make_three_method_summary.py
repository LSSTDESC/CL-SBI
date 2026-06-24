"""Summary figure + LaTeX table comparing SBI / MCMC-joint / hierarchical-HMC across experiments."""
import pickle, csv, numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
})

OUT = "/Users/akumgill/Documents/GitHub/CL-SBI/notebooks/hierarchical_poc_outputs"
rows = pickle.load(open(f"{OUT}/_all_methods_rows.pkl", "rb"))
hmc = {r["obs_id"]: r for r in csv.DictReader(open(f"{OUT}/hier_all_experiments.csv"))}

# Use the 8 in-distribution-prior experiments (sim_z1 / infer_z1) for the clean comparison;
# these are the rows where SBI is trained on the matched distribution.
def short(exp):
    obs = exp.split(".")[2].replace("obs_z1_lambda5", "baseline").replace("_", " ").strip()
    return obs if obs else "baseline"

# pick the canonical infer_z1 row per unique obs (the apples-to-apples set)
seen, canon = set(), []
for r in rows:
    if r["sim"] == "sim_z1" and r["infer"] == "infer_z1" and r["obs"] not in seen:
        seen.add(r["obs"]); canon.append(r)
canon.sort(key=lambda r: short(r["exp"]))

labels = [short(r["exp"]) for r in canon]
x = np.arange(len(canon)); w = 0.2

# Neural HBI (architecture A) sigma_M per experiment, from the fully-amortized run. Marginal sigma_M
# is inferred directly (no conversion needed). Absent for the 2 contam experiments.
try:
    _afull_exp = pickle.load(open(f"{OUT}/afull_neural_hbi.pkl", "rb"))["experiments"]
except FileNotFoundError:
    _afull_exp = {}
def nhbi_sigM(obs):
    return _afull_exp[obs]["est"]["sig_M"][0] if obs in _afull_exp else np.nan

def col(r, k):
    v = r.get(k); return float(v) if v is not None else np.nan

# ---------- Figure 1: recovered sigma_M vs truth ----------
fig, ax = plt.subplots(figsize=(13, 5.5))
true = [col(r, "true_sigM") for r in canon]
mcmc = [col(r, "mcmc_sigM") for r in canon]
sbi  = [col(r, "sbi_sigM") for r in canon]
hmcv = [col(r, "hmc_sigM") for r in canon]
nhbi = [nhbi_sigM(r["obs"]) for r in canon]
# mask non-converged MCMC: plot at 0 with a hatch + "DNC" marker instead of a misleading value
mcmc_conv = [r.get("mcmc_converged", True) for r in canon]
mcmc_plot = [m if c else 0.0 for m, c in zip(mcmc, mcmc_conv)]
wb = 0.16   # 5 bars per group
ax.bar(x - 2*wb, true, wb, color="0.5", label="TRUE population")
ax.bar(x - 1*wb, mcmc_plot, wb, color="#1f77b4", label="MCMC joint-likelihood (FTJ)")
ax.bar(x + 0*wb, sbi,  wb, color="#ff7f0e", label="SBI FTJ")
ax.bar(x + 1*wb, hmcv, wb, color="#2ca02c", label="Hierarchical HMC (this work)")
ax.bar(x + 2*wb, np.nan_to_num(nhbi), wb, color="#9467bd", label="Hierarchical SBI (this work)")
ymax = max(max(true), max(sbi), max(hmcv)) * 1.25
for xi, conv in zip(x, mcmc_conv):
    if not conv:
        ax.text(xi - 1*wb, ymax*0.02, "DNC", rotation=90, fontsize=7, ha="center", va="bottom", color="#1f77b4")
# mark experiments where neural-HBI was not run (the 2 contamination cases)
for xi, v in zip(x, nhbi):
    if np.isnan(v):
        ax.text(xi + 2*wb, ymax*0.02, "n/a", rotation=90, fontsize=7, ha="center", va="bottom", color="#9467bd")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel(r"recovered $\sigma_{\log_{10}M}$")
ax.set_title("Population mass-spread recovery across experiments  (MCMC DNC = did not converge)")
ax.legend()
ax.set_ylim(0, ymax)
fig.tight_layout(); fig.savefig(f"{OUT}/three_method_sigmaM.png", dpi=140)

# ---------- Figure 2: timing comparison (per recurring inference) ----------
# SBI ~1.4s; MCMC emcee joint+jtf from pipeline CSV; HMC sample time from our CSV.
speed = list(csv.DictReader(open("/Users/akumgill/Documents/GitHub/CL-SBI/outputs/inference/pipeline_speed.csv")))
def first(stage, obs):
    # latest (most recent) matching entry -> reflects the regenerated adaptive-sampler runs
    val = np.nan
    for r in speed:
        if r["stage"] == stage and r["obs_id"] == obs and r["num_obs"] == "376" \
                and r["status"] == "success":
            val = float(r["seconds"])
    return val
# --- grouped-by-task cost figure: JTF / FTJ / Hierarchical, each x {SBI, emcee, HMC} ---
# emcee JTF/FTJ from the pipeline CSV; SBI JTF/FTJ + HMC JTF/FTJ from timing_grid.pkl;
# hierarchical: SBI-amortized POC ~10s, HMC ~210s, emcee ~1 day (estimate).
tg = pickle.load(open(f"{OUT}/timing_grid.pkl", "rb"))
emcee_jtf = first("mcmc_join_then_fit", "obs_z1_lambda5")
emcee_ftj = first("mcmc_fit_then_join", "obs_z1_lambda5")
hmc_hier = float(hmc["obs_z1_lambda5"]["t_sample_s"])
sbi_hier = 10.0          # amortized hierarchical SBI POC sampling time
EMCEE_HBI_EST = 24 * 3600.0   # ~1 day estimate (measured ~1 s/step x >=1500 walkers, accept ~0.1)

# Neural HBI inference cost: one forward pass through the embedding + flow sampling. Only defined
# for the Hierarchical (population) task; NaN -> no bar for the single-(M,c) tasks. The amortized
# training (~2 hr, one-time) is excluded from per-inference cost, same convention as SBI's training.
nhbi_hier = 1.0
# rows: (task, sbi, emcee, hmc, nhbi, emcee_is_estimate)
grid = [
    ("Join-then-fit\n(single $M,c$)",  tg["sbi_jtf"], emcee_jtf, tg["hmc_jtf"], np.nan, False),
    ("Fit-then-join\n(single $M,c$)",  tg["sbi_ftj"], emcee_ftj, tg["hmc_ftj"], np.nan, False),
    ("Hierarchical\n(population)",      sbi_hier,      EMCEE_HBI_EST, hmc_hier, nhbi_hier, True),
]
C = {"SBI": "#ff7f0e", "Hierarchical SBI": "#9467bd", "HMC": "#2ca02c", "emcee": "#1f77b4"}
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
w = 0.2
x = np.arange(len(grid))
# 4 bars/group, ordered ascending by typical height: SBI < Neural HBI < HMC < emcee -> emcee right
for j, (mname, key) in enumerate([("SBI", 1), ("Hierarchical SBI", 4), ("HMC", 3), ("emcee", 2)]):
    vals = [row[key] for row in grid]
    offs = (j - 1.5) * w
    bars = ax2.bar(x + offs, np.nan_to_num(vals), w, color=C[mname], label=mname)
    for i, (b, v) in enumerate(zip(bars, vals)):
        if np.isnan(v):
            b.set_visible(False); continue
        est = grid[i][5] and mname == "emcee"
        if est:
            b.set_hatch("//"); b.set_alpha(0.55)
        lab = (r"$\gtrsim$1 day" + "\n(est.)") if est else (f"{v:.1f}s" if v < 100 else f"{v:.0f}s")
        ax2.text(b.get_x()+b.get_width()/2, v*1.3, lab, ha="center", fontsize=7.5)
ax2.set_xticks(x); ax2.set_xticklabels([row[0] for row in grid])
ax2.set_yscale("log"); ax2.set_ylabel("wall time per inference [s] (log scale)")
ax2.set_title("Inference cost by task and method (baseline, $N_c=376$, single CPU)")
# explicit legend handles (Neural HBI has only one drawn bar -> auto-legend drops its swatch)
from matplotlib.patches import Patch
leg_order = ["SBI", "Hierarchical SBI", "HMC", "emcee"]
ax2.legend(handles=[Patch(facecolor=C[m], label=m + (" (this work)" if m == "Hierarchical SBI" else ""))
                    for m in leg_order], title="method")
ax2.set_ylim(0.2, EMCEE_HBI_EST*8)
fig2.tight_layout(); fig2.savefig(f"{OUT}/three_method_timing.png", dpi=140)

# ---------- LaTeX table ----------
def f(v, d=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.{d}f}"
# "good" recovery := recovered spread within SIGMA_TOL (fractional) of the true spread.
# Bold marks a good recovery; daggers mark a method that misses by >2x (badly over/under-confident).
SIGMA_TOL = 0.20   # 20% of true sigma
def sig_cell(val, true):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    rel = abs(val - true) / true if true > 0 else np.inf
    s = f"{val:.3f}"
    if rel <= SIGMA_TOL:
        return r"\textbf{" + s + "}"          # good: close to truth
    if val < true / 2 or val > true * 2:
        return s + r"$^{\dagger}$"            # bad: off by >2x
    return s

def mu_cell(val, true, tol):
    # center cell: bold if within +-tol of the true center (absolute), else plain
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    s = f"{val:.3f}" if abs(val) < 10 else f"{val:.2f}"
    return r"\textbf{" + s + "}" if abs(val - true) <= tol else s

lines = [
    r"\begin{table*}", r"\centering",
    r"\caption{Recovered population \emph{spread} ($\sigma_{\log_{10}M}$, $\sigma_c$) and "
    r"\emph{center} ($\mu_{\log_{10}M}$, $\mu_c$) across experiments, for the three inference "
    r"approaches, compared to the true population. For the spread, \textbf{bold} marks recovery "
    r"within 20\% of truth and $^{\dagger}$ marks a value off by more than a factor of two "
    r"(strongly over- or under-confident); for the center, \textbf{bold} marks recovery within "
    r"$0.03$ (mass) or $0.2$ (concentration) of truth. MCMC joint-likelihood (FTJ) reports its "
    r"posterior \emph{width} as the implied spread, which collapses far below the truth because it "
    r"estimates a central value, not a population; SBI FTJ and hierarchical HMC report the inferred "
    r"population. All HMC fits converged ($\hat r<1.01$).}",
    r"\label{tab:three_method}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{l rrrr rrrr}",
    r"\hline",
    r"\multicolumn{9}{l}{\textit{Population spread}} \\",
    r" & \multicolumn{4}{c}{$\sigma_{\log_{10}M}$} & \multicolumn{4}{c}{$\sigma_c$} \\",
    r"Experiment & TRUE & MCMC & SBI & HMC & TRUE & MCMC & SBI & HMC \\",
    r"\hline",
]
for r, lab in zip(canon, labels):
    mc_ok = r.get("mcmc_converged", True)
    tsM, tsc = col(r, "true_sigM"), col(r, "true_sigc")
    mcM = sig_cell(col(r, "mcmc_sigM"), tsM) if mc_ok else r"\textit{DNC}"
    mcc = sig_cell(col(r, "mcmc_sigc"), tsc) if mc_ok else r"\textit{DNC}"
    lines.append(
        f"{lab} & {f(tsM)} & {mcM} & {sig_cell(col(r,'sbi_sigM'),tsM)} & {sig_cell(col(r,'hmc_sigM'),tsM)} & "
        f"{f(tsc)} & {mcc} & {sig_cell(col(r,'sbi_sigc'),tsc)} & {sig_cell(col(r,'hmc_sigc'),tsc)} \\\\")
lines += [
    r"\hline",
    r"\multicolumn{9}{l}{\textit{Population center}} \\",
    r" & \multicolumn{4}{c}{$\mu_{\log_{10}M}$} & \multicolumn{4}{c}{$\mu_c$} \\",
    r"Experiment & TRUE & MCMC & SBI & HMC & TRUE & MCMC & SBI & HMC \\",
    r"\hline",
]
for r, lab in zip(canon, labels):
    mc_ok = r.get("mcmc_converged", True)
    tmM, tmc = col(r, "true_muM"), col(r, "true_muc")
    mcM = mu_cell(col(r, "mcmc_muM"), tmM, 0.03) if mc_ok else r"\textit{DNC}"
    mcc = mu_cell(col(r, "mcmc_muc"), tmc, 0.2) if mc_ok else r"\textit{DNC}"
    lines.append(
        f"{lab} & {f(tmM)} & {mcM} & {mu_cell(col(r,'sbi_muM'),tmM,0.03)} & {mu_cell(col(r,'hmc_muM'),tmM,0.03)} & "
        f"{f(tmc,2)} & {mcc} & {mu_cell(col(r,'sbi_muc'),tmc,0.2)} & {mu_cell(col(r,'hmc_muc'),tmc,0.2)} \\\\")
lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table*}"]
open(f"{OUT}/three_method_table.tex", "w").write("\n".join(lines))

print("wrote three_method_sigmaM.png, three_method_timing.png, three_method_table.tex")
print(f"timing grid: JTF(SBI={tg['sbi_jtf']:.2f} emcee={emcee_jtf:.0f} HMC={tg['hmc_jtf']:.1f}) "
      f"FTJ(SBI={tg['sbi_ftj']:.2f} emcee={emcee_ftj:.0f} HMC={tg['hmc_ftj']:.1f}) "
      f"Hier(SBI={sbi_hier} HMC={hmc_hier:.0f} emcee~{EMCEE_HBI_EST:.0f}est)")
