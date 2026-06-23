# CL-SBI Lab Notebook

High-level log of key decisions and product updates. Newest entries at top.
Detailed per-session TODOs live in `CLAUDE.md` and `tex_source/REVIEW_TODOS.md`.

---

## 2026-06-22 (eve) — Decision: fully-amortized neural HBI (drop percentiles), staged A-smoke → A-full

**Question that started it (user):** before extending Paper 2 results to all 8 experiments, is the
hierarchical SBI as powerful as it could be? Why not vary the M-c / R-M relations, noise profiles,
etc. and marginalize over them — train on a broader set of sims to dissolve the OOD-inference
argument (Fig 3, where the fixed `infer_z1` percentile net collapses to σ_c≈0.19 vs true 0.72)?

**Reframe (key realization):** the percentile vector (7 logM + 7 c quantiles + correlation) was
always a *workaround* for Paper 1 having no hierarchy — it smuggles population spread into a
fixed-length regression target, which is exactly why it OODs. Broadening *its* training prior just
builds a smarter workaround. Proper HBI **infers** the population hyperparameters, so 4 of the 5 OOD
axes (mc_relation→c0,β; mc_scatter→σ_c; rm_scatter→σ_M; noise→σ_extra) are handled by the model **by
construction, not training breadth** — this is structurally why explicit HMC already recovered
σ_c=0.795 on high-mc-scatter while the percentile net collapsed. Only contamination needs a real
model change (2-component mass mixture; it pushes the mass dist low/bimodal but barely touches σ_c).

**Architecture decision (user):** go for the gold version — **A, fully-amortized "neural HBI"**:
train `q(μ_M,σ_M,c0,β,σ_c,[nuisance] | full stack of N_c profiles)` with hyperparameters AND
nuisance drawn from broad hyperpriors per training stack. One forward pass → population posterior,
**zero per-dataset MCMC.** Here "vary all the params during training" *is* the marginalization
mechanism — the literal answer to the user's question.

**Dropped the intermediate (B = amortized per-cluster likelihood + explicit hierarchy).** Initially
planned B→A as de-risking, but B shares NONE of A's hard parts (permutation-invariant stack embedding;
calibration over a high-dim hyperparameter posterior) and the parts it shares (forward model,
population model) are already validated by the working HMC-HBI. HMC-HBI is also already the
published-grade fallback if A fails. So B was a slower incremental result, not real de-risking.

**Plan: staged A-smoke → A-full.** De-risk A *on its own terms*:
- **A-smoke (~½ day, go/no-go gate):** narrow priors, in-distribution only, small stacks (N_c≈50),
  mean-pool embedding → SNPE on the 5 hyperparameters, **run SBC**. One question: does it calibrate on
  easy data? Uniform rank histograms → proceed. If not → fall back to HMC, ½ day spent.
- **A-full (overnight commit AFTER the gate):** broad hyperpriors + all nuisance axes (continuous
  (c0,β) box spanning child/ludlow/prada, rm_scatter, noise_dex~U(0.1,0.8), f_contam mixture), full
  N_c=376, SBC over the full envelope, collate vs truth/HMC/percentile-SBI. Training sims ~18 hr
  (colossus-dominated, measured 17 ms/mc-pair this session).

**Scope guard (unchanged):** the σ_c noise floor still binds — at 0.3 dex, per-cluster c is
constrained to ±0.9 (~5× true spread), so σ_c≈0.18 is unrecoverable for ANY method. high-mc-scatter
(σ_c=0.72) is the one discriminating case above the floor. A's win is calibrated amortized population
inference over the nuisance envelope, NOT beating the information limit.

**Products this session:** `notebooks/AMORTIZED_HBI_SCOPE.md` (full scoping doc, numbers measured
from the `.376` obs sets), a `\prelim`-tagged preliminary methodology subsection in Paper 2
(`sec:neural_hbi`), and the A-smoke script `notebooks/asmoke_neural_hbi.py` (launched). Open
decisions: contam mixture in A-full (default: include); embedding net mean-pool vs deep-set (start
mean-pool); mc_relation continuous-box vs one-hot (prefer box).

**Background runs at session start:** emcee-HBI baseline (PID 42868) still alive at ~1h54m wall /
12.7% CPU but **still has not written its first checkpoint** — suspect it is effectively stalled
again (cf. the 2026-06-22 am dead run). The ≥1-day timing estimate stands on the measured per-step
cost regardless; consider killing if no checkpoint appears.

**A-smoke RESULT — GATE: GO (with a watch-item).** Ran in **3 min**, not the budgeted ½ day (small
N_c=50 stacks sim at ~6 ms each; 4000 stacks in ~25s, SNPE+deep-set train ~30s, SBC 300×1000 ~35s).
- **Architecture works.** Deep-set embedding is permutation-invariant to machine precision (1.5e-7).
  Point recovery on a baseline-truth stack is near-perfect on one forward pass, no MCMC: σ_M=0.119
  (true 0.120), μ_M=14.380 (14.400), c0=4.597 (4.600), β=−0.839 (−0.850), σ_c=0.188 (0.180).
- **SBC (300 held-out), KS-from-uniform:** μ_M 0.066, σ_M 0.053, c0 0.059 → clean PASS; **β 0.114,
  σ_c 0.115 → CHECK.** Max KS 0.115 < 0.12 gate → GO.
- **Diagnosis of the two CHECK dims (from rank histograms, `asmoke_sbc_ranks.png`):** β and σ_c show
  a clear **U-shape** (edge-bin frac 0.36/0.37 vs ideal 0.20; center frac 0.16 vs 0.20) = mild
  **over-confidence** (posteriors slightly too narrow, truth lands in tails too often). This is the
  *same weakly-identified (β, σ_c) pair* the σ_c noise floor squeezes — expected, not a bug. Most
  likely fixed by A-full's much larger sim budget (4000 stacks is thin for a 5-D amortized posterior
  + embedding); if not, the documented lever is mean-pool → deep-set-with-sum / attention.
- **Verdict:** GO to A-full, but **β and σ_c are the dimensions to scrutinize in A-full's SBC.** Do
  NOT claim full calibration yet. HMC-HBI remains the fallback.

**Products:** `notebooks/asmoke_neural_hbi.py` (deep-set + SNPE + SBC harness; runs under the `base`
conda env which has the sbi 0.21 / torch 2.3 / jax 0.4.38 / numpyro 0.13.2 stack — NOT the default
`python3`), `asmoke_neural_hbi.pkl` (ranks + KS + samples), `asmoke_sbc_ranks.png` (the rank
histograms). **A-full is now a justified commit, not a blind 18-hr gamble — but held pending review
of the rank histograms before greenlighting the overnight sim run.**

**N_TRAIN sweep (`asmoke_ntrain_sweep.py`) — falsifies the under-training hypothesis for beta.**
Retrained at N_TRAIN = 4k/8k/16k (shared cached sim pool, same SBC seed). KS-from-uniform:

| N_TRAIN | mu_M | sig_M | c0 | beta | sig_c |
|---|---|---|---|---|---|
| 4000 | 0.030 | 0.045 | 0.046 | 0.096 | 0.097 |
| 8000 | 0.061 | 0.051 | 0.050 | 0.109 | 0.059 |
| 16000 | 0.035 | 0.044 | 0.076 | 0.109 | 0.083 |

- **sig_M rock-solid (~0.044) at all N_TRAIN** — the headline mass-spread parameter is secure.
- **sig_c flat ~0.06–0.10, hovering at the pass line** — consistent with noise-floor physics, not
  under-training.
- **beta did NOT improve with 4× sims (0.096→0.109, flat).** Under-training hypothesis FALSIFIED for
  beta. (Caveat: at N_SBC=300 the α=0.01 KS critical value ≈0.094, so 0.10–0.11 is right at the
  edge of distinguishable-from-calibrated — beta is *mildly* over-confident, not broken.)
- **Implication:** the lever for beta is **architecture, not sim count** — so 18 hr of A-full sims
  would NOT fix it. beta is the M–c *slope*, identified by how c co-varies with M *across* clusters;
  **mean-pooling averages clusters together and washes out exactly that inter-cluster signal.**

**Architecture probe (`asmoke_arch_probe.py`) — DONE, hypothesis CONFIRMED.** Retrained 4 aggregator
variants on the 16k pool (same SBC seed; `mean` reproduces the sweep's 16k row exactly → controlled).
KS-from-uniform:

| variant | mu_M | sig_M | c0 | beta | sig_c |
|---|---|---|---|---|---|
| mean (baseline) | 0.035 | 0.044 | 0.076 | **0.109** | 0.083 |
| sum | 0.042 | 0.104 | 0.049 | 0.107 | 0.085 |
| **moment** | 0.092 | 0.053 | 0.053 | **0.078** | 0.043 |
| moment_big | 0.047 | 0.047 | 0.080 | 0.120 | 0.079 |

- **WINNER: `moment` pooling** (concat [mean, std] over clusters). beta 0.109→**0.078** (passes) and
  sig_c 0.083→**0.043** (best anywhere). Confirms the mechanism: beta/sig_c are covariance/spread
  params; the **std channel gives the net the inter-cluster spread that identifies them**, which
  mean-pool discarded. Only variant with all 5 dims ≤ pass line (max 0.092 on mu_M, within N_SBC=300
  noise).
- **Controls behaved informatively:** `sum` didn't fix beta AND broke sig_M (0.044→0.104) — sum
  scales with N_c, conflating cluster count with population width. `moment_big` was *worse* (beta
  0.120) — more capacity w/o the right inductive bias overfits; rules out "just use a bigger net."
- **DECISION: lock `moment` pooling (concat mean+std) into A-full.** A-full now goes in with an
  evidence-based, fully-calibrated architecture rather than a guess — bought for ~25 min on the
  cached pool vs an 18-hr commit. `asmoke_arch_probe.pkl` saved.

**Next: greenlight A-full** — broaden priors to full envelope (logM↓12, σ_c↑0.74, continuous c0/β
spanning child/ludlow/prada), add nuisance axes (rm_scatter, noise_dex~U(0.1,0.8), f_contam mixture),
N_c=376, moment-pool embedding, SBC over full envelope + apply to all 8 obs stacks. ~18 hr sims.

---

## 2026-06-22 (pm) — Env restore, Fig-3 OOD diagnosis, paper2-hbi branch

**Environment untangled.** The numpyro 0.13→0.21 upgrade (done for `render_model` PGMs) had pulled
numpy 2.4.6 + jax 0.10.2, which broke the SBI/TensorFlow stack (`np.complex_`/`np.string_`/
`np.dtypes.StringDType` removed in numpy 2). Diagnosed as mutually-exclusive pins: jax 0.10 *requires*
numpy≥2, sbi/TF *requires* numpy<2. Fix: rolled the whole stack back to the pre-PGM-upgrade state —
**numpy 1.26.4 + jax 0.4.38 + jaxlib 0.4.38 + numpyro 0.13.2** (sbi stays 0.21, torch 2.3.0). PGMs
unaffected (already rendered to PNG; `render_model` no longer needed). Verified SBI import + jax/halox
forward model + numpyro HBI all coexist and run. **Did NOT upgrade sbi** (would break `sbi_.py`'s
`SNPE` API — renamed to `NPE` in sbi 0.23+ — and wouldn't fix the root numpy-2/TF conflict anyway).

**Fig 3 high-M-c-scatter "discrepancy" resolved — NOT a bug, a design choice.** User noticed Paper 2
Fig 3 (aggregate_hbi_comparison) shows SBI FTJ *underestimating* σ_c on high-M-c-scatter (0.187 vs
true 0.719), whereas Paper 1 showed SBI nailing it. Cause: the two figures use different inferrers.
Paper 1 used the **matched** `infer_z1_high_mc_scatter` inferrer (trained on σ_c≈0.72, in-distribution
→ recovers 0.721). Paper 2 Fig 3 holds the **baseline** `infer_z1` inferrer fixed across ALL
experiments (the apples-to-apples stress-test design); for high-M-c-scatter that's OOD (trained on
σ_c≈0.18, never saw 0.72), so SBI collapses to ~0.19. HMC (likelihood-based, training-free) recovers
0.795 regardless — which is the panel's whole point: HBI captures population spread that
fixed-distribution SBI misses OOD. Mass spread is fine for all (SBI 0.111, HMC 0.116 vs true 0.118).
**TODO (offered, not yet applied): add one `\akum{}` caption sentence to Fig 3 noting all SBI panels
use the single baseline `infer_z1` inferrer, so its high-scatter underperformance is an OOD effect,
not a contradiction of Paper 1.**

**Discussed "train SBI over a distribution of M-c / λ-M relations + noise profiles to remove priors."**
Verdict: sound but it's marginalization-over-models, not prior-free — replaces a parameter prior with
a hyper-prior over relations (still a modeling choice), trades constraining power, and partly
*confounds the σ_c population-spread measurement* (observed scatter ↔ which relation generated it).
It's the amortized twin of HBI → belongs as a discussion paragraph in "Toward hierarchical SBI."
Optional cheap demo: retrain one SBI on a broad-σ_c mix, show it recovers σ_c where baseline collapsed
(~half-day). NOT launched; Paper 2 is near-complete (16 pp).

**Committed to new branch `paper2-hbi`** (off `tobemerged`). 62 files: tex_source_hbi/ (paper text,
bib, generated table — figures omitted per repo convention: Paper 1 tracks 0 figures, all regenerable
from scripts + Overleaf), all hierarchical notebooks, Paper-1 code changes (adaptive mcmc.py, KS
methodology, prior fix), regen scripts, this notebook. ⚠️ **Caveat: the commit mixes Paper 1 code
changes with Paper 2** (they were intertwined uncommitted work) — split out to tobemerged/main later
if desired. Build artifacts (.aux/.bbl/.log) and plotutils_old.py left untracked.

**Background runs still live at session end:** emcee-HBI baseline (`run_emcee_hbi_baseline.py`, ~40min
in, checkpoints to `emcee_hbi_baseline_progress.pkl`) — the real-data-point for the emcee-HBI cost
claim. Amortized-SBI sample-reweighting (`amortized_hbi_samples.py`) is now UNBLOCKED by the env fix
but not yet re-run.

---

## 2026-06-22 (am) — Paper 2 figures, KS methodology, cost grid, PGMs

**KS test methodology fixed (Paper 1 Table 1).** The KS p-values were sample-size dependent
(pipeline used N=1000, "reduced for speed"); borderline cells were artifacts. Fix: evaluate KS at
N=N_c=376 (the real cluster count), median over 101 bootstrap subsamples; documented in
`plot_calibration.py` (`--ftj-only` avoids the slow JTF-truth path). Regenerated all 16 KS tables.
Net: SBI passes *more* broadly at the honest N (e.g. high-λ-M-scatter passes both). Reframed text
to lean on the sample-size-independent coverage analysis as primary; KS is now "indicative."

**Cost grid (Paper 2 timing figure).** Measured the full task×method grid (JTF/FTJ/Hierarchical ×
SBI/HMC/emcee). New numbers: HMC-FTJ ~6s vs emcee-FTJ ~2400s (HMC ~400× faster even at 3 params);
HMC-HBI ~210s; emcee-HBI estimated ≥1 day (measured ~1 s/step, ~1500 walkers, accept ~0.1). Timing
fig regrouped by task, bars sorted SBI<HMC<emcee. Paper 1 future-work now quantifies the emcee-HBI
cost as the explicit reason hierarchical inference was deferred.

**PGM figures.** Added probabilistic-graphical-model comparison via `numpyro.render_model` (faithful
to model code), with Unicode Greek + true subscripts (HTML-like graphviz labels). Split into two
figures: JTF+FTJ together, Hierarchical alone (legibility). Required upgrading numpyro 0.13→0.21
(pulled jax 0.4.38→0.10.2, numpy→2.4.6); verified halox forward model still works.

**Paper 2 figure restructuring (per AG review):** removed σ_M bar chart (redundant with table);
fixed Table 1 to show consistent spread (σ_logM, σ_c) + center (μ_logM, μ_c) blocks instead of the
odd σ_M/c-center mix; added 3-method calibration plot (baseline: MCMC Δ_max=0.93 overconfident, SBI
0.02, HMC 0.11); bumped all figure fonts (rcParams) for legibility; added `placeins` float barriers.
Fixed a self-inflicted `sed` bug that mangled `\textwidth`→tab in two `\includegraphics` (broke the
main aggregate results figure briefly).

**Dead emcee-HBI run.** The real-data-point emcee-HBI background run stalled (0% CPU, no checkpoints)
after the environment churn; killed it. The timing estimate stands on the measured per-step cost.

**In progress:** (1) calibration grid for all 8 experiments (2×4 small-multiples); (2) U-shaped
realistic-noise experiment (high inner+outer noise) — new pipeline test. Most results in Paper 2 are
still baseline-only (calibration, PPC, inferred-noise, amortized-SBI) — extending coverage is the
top scientific gap.

**Open cross-paper TODO:** Paper 1's future-work section (hierarchical modeling §; differentiable
models §) is deliberately left framed as "future work" with no companion-paper citation and a vague
"~order of magnitude faster" HMC estimate (decision 2026-06-20). When Paper 2 (HBI/HMC) is ready to
cite, revisit and add forward references + the measured speed numbers (HMC ~13× faster than
converged emcee, ~200× slower than amortized SBI). Currently consistent, just not cross-linked.

---

## 2026-06-18 — Follow-up paper (HBI/HMC) + adaptive MCMC + amortized hierarchical SBI

**Context:** Investigating whether MCMC can recover the population (M,c) *spread* (not just a
central value), which Paper I showed only SBI could do.

**Key decisions & findings:**
- **MCMC "overconfidence" is structural, not fundamental.** Paper I's MCMC fit-then-join (FTJ)
  uses a single-(M,c) joint likelihood, so its posterior estimates a central (≈mean) value, not
  the population spread. Reframed Paper I's text accordingly (`\agent{}` tags); softened the
  "~1/√N_c" claim to the empirically-measured "~11× tighter in mass."
- **Built a hierarchical Bayesian model (HBI) in NumPyro/NUTS** with a differentiable JAX forward
  model. Decision: use **halox** for NFW physics rather than hand-rolling (validated to <0.1% vs
  the Colossus data-generator); only re-implemented the analytic Σ(x) step to fix a NaN-gradient
  bug in halox's `jnp.where` branches ("safe-x" trick).
- **Result:** HBI recovers the true population σ_M to a few % across *all 16 experiments*
  (publication-grade: 4 chains, R̂<1.01). Matches SBI in-distribution; beats SBI on the
  high-λ-M-scatter case (0.52 vs true 0.54; SBI gets 0.21). ~31 min total sampling for 8 unique
  observation sets on one CPU.
- **Diagnosed a noise-induced concentration bias** (~+0.15) common to *all analytic-likelihood*
  methods (MCMC + HBI) but absent in SBI. Traced via a noiseless-vs-noisy ladder: it's the
  nonlinear NFW + Gaussian-likelihood + M–c anti-correlation interaction; SBI escapes it by
  learning the inverse map from noisy training pairs. Logged as a second SBI advantage.
- **Fixed a real prior bug** in `mcmcutils.logprior`: richness-selection scatter was applied in
  ln λ but generated in log₁₀M (was ~1.7× too tight). Added `populationutils.get_rm_slope`.
  Confirmed it is NOT the cause of the concentration bias (changed c by only +0.005).
- **Adaptive MCMC convergence checking** (replacing fixed walkers/steps/burnin): added
  autocorrelation-based stopping (`nsteps > 50·τ` + τ stable) with split-R̂ secondary check, in
  `mcmc.py`. Stamps `converged`/`tau_max`/`rhat_max` on each sampler. Trade-off accepted: it's
  slower (JTF ~310s vs old ~52s) because the old fixed runs were under-sampled. **Decision (user):
  run the full 16-config Paper I MCMC regen overnight with the new sampler.** (In progress.)
- **Amortized hierarchical SBI (option c)** POC: train one single-cluster SNPE → use as per-cluster
  likelihood surrogate inside the HBI model. Unifies both papers. Recovers μ_M, σ_M well in ~10s.
- **Concentration spread is fundamentally unrecoverable at 0.3 dex noise** (revised finding).
  Initially thought σ_c overestimate (0.47 vs true 0.18) was a diffuse-prior/slope-runaway artifact
  ("Fix B": informed Child18 β prior + tight σ_c prior). Tested it — **did not help** (σ_c stayed
  ~0.47 even with β pinned at Child18 −0.855). Root cause: per-cluster concentration is constrained
  only to ±0.9 (≈5× the true population spread), so population signal is <4% of observed variance —
  below the noise floor. Per-cluster SBI posteriors are well-calibrated (unit standardized
  residuals); it's an information limit, not a bug. **This is a publishable result**: stacked WL at
  this noise recovers the population mass distribution but NOT the concentration scatter, for any
  method (HMC, amortized SBI, moment deconvolution). Written into Paper 2 §amortized_hbi.
- **Paper 2 authors** set: A. Gill (Harvard + CfA), C. Avestruz, M.-F. Ho. Normalizing-flow
  per-cluster likelihood left as an explicit TODO (option c refinement + option d).
- **Normalizing-flow scope doc** (`notebooks/NORMALIZING_FLOW_SCOPE.md`): no JAX flow lib installed;
  options ranked (GMM surrogate = cheap/no-deps → flowjax = high-fidelity). Recommended staged:
  GMM first to test if non-Gaussian shape matters, then flowjax. Scope guard: the flow will NOT
  rescue σ_c (information limit), only helps non-Gaussian *population* cases.
- **Posterior predictive checks (PPC)** added to Paper 2 (Fig, all 8 experiments): forward-model
  population draws from the HMC posterior vs observed profiles. Two green bands — intrinsic
  population scatter (narrow) and full predictive with obs noise added (matches grey data envelope).
  Predicted distribution reproduces data in every experiment.
- **Jointly inferred per-bin measurement noise** (Paper 2 §infernoise + Fig): promoted the noise to
  30 free per-bin hyperparameters. KEY RESULT: σ_M still recovered (0.124 vs true 0.118) AND the
  per-bin noise recovered bin-by-bin (RMS residual 0.009 dex, mean 0.301 vs true 0.307). The radial
  shape separates white noise from coherent NFW population scatter. This joint population+noise
  inference is impossible for single-(M,c) MCMC or percentile-SBI — a concrete hierarchical
  advantage. Code: `hier_inferred_noise.py`.
- **Paper 1 MCMC regen COMPLETE** (adaptive sampler, corrected richness-scatter prior, 16 configs):
  **32/32 runs converged, 0 failures** (r̂≈1.01-1.02, 2500-6500 steps). The previously-pathological
  high-mc-scatter runaway (σ_M=1.77) is fixed → now a sensible 0.006. ~13 hr total sampling.
- **Three-method comparison re-collated** from fresh converged chains (`collate_three_methods.py`):
  MCMC now converges everywhere, with uniformly tiny σ_M (~0.006, the "estimates the mean"
  signature) — DNC markers dropped from 4 to 1 in Table 1. Regenerated σ_M figure, table, aggregate
  comparison, and three-method PPC.
- **Timing narrative strengthened (important):** converged adaptive emcee MCMC now costs ~2700s
  (JTF+FTJ) vs HMC ~210s → **HMC is ~13× FASTER than properly-converged ensemble MCMC** (was
  "comparable" when emcee was under-sampled at fixed 500 steps). Updated Fig + cost section.
- **Three-method PPC** (replaced single-method): per experiment, each method's population pushed
  through the forward model vs observed profiles, with fractional-residual panel. All reproduce the
  observed median; differ in spread + OOD residuals.
- **Paper 1 plots/tables regenerated** (all 16 configs, 0 errors): chains, diagnostics,
  calibration/KS all rebuilt from the converged chains.
- **Paper 1 KS table (`tab:ks_results`) updated** with regenerated values (all `\agent{}` tagged).
  Decisions: SBI p-values taken from the authoritative `plot_calibration.py` CSV; all MCMC entries
  set to `$<10^{-16}$` (verified by full-precision recompute: true MCMC p ~ 10^-73 or smaller, the
  CSV just truncates to 0.0000). **MCMC fails everywhere (story intact).** Notable SBI shifts vs the
  old under-sampled table: richness-contam c 0.30→0.05 (now borderline), high-λ-M-scatter M
  0.10→0.12 (still borderline pass), high-noise/obs-higher-noise c improved (0.09→0.91 for the
  latter). Also fixed two leftover "1/√N_c" claims (L473, L726) to "central (≈mean) value" to match
  the verified ~11× width finding — Paper 1 now internally consistent with Paper 2.
- **CAUTION noted:** KS p-values are sample-size dependent; always use the pipeline CSV, not ad-hoc
  recomputes (a fresh-draw recompute gave different numbers — do not mix methodologies in the table).

**Products:**
- New follow-up paper `tex_source_hbi/hbi_paper.tex` (9 pp, compiles clean). Authors: A. Gill
  (Harvard + CfA), C. Avestruz, M.-F. Ho. Figures: σ_M-recovery bars, aggregate 3-method
  comparison (Paper-I Fig-12 style), timing; Table 1 with bold=good / †=>2×-off / DNC markers.
  Sections on differentiable forward model, HBI, noise bias, cost, amortized hierarchical SBI,
  "Toward hierarchical SBI" discussion.
- New notebooks: `hierarchical_mcmc_poc.ipynb` (executed, with 2D corner + plane plots),
  `run_hier_all_experiments.py`, `make_three_method_summary.py`, `make_aggregate_hbi_figure.py`,
  `amortized_hier_sbi_poc.py`. Outputs in `notebooks/hierarchical_poc_outputs/`.
- `scripts/regen_paper1_mcmc.sh` — full 16-config regen with adaptive sampler.

---

## 2026-06-18 — Paper I automated referee loop + revisions

- Ran a 2-agent reviewer↔author loop (expert referee + author) to ApJ/OJA standard.
  **Outcome: accept with minor revisions** after 2 rounds; all critical/major issues resolved.
- Critical fixes verified against code: FTJ likelihood text now matches the plain-sum
  implementation; the `ln f` error-inflation nuisance parameter documented; SBI-vs-MCMC fairness
  argument added (training dist ≡ informed prior). Referee judged the written fairness argument
  sufficient — the SBI-on-MCMC-prior control experiment is **not** required (deferred to future
  work).
- Applied 5 minor corrections (JTF/FTJ terminology swap, log-normal bias 2.6×→3.7×, calibration
  geometry label, richness-selection approximation note, speed claim ~200–800× from the CSV).
- All changes marked with new orange `\agent{}` tag (distinct from cyan `\akum{}`). Referee TODOs
  preserved in `tex_source/REVIEW_TODOS.md`.

---

## 2026-06-17 — Paper I figure regeneration + reviewer-feedback pass

- Regenerated all figures with ChainConsumer 1.x (legend positioning fixes in `plotutils.py`).
- Changed JTF-vs-FTJ plot palette to higher-contrast blue/red; updated text references.
- Addressed Round-0 (Anthony) and MF_CA (Ming-Feng/Camille) feedback with `\akum{}`-tagged edits:
  z~1 + LSST Science Book citation, baryonic-effects + Pratt2019, M200m terminology,
  shear-vs-profile stacking note, FTJ baseline config (Nc=376, 30<λ<45, 0.2<z<0.35),
  hierarchical-modeling citations (Mantz2015, Grandis2021), pruned redundant JTF/FTJ definitions,
  moved differentiable-models text to future work.
- Verified the full clean LaTeX build cycle (stale aux files were silently breaking citations).

---

## Earlier (pre-session, from git history) — Paper I development

- Core pipeline: gen_simulations → gen_observations → train_inferrer → run_inference → plots.
- Two aggregation approaches established: join-then-fit (stack then fit) and fit-then-join
  (joint likelihood / SBI percentile inference).
- All observables moved to log-space; MCMC FTJ reworked to a joint likelihood; SBI FTJ infers
  7 percentiles + correlation coefficient; added richness-bin selection function with mass
  function, M-c and λ-M scatter experiments, richness contamination ("leak"), calibration plots,
  and the informed MCMC prior (mass function + richness selection) to match SBI.
