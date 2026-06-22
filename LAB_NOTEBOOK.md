# CL-SBI Lab Notebook

High-level log of key decisions and product updates. Newest entries at top.
Detailed per-session TODOs live in `CLAUDE.md` and `tex_source/REVIEW_TODOS.md`.

---

## 2026-06-22 — Paper 2 figures, KS methodology, cost grid, PGMs

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
