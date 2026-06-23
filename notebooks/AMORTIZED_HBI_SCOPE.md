# Scoping: fully-amortized neural hierarchical SBI (staged A-smoke → A-full)

## Goal

Replace the **percentile-vector SBI** (7 logM + 7 c quantiles + correlation = 15-dim regression
target) with a **fully-amortized neural hierarchical model**, robust across all 8 stress-test
experiments — including the OOD cases where the fixed-distribution percentile net collapses
(Paper 2 Fig 3, high-M-c-scatter: percentile SBI → σ_c ≈ 0.19 vs true 0.72).

**The architecture — A ("neural HBI").** Train `q(μ_M, σ_M, c0, β, σ_c, [nuisance] | full stack)`,
drawing hyperparameters *and* nuisance params from broad hyperpriors per training stack. One forward
pass on the real stack → population posterior, **zero per-dataset MCMC.** Here "vary all the params
during training" IS the marginalization mechanism — the direct answer to the original question.

**Decision: go A directly, drop the per-cluster intermediate (B).** An earlier plan staged a
per-cluster amortized likelihood + explicit NumPyro hierarchy (B) first, on the theory that it would
de-risk A. That theory is weak: B shares *none* of A's hard, novel parts (the permutation-invariant
stack embedding and calibration over a high-dim hyperparameter posterior), and the parts it *does*
share (differentiable forward model, population model) are **already validated by the working
explicit HMC-HBI**. And HMC-HBI already serves as the robust published-grade fallback if A fails. So
B's marginal value is a slower incremental result, not real de-risking. Instead we de-risk A *on its
own terms* with a cheap smoke test (see "Staging" below).

## Why percentiles were always a workaround

The 15-dim percentile target exists because Paper 1's amortized SBI had **no hierarchy** — it needed
a fixed-length summary that smuggled population spread into a regression target. That is exactly why
it OODs: the net learned to read off a σ_c it had only ever seen at ≈0.18. Broadening *its* training
prior would only produce a smarter workaround. Proper HBI infers the population hyperparameters
directly, so the OOD axes are handled by the model, not by training breadth.

## Why proper HBI does not OOD the way the percentile net does

Map each OOD axis to what it physically changes, and to the hyperparameter that absorbs it:

| OOD axis | In-distro → OOD | Physically changes | Handled by |
|----------|-----------------|--------------------|------------|
| `mc_relation` | child18 → ludlow16 / prada12 | mean c(M) normalization + slope | **c0, β** (inferred) |
| `mc_scatter` | 0.1 → 0.5 / 0.7 | concentration scatter | **sig_c** (inferred) |
| `rm_scatter` | 0.1 → 0.5 / 0.7 | within-bin mass spread | **sig_M** (inferred) |
| `profile_noise_dex` | 0.3 → 0.1 / 0.5 / 0.7 | per-bin measurement noise | **sig_extra** / net conditioning |
| `richness_contam` | 0 → 0.1 / 0.5 | non-Gaussian / bimodal mass dist | **needs mixture component** |

For 4 of 5 axes the hierarchy is robust **by construction**: switching child18 → ludlow16 is not
"out of distribution," it just lands at a different inferred (c0, β). This is structurally why
explicit HMC already recovered σ_c = 0.795 on high-M-c-scatter while the percentile net collapsed.

## What A's training distribution must cover

The hyperpriors and nuisance ranges A trains over must envelope every experiment, or A is OOD in the
same way the percentile net is — the marginalization only protects you *inside* the trained envelope.
Per-experiment true ranges (measured from the `.376` obs sets) pin down the required coverage:

| experiment | logM range | c range | true σ_c | mean σ_obs |
|------------|-----------|---------|----------|-----------|
| baseline | [14.05, 14.71] | [4.23, 5.15] | 0.178 | 0.307 |
| high_mc_scatter | [14.05, 14.71] | [2.83, 6.77] | **0.719** | 0.311 |
| high_rm_scatter | **[12.25, 14.62]** | [4.15, **7.77**] | 0.739 | **0.450** |
| high_noise | [14.05, 14.71] | [4.23, 5.15] | 0.178 | **0.703** |
| ludlow | [14.05, 14.71] | [4.74, 5.85] | 0.219 | 0.307 |
| prada | [14.05, 14.71] | [5.84, 6.54] | 0.137 | 0.308 |
| low_richness_contam | [13.98, 14.71] | [4.21, 5.24] | 0.186 | 0.310 |
| high_richness_contam | **[13.78, 14.71]** | [4.29, 5.47] | 0.212 | 0.314 |

**Coverage 1 — per-cluster (M, c) support.** Latent clusters must be drawable down to logM ≈ 12.0
(`high_rm_scatter` reaches 12.25) and c up to ≈ 8 (reaches 7.77). The per-stack generative draw must
span logM ∈ [12.0, 15.2], c ∈ [2.0, 8.0].

**Coverage 2 — noise as a varied nuisance.** Stacks must be generated with noise_dex spanning every
regime: baseline 0.30, high_rm 0.45, high_noise 0.70. Draw noise_dex ~ U(0.1, 0.8) per training
stack; the real σ̄ is fed to A's embedding as a known per-stack input (it is measured, not inferred).

**Coverage 3 — hyperparameter ranges.** σ_c hyperprior must reach ≥ 0.74 (high_mc/high_rm), σ_M wide
enough for high_rm's broad mass spread, and (c0, β) must span the child18 → ludlow16 → prada12
envelope (prada pushes mean c to ≈ 6.2). mc_relation is a **discrete nuisance** — either train on a
continuous (c0, β) box that contains all three relations (cleaner — A never sees the relation label,
just its (c0, β) consequence), or one-hot encode it. Continuous-box is preferred.

**Contamination — a population-model-shape requirement, not just coverage.** `low/high_richness_contam`
push the mass distribution low (logM → 13.78) and make it skewed/bimodal — a single Gaussian
(μ_M, σ_M) cannot represent it. A must therefore train with a **2-component mixture** in the
population *mass* model (main bin + contaminating lower-richness bin, mixing fraction f_contam as an
extra hyperparameter/nuisance). Contamination barely touches σ_c (0.19 / 0.21 vs baseline 0.18), so
it is purely a mass-population-shape issue. **Open decision:** include the mixture in A's training, or
report the two contam cases as a known limitation and exclude them from the headline grid.

## Scope guard: the σ_c noise floor still binds

At 0.3 dex noise, per-cluster concentration is constrained only to ~±0.9 (≈5× the true population
spread), so the population σ_c signal is < 4% of observed variance — below the noise floor. **No
architecture** (percentile, amortized per-cluster, fully-amortized, or explicit HMC) recovers σ_c at
σ_c ≈ 0.18 and 0.3 dex. The one experiment where σ_c rises above the floor is **high_mc_scatter**
(true σ_c = 0.72): it is the discriminating case, and the one where A/HMC should beat the percentile
net's collapse. Everything at σ_c ≈ 0.18 stays unrecoverable — state this plainly, do not claim
otherwise.

## Staging: A-smoke → A-full

B shares none of A's hard parts, so it cannot de-risk them. The two genuinely risky, novel pieces of
A are (1) the **permutation-invariant embedding net** over N_c profiles and (2) **certifying
calibration** over a high-dim hyperparameter posterior (A can be point-accurate yet miscalibrated, and
unlike per-dataset NUTS there is no R̂ to tell you). De-risk *those*, cheaply, before the
full-scale training commit.

### Stage 1 — A-smoke (~half day, the go/no-go gate)

Smallest experiment that exercises A's hard parts on easy data:
- **Narrow** hyperpriors, **in-distribution only** (baseline regime, no nuisance variation).
- Small stacks (N_c ≈ 50) so training sims are cheap.
- Mean-pool **embedding net** (simplest permutation-invariant choice) → SNPE on the 5 hyperparameters.
- **Run SBC** (simulation-based calibration) on held-out stacks.
- **Single question:** does embedding-net + SNPE recover (μ_M, σ_M, c0, β, σ_c) with calibrated
  coverage on easy data? Uniform rank histograms → architecture works, proceed. If not → half a day
  spent, fall back to the existing HMC-HBI result (already in the paper).

### Stage 2 — A-full (after the gate passes)

- Broad hyperpriors + **all nuisance axes** (continuous (c0, β) box spanning child/ludlow/prada;
  rm_scatter; noise_dex ~ U(0.1, 0.8); f_contam with the 2-component mass mixture).
- Full N_c = 376 stacks; coverage ranges from the table above.
- Train, then **SBC over the full envelope** + apply to all 8 real obs stacks.
- **Collate vs. truth + vs. HMC + vs. percentile-SBI:** does A match explicit HMC across the grid,
  and where does it beat the percentile net's OOD collapse (the Fig-3 high-mc-scatter case)?

## Cost context (measured)

| method | per-dataset cost | notes |
|--------|-----------------|-------|
| percentile SBI (current) | ~1 s | OODs on relation/scatter shifts |
| explicit HMC-HBI | ~210 s | training-free, robust, per-dataset; the **fallback** |
| emcee-HBI | ≥ 1 day | 758-dim, affine-invariant; the cost data-point |
| **A (fully amortized)** | **~1 forward pass** | one-time training; zero per-dataset MCMC |

Forward-model timing (measured this session): `gen_mc_pairs` ≈ 17 ms/pair (colossus concentration +
mass function dominate), `simulate_nfw` ≈ 0.75 ms/profile. **A-smoke** trains on small stacks
(N_c ≈ 50), so its sim cost is minutes. **A-full** trains on full stacks: ~10k stacks × 376 ≈ 3.76M
mc-pairs ≈ **~18 hr** of simulation (colossus-dominated; parallelizable / cacheable) — an overnight
commit made only *after* the A-smoke gate passes.

## Scope guard restated for A

The σ_c noise floor still binds: A cannot manufacture concentration information at 0.3 dex where
σ_c ≈ 0.18. high_mc_scatter (σ_c = 0.72) is the discriminating case. A's win is calibrated, amortized
population inference over the whole nuisance envelope — *not* beating the information limit.

## Decisions

- [x] Architecture: **A directly**, staged A-smoke → A-full. Drop B (no shared de-risking; HMC-HBI is
      the fallback if A miscalibrates).
- [x] Noise handling: noise_dex a **varied per-stack nuisance**, fed to A's embedding as a known input.
- [ ] Contam: include 2-component mass mixture in A-full, or report contam as a known limitation and
      exclude from the headline grid? (default: include mixture)
- [x] Embedding net: **moment-pool (concat [mean, std] over clusters).** A-smoke + the arch probe
      showed mean-pool leaves the slope β (and σ_c) mildly over-confident (SBC KS≈0.11, flat in
      N_TRAIN); adding the per-cluster std channel drops β to 0.078 and σ_c to 0.043 (both pass),
      because β/σ_c are covariance/spread parameters that need the inter-cluster spread mean-pool
      discards. sum-pool broke σ_M; bigger-net did not help → it is the aggregator, not capacity.
- [ ] mc_relation encoding: continuous (c0, β) box (preferred) vs. one-hot label.
