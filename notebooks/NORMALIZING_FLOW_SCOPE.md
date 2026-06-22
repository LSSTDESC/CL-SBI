# Scoping: normalizing-flow per-cluster likelihood for amortized hierarchical SBI

## Goal

Replace the per-cluster **Gaussian summary** of the SBI posterior with a **normalizing-flow
density** that preserves the true non-Gaussian shape (the curved M–c degeneracy banana and any
skew), inside the hierarchical population model.

**Important scope note (from the σ_c finding):** this will NOT make the concentration population
spread recoverable at 0.3 dex noise — that is an information limit, not a shape problem. The flow
matters for: (a) the **non-Gaussian-population experiments** (high λ-M scatter), where the Gaussian
*summary* genuinely loses information; (b) **methodological correctness/honesty** (using the full
SBI posterior, not a lossy 2-moment compression); and (c) being the clean foundation for the
lower-noise regime where concentration *does* become identifiable.

## The core problem

`sbi` trains its density estimator in **PyTorch**. The hierarchical sampler (NUTS) needs a
**JAX** log-density with gradients. The per-cluster likelihood appears inside a `plate` over 376
clusters and is evaluated thousands of times per chain. So we need the flow's `log_prob(theta | d_j)`
callable from JAX, differentiably, for all 376 conditioning data vectors.

## Environment audit (what we have / lack)

| Need | Status |
|------|--------|
| JAX 0.4.38 | available |
| NumPyro 0.13.2 | available (NUTS, BNAF transform exists) |
| sbi 0.21.0 (PyTorch SNPE/flows) | available — but PyTorch, not JAX |
| flowjax / distrax / TFP / flax / optax / equinox | **all MISSING** |
| numpyro spline/MAF flow transforms | **not shipped** in this version |

Implication: there is **no ready-made JAX normalizing-flow library installed**. We either install
one, hand-roll a small flow, or avoid a flow at the sampling step. This drives the options below.

## Options (in increasing fidelity / effort)

### Option 1 — Mixture-of-Gaussians per-cluster surrogate  (LOW effort, no new deps)
- Fit a small K-component GMM (e.g. K=3–5) to each per-cluster SBI sample set instead of a single
  Gaussian. Its log-prob is a closed-form `logsumexp` of Gaussians — trivially JAX-differentiable,
  no new library.
- Captures multi-modality and approximates curved degeneracies; far better than 1 Gaussian.
- **Cost:** ~1 day. Fit 376 small GMMs (sklearn, one-time), store means/covs/weights, evaluate the
  mixture log-prob in the existing NumPyro model.
- **Limitation:** still not the *exact* SBI density; a smooth banana needs several components.

### Option 2 — Hand-rolled conditional MAF/spline flow in JAX/NumPyro  (MEDIUM effort)
- Train a single **conditional** flow q(M,c | d) directly in JAX, conditioned on the data vector,
  using NumPyro's autoregressive transforms (BNAF is available) or a hand-built rational-quadratic
  spline coupling layer. One amortized flow, evaluated per cluster by conditioning on d_j.
- Fully native: log-prob and gradients are JAX, drops straight into the `plate`.
- **Cost:** ~3–5 days. Implementing/validating a conditional spline flow + its training loop in
  NumPyro is the bulk; needs `optax` (install) or NumPyro's SVI for training.
- **Cleanest scientifically:** one coherent amortized JAX density, no torch↔jax bridge.

### Option 3 — Reuse the trained PyTorch sbi flow via a torch↔JAX bridge  (MEDIUM/risky)
- Keep sbi's trained flow; expose its `log_prob` to JAX. Either (a) `jax.pure_callback` into torch
  (works but **kills gradient flow** → forces gradient-free sampling; defeats HMC), or
  (b) re-export the flow weights and re-implement its forward pass in JAX (fragile, version-coupled
  to sbi's internal nflows architecture).
- **Cost:** ~2–4 days; (a) is quick but loses HMC's main advantage, (b) is brittle.
- **Not recommended** unless we specifically want to reuse the exact trained net.

### Option 4 — Install flowjax and use it natively  (LOW–MEDIUM, adds dependency)
- `flowjax` is a mature JAX normalizing-flow library (built on equinox) with conditional flows and
  a clean `log_prob`. Would make Option 2 much faster to implement.
- **Cost:** add deps (flowjax + equinox + optax), then ~2 days. Train one conditional flow
  q(M,c|d); use `flowjax`'s log_prob inside NumPyro via a custom factor.
- **Best effort/fidelity trade-off** if adding a dependency is acceptable.

## Recommendation

**Staged:** start with **Option 1 (mixture-of-Gaussians)** as a fast, dependency-free upgrade that
already tests "does preserving non-Gaussian per-cluster shape change the high-λ-M-scatter result?"
If that shows the shape matters, graduate to **Option 4 (flowjax)** for the publication-grade
conditional flow. Skip Option 3 (bridge) — losing differentiability defeats the purpose.

**Decision needed:** is adding a JAX flow dependency (`flowjax`/`equinox`/`optax`) acceptable for
this repo, or should we stay dependency-free (Option 1 only)?

## Validation plan (any option)

1. Per-cluster sanity: flow log-prob reproduces sbi posterior samples (KS / coverage on held-out).
2. Re-run the baseline amortized-HBI: μ_M, σ_M should match the Gaussian-summary result
   (sanity — mass is already fine).
3. The real test: **high-λ-M-scatter experiment**, where the population is non-Gaussian. Compare
   recovered population vs Gaussian-summary vs HMC vs truth.
4. Confirm σ_c is still ~unrecoverable at 0.3 dex (expected; the flow doesn't add information),
   and optionally sweep noise down to find where σ_c becomes identifiable (the TODO in the paper).

## Effort summary

| Option | New deps | Effort | Fidelity | HMC-compatible |
|--------|----------|--------|----------|----------------|
| 1 GMM  | none | ~1 day | medium | yes |
| 2 hand-rolled JAX flow | optax | ~3–5 d | high | yes |
| 3 torch bridge | none | ~2–4 d | exact (trained net) | no (loses grads) |
| 4 flowjax | flowjax+equinox+optax | ~2 d | high | yes |
