# Run 1: first real traces (math500, R1-Distill-Qwen-1.5B)

Kaggle T4, free tier, 2026-10-28. Raw outputs are in `results/cloud-math500/` (gitignored). The headroom
numbers below come from re-running the analysis offline with the regularized bounds
(`d363b39`). That output is in `results/cloud-math500/rerun-r10/`.

## Setup

| Setting | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, float16 |
| Problems | 97 of 100 math500 problems collected; collection took 8.3 h |
| Steps | paragraph segmentation, max 16 steps, 2048 thinking tokens; mean horizon 16.4 states |
| Forced answers | 2 samples per state |
| Siblings | 2 per transition, every 4th transition (380 of 1491 transitions) |

## Headroom (held-out, 5 problem-grouped folds)

Each value is accuracy − λ·tokens/1000, averaged over traces.

| Policy | λ = 0 | λ = 0.05 | λ = 0.1 |
|---|---:|---:|---:|
| Hindsight prophet (oracle) | 0.830 | 0.801 | 0.772 |
| **Selected upper bound** | **0.830** | **0.801** | **0.772** |
| Full trace | 0.763 | 0.680 | 0.596 |
| LSM (cross-fit, margin-tuned) | 0.763 | 0.680 | 0.592 |
| Correctness probe (cross-fit) | 0.722 | 0.666 | 0.606 |
| Answer convergence (best k, cross-fit) | 0.629 | 0.586 | 0.543 |

At λ = 0, the hindsight prophet reaches 83.0% accuracy with 583 tokens. Full traces reach 76.3% with
1667 tokens. Paired headroom (upper bound − LSM) is 0.067, with 95% CI [0.031, 0.113].

### What the bounds show

- **The bounds are not tighter than hindsight yet.** With 2 siblings at every 4th step, the
  martingale corrections are mostly noise. Inner cross-validation chose shrink factor 0 in every
  fold, and the L1-penalized pathwise LP also reverted to the zero martingale. Before
  regularization, the same data gave meaningless bounds of 0.98 (LSM dual) and 1.41 (pathwise).
  The regularized estimators fall back to the hindsight value, which is itself a valid bound. The
  foresight gap (hindsight − V*) is therefore unmeasured, not zero.
- **No implementable policy beats finishing at λ = 0.** LSM selects margin = ∞ (never stop early) in
  every fold. Before the margin fix, it scored 0.598.

## Where the headroom comes from

- **Correct-then-wrong flips are rare but drive most of the gap.** 10 traces end below their best
  state. The 5 flips (best ≥ 0.5, final < 0.5) contribute 4.0 of the 6.5 trace-units of the
  λ = 0 hindsight gap. The other 5 traces go from 0.5 to 1.0 and contribute 2.5. Five flip events
  are too few to power the Idea 2 attention pilot.
- **Most of the token headroom is late confirmation.** The first state with accuracy ≥ 0.5 comes at
  a median of 29% of the horizon, yet the model keeps reasoning.
- **Answer stability predicts correctness.** It is not enough to justify stopping:

  | Steps the current answer has been stable | States | Accuracy now | Final accuracy |
  |---|---:|---:|---:|
  | 0 | 663 | 0.18 | 0.67 |
  | 1 | 191 | 0.36 | 0.74 |
  | 2 | 101 | 0.56 | 0.81 |
  | 3 | 78 | 0.64 | 0.85 |
  | 4 | 61 | 0.70 | 0.84 |
  | 5+ | 300 | 0.83 | 0.92 |

  Even after 5+ stable steps, finishing is about 9 points more accurate than stopping. A
  prototype LSM on answer-stability features alone also chose never to stop early. The same held
  when answers were matched by mathematical equivalence instead of exact text, which adds only
  about 1 point to answer convergence.

## Idea 3 pilot (elicited remaining steps)

| Metric | Value |
|---|---|
| Uncensored traces | 14 |
| Censored traces (hit the 2048-token cap) | 83 |
| Elicitations | 177 |
| Spearman correlation, elicited vs. realized remaining steps | 0.78 |
| Mean bias | −5.3 steps (underestimates remaining work) |
| Receding-horizon slope | 0.86 (SE 0.007) |

A slope of 0.86 means the predicted finishing point moves 0.86 steps later for every step taken.
This is suggestive of a receding horizon, but 14 traces is too few for a claim.

## Idea 2 pilot (attention before flips)

The pilot crashed: fp16 probe logits produced values outside [0, 1]. This is fixed in `edd861f`, which
uses float32 probe logits, a rounding tolerance and per-event skips. With 5 flip events, run 1
could not have powered this pilot anyway.

## Changes made for run 2

| Change | Commit |
|---|---|
| Shrunk LSM dual and L1-penalized pathwise dual, tuned by inner cross-validation | `edd861f`, `d363b39` |
| Headline bound is min(hindsight, duals) | `edd861f` |
| LSM with a cross-validated stopping margin that can reproduce full traces | `edd861f` |
| Collection with 4 siblings every 2nd step, 3072 thinking tokens, max 20 steps, ≤ 60 problems | `notebooks/kaggle_run.ipynb` |

Run 2 tests whether denser siblings make the dual bound informative, that is, below hindsight.

## Open problems

1. **A tighter upper bound needs cleaner continuation values.** If 4 siblings every 2nd step still
   gives shrink factor 0, the next options are more siblings at fewer steps, or value functions
   fit on sibling payoffs directly.
2. **The pilots need many more problems than the bounds do.** Flips occur in about 5% of traces. A
   collection without siblings over several hundred problems would serve Ideas 2 and 3 far better
   than the sibling-heavy runs.
3. **A stronger observable signal is needed for implementable gains.** Hidden-state PCA and answer
   stability both fall back to finishing. Candidates are disagreement between forced-answer samples
   (not stored today) and verifier-style self-checks.
