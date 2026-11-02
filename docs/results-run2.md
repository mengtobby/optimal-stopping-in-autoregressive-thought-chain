# Run 2: denser siblings (math500, R1-Distill-Qwen-1.5B)

Kaggle T4, free tier, 2026-11-01, code at `d363b39`. Raw outputs are in `results/cloud-math500-r2/`
(gitignored). Run 2 tested one question from [run 1](results-run1.md): do denser sibling
continuations make the dual upper bound informative, that is, below the hindsight value?

## Setup

| Setting | Run 1 | Run 2 |
|---|---|---|
| Problems collected | 97 | 41 of 60 (time budget) |
| Collection time | 8.3 h | 8.3 h (727 s per problem) |
| Thinking tokens / max steps | 2048 / 16 | 3072 / 20 |
| Mean horizon (states) | 16.4 | 19.3 |
| Siblings | 2 per transition, every 4th | 4 per transition, every 2nd |
| Transitions with siblings | 380 of 1491 (25%) | 377 of 750 (50%) |

## Headroom (held-out, 5 problem-grouped folds)

Each value is accuracy − λ·tokens/1000, averaged over traces. With 41 problems, the standard errors
are about 0.06.

| Policy | λ = 0 | λ = 0.05 | λ = 0.1 |
|---|---:|---:|---:|
| Hindsight prophet (oracle) | 0.793 | 0.759 | 0.726 |
| LSM dual, shrunk | 0.827 | 0.779 | 0.741 |
| Pathwise dual, L1 | 0.793 | 0.761 | 0.726 |
| **Selected upper bound** | **0.793** | **0.759** | **0.726** |
| Full trace | 0.744 | 0.644 | 0.545 |
| Correctness probe (cross-fit) | 0.720 | **0.661** | **0.580** |
| LSM (cross-fit, margin-tuned) | 0.707 | 0.619 | 0.517 |
| Token budget (cross-fit) | 0.634 | 0.587 | 0.541 |
| Answer convergence (best k, cross-fit) | 0.610 | 0.567 | 0.524 |

Selected LSM shrink factors by fold:

| λ | Shrink factors |
|---|---|
| 0 | 0, 0, 1, 0, 0 |
| 0.05 | 0.1, 0, 0.5, 0, 0.1 |
| 0.1 | 0.1, 0.1, 0.25, 0.1, 0.5 |

Paired headroom (upper bound − LSM), with 95% CIs:

| λ | Paired headroom | 95% CI |
|---|---:|---|
| 0 | 0.085 | [0.024, 0.171] |
| 0.05 | 0.142 | [0.076, 0.223] |
| 0.1 | 0.209 | [0.135, 0.295] |

## What run 2 shows

- **Denser siblings carry some signal, but the bound is still not below hindsight.** In run 1,
  inner cross-validation chose shrink factor 0 everywhere. In run 2 it keeps 10–50% of the LSM
  martingale in most folds at λ > 0. Even so, both duals stay at or slightly above the hindsight
  value, so the selected bound is still the hindsight prophet. The foresight gap remains unmeasured.
- **Sample size now limits the bounds more than sibling density does.** Standard errors of about
  0.06 are larger than any plausible foresight gap, and fold-level tuning works with about 33
  training problems. Denser siblings also cost throughput: each problem took 2.4 times longer than
  in run 1.
- **The margin-tuned LSM loses to finishing at every λ.** In run 1 it fell back to finishing.
  With 41 problems, the inner cross-validation picks margins that stop too early on held-out
  problems.
- **The correctness probe is the only policy that beats finishing,** at λ = 0.05 (0.661 vs 0.644)
  and λ = 0.1 (0.580 vs 0.545). The intervals overlap heavily.

## Pilots

| Pilot | Result |
|---|---|
| Idea 3: elicited remaining steps | 10 uncensored traces (31 excluded); Spearman 0.80; mean bias −6.0 steps; receding-horizon slope 0.88 (SE 0.008) |
| Idea 2: attention before flips | Ran without crashing (fix confirmed). 5 flip events, 18 controls, 11 skipped; AUROC 0.27, permutation p = 0.76 |

The elicitation result matches run 1: the model underestimates its remaining work, and its
predicted finishing point recedes by about 0.9 steps per step taken. Both runs still have too few
uncensored traces to support a claim. The flips pilot runs end to end now, but 5 events give it no
power.

## Re-analysis with answer signals (runs 1 and 2)

The signals round (`f066ad3`, `2b0d809`) adds four answer-stability signals to the LSM and the
correctness probe, which bypass PCA:
- `stable_run`: steps the current answer has been unchanged.
- `agree_past`: share of earlier answers that agree with the current one.
- `n_distinct`: distinct answers so far.
- `has_answer`: whether the state has an answer.

Answers are matched by mathematical equivalence. Runs 1 and 2 have no forced-answer agreement or
confidence, so only these four signals were available. Results are in `results/*/rerun-r11/`.

| Run, λ | Full trace | LSM | LSM + signals | Probe | Probe + signals |
|---|---:|---:|---:|---:|---:|
| Run 1, 0 | 0.763 | 0.763 | 0.763 | 0.722 | 0.747 |
| Run 1, 0.05 | 0.680 | 0.680 | 0.675 | 0.666 | 0.681 |
| Run 1, 0.1 | 0.596 | 0.592 | 0.592 | 0.606 | 0.602 |
| Run 2, 0 | 0.744 | 0.707 | 0.720 | 0.720 | 0.707 |
| Run 2, 0.05 | 0.644 | 0.619 | 0.623 | 0.661 | 0.644 |
| Run 2, 0.1 | 0.545 | 0.517 | 0.525 | 0.580 | 0.582 |

The signals change values by at most 0.025, well inside the intervals (about ±0.08 in run 1 and
±0.13 in run 2). Answer stability alone does not make early stopping pay. This agrees with the
run 1 table, which shows that finishing still adds about 9 points after 5+ stable steps. The
untested signals are agreement and confidence across forced-answer samples, which the next
collection records.

## Next steps

1. **Collect many more problems without siblings.** The signals round (`kaggle_pilots.ipynb`)
   collects about 300 problems with 4 forced answers per state and no siblings. It stores
   forced-answer agreement and confidence for the signal-aware policies. It also gives the flips and
   elicitation pilots enough events to test.
2. **Revisit the bound only after policies improve.** A tighter dual needs both more problems and
   better value functions. The cheapest test is to keep run 2's sibling density and use about 150
   problems, which takes about 30 GPU hours.
