# Pilots run: forced-answer signals (math500, R1-Distill-Qwen-1.5B)

Kaggle T4, free tier, 2026-11-02, code at `8cc79d6`. Raw outputs are in
`results/cloud-math500-pilots/` (gitignored). This run dropped siblings and collected 4 forced
answers per state. It tested two questions:
- Do agreement and confidence across forced answers let a policy stop early without losing accuracy?
- Do the Idea 2 and Idea 3 pilots get enough events to test?

## Setup

| Setting | Run 1 | Run 2 | Pilots run |
|---|---|---|---|
| Problems collected | 97 | 41 of 60 | 102 of 300 (time budget) |
| Collection time | 8.3 h | 8.3 h | 8.3 h (294 s per problem) |
| Thinking tokens / max steps | 2048 / 16 | 3072 / 20 | 3072 / 20 |
| Mean horizon (states) | 16.4 | 19.3 | 19.6 |
| Siblings | 2 every 4th step | 4 every 2nd step | none |
| Forced answers per state | 1 | 1 | 4 (agreement and confidence) |
| Full-trace accuracy | 0.763 | 0.744 | 0.730 |

The forced samples cost more time than expected, so the run collected 102 problems instead of the
planned 150–250. With no siblings, the dual bound is unavailable, and the selected upper bound is
the hindsight prophet.

## Headroom (held-out, 5 problem-grouped folds)

Each value is accuracy − λ·tokens/1000, averaged over traces. The 95% intervals are about ±0.08.

| Policy | λ = 0 | λ = 0.05 | λ = 0.1 |
|---|---:|---:|---:|
| Hindsight prophet (oracle, upper bound) | 0.809 | 0.773 | 0.736 |
| Full trace | 0.730 | 0.626 | 0.521 |
| Correctness probe (cross-fit) | 0.728 | **0.642** | **0.577** |
| Correctness probe + signals | 0.703 | 0.631 | 0.560 |
| LSM (cross-fit, margin-tuned) | 0.730 | 0.627 | 0.532 |
| LSM + signals | 0.730 | 0.629 | 0.522 |
| Token budget (cross-fit) | 0.608 | 0.560 | 0.517 |
| Answer convergence (best k, cross-fit) | 0.581 | 0.536 | 0.492 |

The signal-aware policies now use six signals:
- `stable_run`, `agree_past`, `n_distinct` and `has_answer`, as in runs 1 and 2;
- `forced_agreement`: the share of the 4 forced answers that match the current one;
- `forced_confidence`: the mean token probability of the forced answers.

Adding them does not help. The probe gets slightly worse at every λ. The probe is again the only
policy ahead of finishing at λ > 0, as in run 2, but its intervals still overlap with finishing.

## Why strong signals do not produce early stops

`scripts/signal_diagnostics.py` checks the signals directly. Its output is in
`results/cloud-math500-pilots/signal_diagnostics.txt`.

**The signals predict whether the current answer is correct.** The table gives AUROC over 2002
states, with 95% intervals from a problem bootstrap:

| Signal | AUROC |
|---|---:|
| `forced_confidence` | 0.86 [0.81, 0.90] |
| `forced_agreement` | 0.83 [0.78, 0.87] |
| `stable_run` | 0.80 [0.75, 0.84] |
| `agree_past` | 0.70 [0.64, 0.76] |
| Step position (baseline) | 0.66 [0.63, 0.69] |

**Being right now is not enough, because finishing still pays.** This table compares accuracy at
a state with accuracy at the end of the same trace:

| Forced agreement | States | Accuracy now | Accuracy at end | Gain from finishing |
|---:|---:|---:|---:|---:|
| 0.25 | 289 | 0.03 | 0.56 | +0.53 |
| 0.50 | 447 | 0.11 | 0.57 | +0.46 |
| 0.75 | 286 | 0.25 | 0.65 | +0.41 |
| 1.00 | 980 | 0.74 | 0.85 | +0.11 |

Even when all 4 forced answers agree, finishing adds 11 points. Early states where all 4 agree
are often wrong: the first such state is correct only 47% of the time.

The simplest rule that uses both signals stops at the first state where all forced answers agree
and confidence is at least c. Here c is tuned on the training folds over 5 random splits:

| λ | Finish | Consensus rule | Paired difference [95% CI] |
|---|---:|---:|---|
| 0 | 0.730 | 0.725 | −0.006 [−0.018, 0.000] |
| 0.05 | 0.626 | 0.624 | −0.002 [−0.027, +0.028] |
| 0.1 | 0.521 | 0.521 | +0.000 [−0.036, +0.039] |

Choosing c on the full sample instead suggested a gain of +0.03 at λ = 0.1. Cross-fitting removes
it, so that gain was selection, not signal. Tuning learns that only very confident late states are
safe to stop at. Those states are already near the end, so stopping there saves few tokens.

## Pilots

| Pilot | Run 2 | Pilots run |
|---|---|---|
| Idea 3: elicited remaining steps | 10 uncensored traces; Spearman 0.80; bias −6.0; slope 0.88 | 21 uncensored (81 excluded); Spearman 0.68; bias −6.4 steps; slope 0.89 (SE 0.005); reversal rate 0.85 |
| Idea 2: attention before flips | 5 flips, 18 controls; AUROC 0.27, p = 0.76 | 9 flips, 50 controls, 26 skipped; AUROC 0.45, p = 0.66 |

- **Idea 3 replicates for the third time.** The model underestimates its remaining work by about
  6 steps. Its predicted finishing point recedes by about 0.89 steps for every step taken, so the
  estimated finish keeps moving later. The slope is stable across runs (0.86, 0.88, 0.89).
  The sample is still small: 21 traces, and censored traces are excluded, which biases toward
  short traces.
- **Idea 2 shows no effect.** With 9 flip events, attention to the earlier answer does not separate
  flips from controls (AUROC 0.45, p = 0.66). A null at this size does not rule out an effect, but
  no run has hinted at one.

## What this run shows

1. **Observable answer signals predict current correctness well, and that does not translate into
   early stopping.** The bottleneck is not detecting a correct answer. The model keeps fixing wrong
   ones that already look settled. Policies would need to predict the value of continuing, not
   current correctness.
2. **Across three runs, nothing implementable beats finishing by more than its interval.** The
   hindsight oracle sits 5–22 points above finishing. Most of that gap appears to be foresight the
   observable signals cannot provide, but the dual bound needed to show this is not yet tight
   enough.
3. **Idea 3 is the most robust finding so far.** It is the only result that has replicated with
   stable estimates.

## Next steps

1. **Make Idea 3 the next paper-level result.** Raise the elicitation budget so fewer traces are
   censored. Report the receding slope with a problem-clustered interval, and test whether it
   depends on difficulty or on the step count.
2. **Store per-trace stop times and payoffs in headroom reports,** so paired comparisons between
   policies (for example, probe vs finishing) can be made from the reports alone.
3. **Pause the Idea 2 attention pilot** unless a larger run finds at least about 30 flip events.
4. **Do not spend GPU time on more forced samples.** They add generation at every state for signals that
   do not change decisions.
