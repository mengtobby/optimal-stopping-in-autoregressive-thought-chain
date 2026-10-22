# Optimal Stopping in Autoregressive Thought Chains — Research Ideas

Target venue: NeurIPS / ICML (main track). Literature check performed 2026-10-20 (~25 targeted searches; not exhaustive — re-check before writing).

## Core problem

Long reasoning chains "overthink": models second-guess correct intermediate states or enter degenerate backtracking loops. Existing early exit uses static token caps or ad-hoc entropy/confidence thresholds.

## What is already taken (avoid pitching as novel)

| Area | Prior work |
|---|---|
| CoT pruning framed as optimal stopping (learned stop head, Bellman eq.) | OS-Pruner [2607.11089](https://arxiv.org/abs/2607.11089); ESTAR [2602.10004](https://arxiv.org/abs/2602.10004) |
| Anytime-valid / e-process stopping | [2610.25212](https://arxiv.org/abs/2610.25212); Sequential-EDFL [2510.06478](https://arxiv.org/abs/2510.06478); Certified Self-Consistency [2510.17472](https://arxiv.org/abs/2510.17472); [2602.13935](https://arxiv.org/abs/2602.13935) |
| Conformal / risk-controlled budgets | Conformal Thinking [2602.03814](https://arxiv.org/abs/2602.03814); Compute When Worth It (NeurIPS 2025); Thought Calibration [2505.18404](https://arxiv.org/abs/2505.18404) |
| Confidence / entropy / convergence heuristics | DEER; ES-CoT [2509.14004](https://arxiv.org/abs/2509.14004); EAT [2509.26522](https://arxiv.org/abs/2509.26522); HALT-CoT; ThinkBrake [2510.00546](https://arxiv.org/abs/2510.00546); Answer Convergence (EMNLP 2025) |
| Loop detection | Circular Reasoning (CUSUM on hidden states) [2601.05693](https://arxiv.org/abs/2601.05693) |
| Belief martingales | Martingale Score [2512.02914](https://arxiv.org/abs/2512.02914); From Drift to Coherence [2606.17832](https://arxiv.org/abs/2606.17832) |
| VOC-penalized training | Rational Metareasoning [2410.05563](https://arxiv.org/abs/2410.05563); S-GRPO |
| Parallel first-to-finish | short-m@k [2505.17813](https://arxiv.org/abs/2505.17813); First Finish Search [2505.18149](https://arxiv.org/abs/2505.18149) |
| Global budget allocation | CLEAR shadow price [2606.03092](https://arxiv.org/abs/2606.03092) |
| Step-error hazard | "When Does Reasoning Age?" (EMNLP 2026) |

## Ideas (ranked by impact × novelty)

### 1. Speed-of-light bounds on removable overthinking
Published "oracle stopping" savings use a hindsight prophet, unattainable by any prefix-adapted rule. Borrow American-option pricing:
- **Primal:** Longstaff–Schwartz backward induction on hidden states → stopping policy (lower bound).
- **Dual:** Rogers / Haugh–Kogan martingale duality → certified upper bound on value achievable by *any* rule using the model's own filtration.
- **Prophet inequalities:** when does a single threshold achieve ≥1/2 of the prophet under strongly correlated step values?

Decomposition: hindsight-prophet − dual = irreducible overthinking; dual − existing methods = remaining headroom.
Novelty: no dual/prophet results in LLM stopping literature; OS-Pruner has only a separation theorem.

### 2. "Overthinking is forgetting, not failing to stop"
Hypothesis: correct→wrong flips occur when attention/attribution from the current position to the earlier conclusion span decays (self-generated lost-in-the-middle). Test with attention/attribution tracking, activation patching, and *conclusion pinning* (KV-cache / attention boost on earlier answer span). If flips drop without truncation, the problem is memory, not stopping.
Closest: flip statistics [2604.10739](https://arxiv.org/abs/2604.10739); reasoning-focus heads [2509.23676](https://arxiv.org/abs/2509.23676). No link found.

### 3. Time inconsistency and commitment devices
Elicit mid-trace plans ("how many more steps?") vs realized behaviour; measure preference reversals; fit quasi-hyperbolic (β–δ) model to the stop hazard. Prediction: binding self-set budgets beat external budgets. Closest: SelfBudgeter (from memory, unverified).

### 4. Inverse optimal stopping
Recover the implicit per-token cost λ(x,t) rationalizing observed `</think>` times. Predict RL-trained reasoners have negative/difficulty-dependent λ; correct via `</think>` logit shift (training-free).

### 5. Critical slowing down before reasoning collapse
Rising lag-1 autocorrelation, variance, flickering, RQA determinism in hidden-state windows as label-free early-warning signals preceding loops; respond with rescue perturbations rather than stopping.

### 6. Stop / continue / restart / rewind with Luby schedules
If time-to-solution is heavy-tailed, universal Luby restarts are within O(log) of optimal with no distributional knowledge. Estimate per-model solve hazard; learn state-conditioned restarts.

### 7. Whittle indices for fleet-level thinking
Per-request reasoning as restless arms (state = probe belief); prove indexability; dynamic extension of the CLEAR shadow price. Better suited to MLSys.

### 8. Innovation-to-drift stopping
Online Doob decomposition of the forced-answer belief process; stop when predictable drift dominates. Weakest novelty (adjacent to Martingale Score, e-processes).

## Recommendation

Flagship paper: **Idea 1 as the theoretical spine + Idea 2 (or 3) as the surprising empirical finding** — "Here is the provable ceiling on removable overthinking; most of what's left isn't a stopping problem."

Cheap pilots (R1-Distill-Qwen-7B on MATH-500):
1. Idea 2: attention to earlier-answer span immediately before flips vs non-flips.
2. Idea 3: elicited vs realized remaining steps.
3. Idea 5: lag-1 autocorrelation of hidden states before loop onset.
