# Optimal Stopping in Autoregressive Thought Chains

Research code for measuring how much "overthinking" in reasoning models can be removed by any early-exit rule, and why the rest remains.

- [docs/research-ideas.md](docs/research-ideas.md): candidate ideas, the prior work they were checked against, and the chosen direction.
- [docs/results-run1.md](docs/results-run1.md): first real-trace results (math500, R1-Distill-Qwen-1.5B) and what changed for run 2.
- [docs/results-run2.md](docs/results-run2.md): denser siblings; why sample size now limits the bounds.
- [docs/cloud.md](docs/cloud.md): running collection and analysis on free Kaggle/Colab GPUs.

## What is here

| Module | Purpose |
|---|---|
| `osc.data` | Immutable `Trace`/`TraceSet` records, pickle-free IO |
| `osc.stopping.payoff` | Padded payoff panel `Z = accuracy − λ·tokens/1000`, stop-time evaluation |
| `osc.stopping.lsm` | Longstaff–Schwartz primal policy (cross-fitted, problem-grouped folds) → lower bound |
| `osc.stopping.dual` | Andersen–Broadie/Rogers martingale dual from sibling continuations → upper bound |
| `osc.stopping.pathwise` | Pathwise-optimized dual (LP over basis martingales, cross-fitted) → tighter upper bound |
| `osc.stopping.prophet`, `baselines` | Hindsight prophet, prophet-inequality thresholds, token budget, answer convergence, correctness probe |
| `osc.stopping.synthetic` | Markov reasoning models with exact Snell-envelope values (sanity checks) |
| `osc.stopping.stats` | Problem-clustered bootstrap CIs and standard errors |
| `osc.analysis.headroom` | Headroom decomposition report (markdown + JSON) |
| `osc.collect` | Step segmentation (causal rule), answer grading, datasets, HF backend, collection pipeline |
| `osc.pilots` | Idea 2: attention to the earlier answer before correct→wrong flips. Idea 3: time consistency of self-predicted remaining steps |

## Decomposition

```
hindsight prophet  ≥  dual upper bound  ≥  V*(prefix information)  ≥  learned policy (LSM)  ≥  heuristics
   └─ foresight gap ─┘                     └──────── representation + headroom ────────┘
```

The dual bound is valid because sibling continuations are sampled from the same next-step law as the trace itself, which is why segmentation uses a causal step rule and siblings share the thinking budget.

## Usage

```bash
pip install -e ".[dev]"              # core (numpy/scipy/sklearn)
pip install -e ".[model]"            # GPU machine: torch/transformers

# Synthetic sanity check (no GPU)
python scripts/headroom.py --synthetic 3000 --model overthinking --feature-noise 0.6 \
    --lam 0.05 --k-siblings 8 --out results/synthetic

# Real traces (GPU): collect, then analyse
python scripts/collect.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dataset math500 \
    --limit 500 --out data/traces/r1d7b-math500
python scripts/headroom.py --traces data/traces/r1d7b-math500/merged --lam 0.0 0.05 0.1 --out results/r1d7b
python scripts/pilot_flips.py  --traces data/traces/r1d7b-math500/merged --problems problems.jsonl --model ... --out results/flips.json
python scripts/pilot_elicit.py --traces data/traces/r1d7b-math500/merged --problems problems.jsonl --model ... --out results/elicit.json
```

Tests: `pytest -q --cov` (coverage gate 80%). Large-sample statistical checks are
marked `slow` and skipped by default; run them with `pytest -m slow`.

## Status (2026-11-02)

- Core bounds, baselines, collection and pilots are implemented and tested on synthetic data.
- Run 1 (97 math500 traces, free Kaggle T4) is analysed in [docs/results-run1.md](docs/results-run1.md):
  - Hindsight stopping would reach 83% accuracy with 65% fewer tokens, against 76% for full traces.
  - No implementable policy beats finishing yet.
  - The regularized dual bounds fall back to hindsight because 2 siblings every 4th step are too noisy.
- Run 2 (41 traces, 4 siblings every 2nd step) is analysed in [docs/results-run2.md](docs/results-run2.md):
  - Denser siblings let some martingale signal through, but the bound is still not below hindsight.
  - With about 0.06 standard error, sample size is now the limit.
  - The correctness probe is the only policy that beats finishing (at λ = 0.05 and 0.1).
- Next: a sibling-free collection of about 300 problems (`notebooks/kaggle_pilots.ipynb`) for
  signal-aware policies and the pilots.
