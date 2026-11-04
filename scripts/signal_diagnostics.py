"""Diagnose answer signals on collected traces: predictive power and a consensus stopping rule.

Prints (1) problem-clustered AUROCs of each signal for the current forced answer being
correct, (2) the gain from finishing grouped by forced-answer agreement, and (3) a
cross-fitted rule that stops at the first state where all forced answers agree with
confidence >= c (c tuned on training folds), compared with finishing.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from osc.data.io import load_traceset
from osc.stopping.payoff import PayoffPanel, build_panel
from osc.stopping.signals import answer_signals

N_BOOT = 1000
N_FOLDS = 5
N_SPLITS = 5
CONFIDENCE_GRID = np.concatenate(([np.inf], np.linspace(0.8, 0.995, 40)))


def _cluster_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    means = [values[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _auroc_table(panel: PayoffPanel, signals: np.ndarray, names, rng) -> None:
    rows, cols = np.nonzero(panel.alive)
    label = panel.accuracy[rows, cols] >= 0.5
    groups = [np.nonzero(rows == i)[0] for i in range(panel.n_traces)]
    span = np.maximum(panel.horizons[:, None] - 1, 1)
    scores = {name: signals[..., j] for j, name in enumerate(names)}
    scores["step_fraction"] = np.arange(panel.t_max)[None, :] / span
    print(f"AUROC, current forced answer correct ({len(rows)} states, problem bootstrap):")
    for name, score in scores.items():
        values = score[rows, cols]
        boots = []
        for _ in range(N_BOOT // 4):
            pick = np.concatenate(
                [groups[i] for i in rng.integers(0, panel.n_traces, panel.n_traces)]
            )
            if 0 < label[pick].mean() < 1:
                boots.append(roc_auc_score(label[pick], values[pick]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {name:18s} {roc_auc_score(label, values):.3f} [{lo:.3f}, {hi:.3f}]")


def _gain_by_agreement(panel: PayoffPanel, agree: np.ndarray, final: np.ndarray) -> None:
    print("Gain from finishing by forced agreement (accuracy at trace end minus now):")
    for level in np.unique(agree[panel.alive]):
        rows, cols = np.nonzero(panel.alive & (agree == level))
        now, end = panel.accuracy[rows, cols], final[rows]
        print(
            f"  agreement={level:.2f} states={len(rows):4d} problems={len(np.unique(rows)):3d} "
            f"now={now.mean():.3f} end={end.mean():.3f} gain={(end - now).mean():+.3f}"
        )


def _consensus_stop_times(panel: PayoffPanel, consensus: np.ndarray, conf, c) -> np.ndarray:
    hit = consensus & (conf >= c)
    last = panel.horizons - 1
    return np.where(hit.any(axis=1), hit.argmax(axis=1), last)


def _cross_fit_consensus(panel: PayoffPanel, consensus, conf, rng, lams) -> None:
    n = panel.n_traces
    idx = np.arange(n)
    taus = {c: _consensus_stop_times(panel, consensus, conf, c) for c in CONFIDENCE_GRID}
    print("Cross-fit consensus rule vs finishing (paired, problem bootstrap):")
    for lam in lams:
        pay = {
            c: panel.accuracy[idx, t] - lam * panel.cum_tokens[idx, t] / panel.token_scale
            for c, t in taus.items()
        }
        base = pay[np.inf]
        held = np.zeros(n)
        for split in range(N_SPLITS):
            folds = np.random.default_rng(split).permutation(n) % N_FOLDS
            for fold in range(N_FOLDS):
                train, test = folds != fold, folds == fold
                best = max(CONFIDENCE_GRID, key=lambda c, tr=train: pay[c][tr].mean())
                held[test] += pay[best][test] / N_SPLITS
        diff = held - base
        lo, hi = _cluster_ci(diff, rng)
        print(
            f"  lam={lam:g}: finish={base.mean():.3f} consensus={held.mean():.3f} "
            f"diff={diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--lam", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = load_traceset(args.traces)
    if any(t.forced_agreement is None or t.forced_confidence is None for t in traces):
        raise SystemExit(
            "traces lack forced_agreement/forced_confidence; collect with --n-forced > 1"
        )
    panel = build_panel(traces, 0.0)
    signals, names = answer_signals(panel)
    rng = np.random.default_rng(args.seed)
    final = panel.accuracy[np.arange(panel.n_traces), panel.horizons - 1]
    agree = np.where(panel.alive, signals[..., names.index("forced_agreement")], np.nan)
    conf = signals[..., names.index("forced_confidence")]

    print(f"N={panel.n_traces} horizon={panel.horizons.mean():.1f} final acc={final.mean():.3f}")
    _auroc_table(panel, signals, names, rng)
    _gain_by_agreement(panel, agree, final)
    _cross_fit_consensus(panel, panel.alive & (agree == 1.0), conf, rng, args.lam)


if __name__ == "__main__":
    main()
