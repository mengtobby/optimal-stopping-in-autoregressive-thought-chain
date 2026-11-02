"""Cross-fitted policy comparisons and primal-dual headroom decomposition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from osc.data.schema import TraceSet
from osc.stopping.baselines import (
    answer_convergence_stop_times,
    fit_correctness_probe,
    full_trace_stop_times,
    probe_threshold_stop_times,
    select_param,
    token_budget_stop_times,
)
from osc.stopping.dual import ValueFn, cross_fit_shrunk_dual
from osc.stopping.lsm import LSMConfig, cross_fit_lsm, fit_lsm
from osc.stopping.pathwise import PathwiseConfig, cross_fit_pathwise_dual
from osc.stopping.payoff import PayoffPanel, build_panel, evaluate_stop_times
from osc.stopping.prophet import (
    hindsight_prophet,
    independent_dp_thresholds,
    kleinberg_weinberg_threshold,
    samuel_cahn_threshold,
    threshold_stop_times,
    time_threshold_stop_times,
)
from osc.stopping.signals import with_signals
from osc.stopping.stats import cluster_bootstrap_mean_ci, cluster_standard_error


class HeadroomError(ValueError):
    """Invalid report configuration."""


@dataclass(frozen=True)
class HeadroomRow:
    method: str
    family: str
    value: float
    value_ci: tuple[float, float]
    accuracy: float
    mean_tokens: float
    token_savings: float


@dataclass(frozen=True)
class HeadroomReport:
    lam: float
    n_traces: int
    rows: tuple[HeadroomRow, ...]
    hindsight_value: float
    dual_upper: float
    dual_upper_se: float
    lsm_lower: float
    irreducible_gap: float
    headroom: float
    notes: tuple[str, ...]
    headroom_paired: float = float("nan")
    headroom_paired_ci: tuple[float, float] = (float("nan"), float("nan"))
    dual_lsm_upper: float = float("nan")
    dual_lsm_upper_se: float = float("nan")
    lsm_shrink: tuple[float, ...] = ()
    upper_bound: float = float("nan")
    upper_bound_source: str = ""
    signal_names: tuple[str, ...] = ()


_DEFAULT_LSM_CONFIG = LSMConfig()


def _lsm_refitter(config: LSMConfig, alpha: float) -> Callable[[PayoffPanel], ValueFn]:
    """Refit with the outer fold's selected alpha, so shrinkage suits that value fn."""
    tuned = replace(config, ridge_alpha=alpha)
    return lambda train: fit_lsm(train, tuned).value


def headroom_report(
    traces: TraceSet,
    lam: float,
    token_scale: float = 1000.0,
    n_folds: int = 5,
    seed: int = 0,
    lsm_config: LSMConfig = _DEFAULT_LSM_CONFIG,
    budget_quantiles: tuple[float, ...] = tuple(i / 10 for i in range(1, 10)),
    convergence_ks: tuple[int, ...] = (2, 3, 4),
    exact_stop_times: np.ndarray | None = None,
    pathwise_config: PathwiseConfig | None = None,
) -> HeadroomReport:
    """Evaluate held-out policies; oracle rows deliberately use ground truth.

    Dual rows are bounds, not stopping policies, so their accuracy/token fields
    are undefined. Bootstrap intervals condition on the fitted fold policies.
    Paired headroom bootstraps dual_i - payoff_i(tau_i) on the same paths;
    pathwise this difference is at least -M_i(tau_i).
    """
    if not budget_quantiles or any(not 0 <= q <= 1 for q in budget_quantiles):
        raise HeadroomError("budget_quantiles must be nonempty and lie in [0, 1]")
    if not convergence_ks or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in convergence_ks
    ):
        raise HeadroomError("convergence_ks must contain positive integers")
    clusters = [trace.problem_id for trace in traces]
    panel = build_panel(traces, lam, token_scale)
    lsm_times, policies, folds = cross_fit_lsm(panel, lsm_config, n_folds, seed)
    full_tokens = float(panel.cum_tokens[np.arange(len(traces)), panel.horizons - 1].mean())
    rows: list[HeadroomRow] = []

    def add(method: str, family: str, times: np.ndarray) -> None:
        result = evaluate_stop_times(panel, times, method)
        values = panel.payoff[np.arange(len(traces)), times]
        savings = 1 - result.mean_tokens / full_tokens if full_tokens else 0.0
        rows.append(
            HeadroomRow(
                method,
                family,
                result.value,
                cluster_bootstrap_mean_ci(values, clusters, seed=seed),
                result.accuracy,
                result.mean_tokens,
                savings,
            )
        )

    if exact_stop_times is not None:
        add("exact-optimal", "oracle", exact_stop_times)
    prophet = hindsight_prophet(panel)
    add("hindsight-prophet", "oracle", prophet.stop_times)
    add("oracle-Samuel-Cahn", "oracle", threshold_stop_times(panel, samuel_cahn_threshold(panel)))
    add(
        "oracle-Kleinberg-Weinberg",
        "oracle",
        threshold_stop_times(panel, kleinberg_weinberg_threshold(panel)),
    )
    add(
        "oracle-independent-DP",
        "oracle",
        time_threshold_stop_times(panel, independent_dp_thresholds(panel)),
    )
    add("full-trace", "heuristic", full_trace_stop_times(panel))
    for k in convergence_ks:
        add(f"answer-convergence-k{k}", "heuristic", answer_convergence_stop_times(panel, k))
    budget_times = np.empty(len(traces), dtype=int)
    convergence_times = np.empty(len(traces), dtype=int)
    probe_times = np.empty(len(traces), dtype=int)
    for fold in range(n_folds):
        test_indices = np.flatnonzero(folds == fold)
        train = panel.subset(np.flatnonzero(folds != fold))
        test = panel.subset(test_indices)
        budgets = np.quantile(train.cum_tokens[train.alive], budget_quantiles).tolist()
        budget = select_param(train, budgets, token_budget_stop_times)
        budget_times[test_indices] = token_budget_stop_times(test, budget)
        k = select_param(
            train, convergence_ks, lambda p, c: answer_convergence_stop_times(p, int(c))
        )
        convergence_times[test_indices] = answer_convergence_stop_times(test, int(k))
        probe = fit_correctness_probe(train)
        train_scores = probe.scores(train)
        thresholds = np.linspace(0, 1, 21).tolist() + [1.01]
        threshold = select_param(
            train,
            thresholds,
            lambda p, c, scores=train_scores: probe_threshold_stop_times(p, scores, c),
        )
        probe_times[test_indices] = probe_threshold_stop_times(test, probe.scores(test), threshold)
    add("token-budget-cross-fit", "heuristic", budget_times)
    add("answer-convergence-best-cross-fit", "heuristic", convergence_times)
    add("correctness-probe-cross-fit", "learned", probe_times)
    add("LSM-cross-fit", "learned", lsm_times)
    lower = rows[-1].value
    sig_panel, names = with_signals(panel)
    signal_times, _, signal_folds = cross_fit_lsm(
        sig_panel, replace(lsm_config, passthrough=len(names)), n_folds, seed
    )
    assert np.array_equal(signal_folds, folds)
    add("LSM-signals-cross-fit", "learned", signal_times)
    signal_probe_times = np.empty(len(traces), dtype=int)
    for fold in range(n_folds):
        test_indices = np.flatnonzero(folds == fold)
        train_sig = sig_panel.subset(np.flatnonzero(folds != fold))
        test_sig = sig_panel.subset(test_indices)
        probe = fit_correctness_probe(train_sig, passthrough=len(names))
        train_scores = probe.scores(train_sig)
        thresholds = np.linspace(0, 1, 21).tolist() + [1.01]
        threshold = select_param(
            train_sig,
            thresholds,
            lambda p, c, scores=train_scores: probe_threshold_stop_times(p, scores, c),
        )
        signal_probe_times[test_indices] = probe_threshold_stop_times(
            test_sig, probe.scores(test_sig), threshold
        )
    add("correctness-probe-signals-cross-fit", "learned", signal_probe_times)
    notes = ["Intervals condition on fitted policies; they do not include training uncertainty."]
    notes.append("Confidence intervals and dual standard errors are clustered by problem.")
    upper, se = float("nan"), float("nan")
    lsm_upper, lsm_se = float("nan"), float("nan")
    paired = float("nan")
    paired_ci = (float("nan"), float("nan"))
    lsm_shrink: tuple[float, ...] = ()
    # Dual anchors stay features-only: sibling states carry no answer signals.
    if any(trace.has_siblings for trace in traces):
        lsm_dual, lsm_shrink = cross_fit_shrunk_dual(
            panel,
            folds,
            [p.value for p in policies],
            fit_value_fn=[_lsm_refitter(lsm_config, p.payoff_model.alpha) for p in policies],
            seed=seed,
        )
        values = lsm_dual.per_trace
        lsm_upper = float(values.mean())
        lsm_se = cluster_standard_error(values, clusters)
        rows.append(
            HeadroomRow(
                "dual-lsm-shrunk-cross-fit",
                "bound",
                lsm_upper,
                cluster_bootstrap_mean_ci(values, clusters, seed=seed),
                float("nan"),
                float("nan"),
                float("nan"),
            )
        )
        anchors_by_fold = [
            [
                ("lsm_value", policy.value),
                ("lsm_payoff", policy.payoff_estimate),
                ("lsm_continuation", policy.continuation),
            ]
            for policy in policies
        ]
        pathwise, _ = cross_fit_pathwise_dual(
            panel,
            folds,
            anchors_by_fold,
            PathwiseConfig() if pathwise_config is None else pathwise_config,
        )
        values = pathwise.per_trace
        upper = float(values.mean())
        se = cluster_standard_error(values, clusters)
        differences = values - panel.payoff[np.arange(len(traces)), lsm_times]
        paired = float(differences.mean())
        paired_ci = cluster_bootstrap_mean_ci(differences, clusters, seed=seed)
        rows.append(
            HeadroomRow(
                "dual-pathwise-l1-cross-fit",
                "bound",
                upper,
                cluster_bootstrap_mean_ci(values, clusters, seed=seed),
                float("nan"),
                float("nan"),
                float("nan"),
            )
        )
        notes.append("Dual accuracy and token usage are undefined because a bound is not a policy.")
    else:
        notes.append("No traces have siblings; the dual upper bound is unavailable.")
    best = max(row.value for row in rows if row.family in ("heuristic", "learned"))
    upper_bound_source, upper_bound = min(
        (
            (source, value)
            for source, value in (
                ("hindsight-prophet", prophet.value),
                ("dual-pathwise-l1-cross-fit", upper),
                ("dual-lsm-shrunk-cross-fit", lsm_upper),
            )
            if not np.isnan(value)
        ),
        key=lambda item: item[1],
    )
    return HeadroomReport(
        float(lam),
        len(traces),
        tuple(rows),
        prophet.value,
        upper,
        se,
        lower,
        prophet.value - upper_bound,
        upper_bound - best,
        tuple(notes),
        paired,
        paired_ci,
        lsm_upper,
        lsm_se,
        lsm_shrink,
        upper_bound,
        upper_bound_source,
        names,
    )


def report_to_json(report: HeadroomReport) -> dict[str, Any]:
    """Return strict JSON-compatible data, representing undefined numbers as null."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    return clean(asdict(report))


def report_to_markdown(report: HeadroomReport) -> str:
    """Render all method rows and a plain-language interpretation."""

    def number(value: float) -> str:
        return f"{value:.4f}" if np.isfinite(value) else "N/A"

    lines = [
        f"# Headroom report (lambda={report.lam:g}, N={report.n_traces})",
        "",
        "| Method | Family | Value | 95% CI | Accuracy | Mean tokens | Token savings |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in report.rows:
        ci = f"[{number(row.value_ci[0])}, {number(row.value_ci[1])}]"
        lines.append(
            f"| {row.method} | {row.family} | {number(row.value)} | {ci} | "
            f"{number(row.accuracy)} | {number(row.mean_tokens)} | "
            f"{number(row.token_savings)} |"
        )
    lines.extend(
        [
            "",
            "Answer signals: " + (", ".join(report.signal_names) or "N/A") + ".",
            "Signal policies use answer stability and, when collected, trial-answer agreement "
            "and confidence; dual bounds use hidden-state features only because sibling states "
            "carry no answers.",
            f"Selected upper bound: {number(report.upper_bound)}. "
            f"Source: {report.upper_bound_source}.",
            "LSM shrink factors by fold: "
            + (", ".join(number(c) for c in report.lsm_shrink) or "N/A")
            + ".",
            "The hindsight value is itself a valid upper bound (zero martingale); "
            "the reported bound is the smallest of three valid estimates, which is "
            "slightly optimistic in finite samples.",
            "",
            f"Hindsight value: {number(report.hindsight_value)}. "
            f"Pathwise dual upper: {number(report.dual_upper)} "
            f"(SE {number(report.dual_upper_se)}). "
            f"LSM dual upper: {number(report.dual_lsm_upper)} "
            f"(SE {number(report.dual_lsm_upper_se)}). "
            f"LSM lower: {number(report.lsm_lower)}. "
            f"Irreducible gap: {number(report.irreducible_gap)}. "
            f"Headroom: {number(report.headroom)}. "
            f"Paired headroom: {number(report.headroom_paired)} "
            f"(95% CI [{number(report.headroom_paired_ci[0])}, "
            f"{number(report.headroom_paired_ci[1])}]).",
            "",
            "Hindsight can choose after seeing the future. The dual upper bound limits "
            "what an observable stopping policy can achieve, while held-out LSM gives an "
            "achievable lower estimate. The hindsight-minus-selected-bound gap measures "
            "the estimated cost of lacking future information; paired headroom is the mean "
            "per-trace dual minus LSM payoff, with a paired bootstrap interval. "
            "Headroom is the selected upper bound minus the best "
            "evaluated implementable policy. Sampling noise can reverse estimated gaps.",
            "The pathwise bound targets a tighter upper estimate by optimizing a combination "
            "of basis martingales on training folds, including the LSM anchors, then "
            "evaluating on held-out problems.",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines) + "\n"
