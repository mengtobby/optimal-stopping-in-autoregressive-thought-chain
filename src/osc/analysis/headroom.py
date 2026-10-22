"""Cross-fitted policy comparisons and primal-dual headroom decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from osc.stopping.dual import dual_upper_bound
from osc.stopping.lsm import LSMConfig, cross_fit_lsm
from osc.stopping.payoff import bootstrap_mean_ci, build_panel, evaluate_stop_times
from osc.stopping.prophet import (
    hindsight_prophet,
    independent_dp_thresholds,
    kleinberg_weinberg_threshold,
    samuel_cahn_threshold,
    threshold_stop_times,
    time_threshold_stop_times,
)


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


_DEFAULT_LSM_CONFIG = LSMConfig()


def headroom_report(
    traces: TraceSet,
    lam: float,
    token_scale: float = 1000.0,
    n_folds: int = 5,
    seed: int = 0,
    lsm_config: LSMConfig = _DEFAULT_LSM_CONFIG,
    budget_quantiles: tuple[float, ...] = tuple(i / 10 for i in range(1, 10)),
    convergence_ks: tuple[int, ...] = (2, 3, 4),
) -> HeadroomReport:
    """Evaluate held-out policies; oracle rows deliberately use ground truth.

    Dual rows are bounds, not stopping policies, so their accuracy/token fields
    are undefined. Bootstrap intervals condition on the fitted fold policies.
    """
    if not budget_quantiles or any(not 0 <= q <= 1 for q in budget_quantiles):
        raise HeadroomError("budget_quantiles must be nonempty and lie in [0, 1]")
    if not convergence_ks or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in convergence_ks
    ):
        raise HeadroomError("convergence_ks must contain positive integers")
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
                bootstrap_mean_ci(values, seed=seed),
                result.accuracy,
                result.mean_tokens,
                savings,
            )
        )

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
    notes = ["Intervals condition on fitted policies; they do not include training uncertainty."]
    upper, se = float("nan"), float("nan")
    if any(trace.has_siblings for trace in traces):
        values = np.empty(len(traces))
        for fold, policy in enumerate(policies):
            indices = np.flatnonzero(folds == fold)
            values[indices] = dual_upper_bound(panel.subset(indices), policy.value).per_trace
        upper = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(
            HeadroomRow(
                "dual-upper-cross-fit",
                "bound",
                upper,
                bootstrap_mean_ci(values, seed=seed),
                float("nan"),
                float("nan"),
                float("nan"),
            )
        )
        notes.append("Dual accuracy and token usage are undefined because a bound is not a policy.")
    else:
        notes.append("No traces have siblings; the dual upper bound is unavailable.")
    best = max(row.value for row in rows if row.family in ("heuristic", "learned"))
    return HeadroomReport(
        float(lam),
        len(traces),
        tuple(rows),
        prophet.value,
        upper,
        se,
        lower,
        prophet.value - upper,
        upper - best,
        tuple(notes),
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
            f"Hindsight value: {number(report.hindsight_value)}. "
            f"Dual upper: {number(report.dual_upper)} (SE {number(report.dual_upper_se)}). "
            f"LSM lower: {number(report.lsm_lower)}. "
            f"Irreducible gap: {number(report.irreducible_gap)}. "
            f"Headroom: {number(report.headroom)}.",
            "",
            "Hindsight can choose after seeing the future. The dual upper bound limits "
            "what an observable stopping policy can achieve, while held-out LSM gives an "
            "achievable lower estimate. The hindsight-minus-dual gap measures the estimated "
            "cost of lacking future information; headroom is the dual minus the best "
            "evaluated implementable policy. Sampling noise can reverse estimated gaps.",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines) + "\n"
