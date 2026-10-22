"""Headroom decomposition and CLI integration checks."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from osc.analysis.headroom import (
    HeadroomError,
    headroom_report,
    report_to_json,
    report_to_markdown,
)
from osc.stopping.pathwise import PathwiseConfig, cross_fit_pathwise_dual
from osc.stopping.synthetic import (
    default_model,
    exact_optimal_stop_times,
    exact_value,
    overthinking_model,
    sample_traceset,
)


def test_synthetic_bounds_and_rendering():
    model = default_model()
    report = headroom_report(sample_traceset(model, 1500, 4, 0), 0.05)
    assert report.hindsight_value >= report.dual_upper - 1e-9
    assert report.dual_upper >= report.lsm_lower - 3 * report.dual_upper_se
    assert report.dual_upper >= exact_value(model, 0.05) - 3 * report.dual_upper_se
    markdown = report_to_markdown(report)
    assert all(row.method in markdown for row in report.rows)
    data = json.loads(json.dumps(report_to_json(report), allow_nan=False))
    assert data["n_traces"] == 1500
    assert data["rows"][-1]["accuracy"] is None
    best = max(r.value for r in report.rows if r.family in ("heuristic", "learned"))
    assert report.headroom == report.dual_upper - best


def test_no_siblings_and_validation():
    traces = sample_traceset(default_model(), 30, 0, 3)
    report = headroom_report(traces, 0.05)
    assert np.isnan(report.headroom_paired)
    assert report_to_json(report)["headroom_paired_ci"] == [None, None]
    assert np.isnan(report.dual_upper)
    assert np.isnan(report.dual_lsm_upper)
    assert report_to_json(report)["dual_lsm_upper_se"] is None
    assert any("No traces have siblings" in note for note in report.notes)
    assert all(row.family != "bound" for row in report.rows)
    assert report_to_json(report)["dual_upper"] is None
    assert "N/A" in report_to_markdown(report)
    with pytest.raises(HeadroomError, match="budget_quantiles"):
        headroom_report(traces, 0.05, budget_quantiles=())
    with pytest.raises(HeadroomError, match="convergence_ks"):
        headroom_report(traces, 0.05, convergence_ks=(0,))


def test_cli(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "headroom.py"
    spec = importlib.util.spec_from_file_location("headroom_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--synthetic", "300", "--lam", "0.05", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "report_lam0.05.json").read_text())
    assert data["exact_value"] == exact_value(default_model(), 0.05)
    assert (tmp_path / "report_lam0.05.md").is_file()
    assert any(row["method"] == "exact-optimal" for row in data["rows"])
    for noise in ("0", "0.6"):
        assert (
            module.main(
                [
                    "--synthetic",
                    "100",
                    "--model",
                    "overthinking",
                    "--feature-noise",
                    noise,
                    "--lam",
                    "0.05",
                    "--time-bins",
                    "2",
                    "--pathwise-components",
                    "0",
                    "--out",
                    str(tmp_path),
                ]
            )
            == 0
        )
        data = json.loads((tmp_path / "report_lam0.05.json").read_text())
        assert any(row["method"] == "exact-optimal" for row in data["rows"]) == (noise == "0")


def test_paired_headroom_and_exact_row():
    from osc.stopping.dual import dual_upper_bound, martingale_increments
    from osc.stopping.lsm import cross_fit_lsm
    from osc.stopping.payoff import build_panel
    from osc.stopping.stats import cluster_bootstrap_mean_ci, cluster_standard_error

    model = default_model()
    traces = sample_traceset(model, 300, 8, 8)
    traces = replace(
        traces,
        traces=tuple(
            replace(trace, problem_id=f"problem-{i // 3}") for i, trace in enumerate(traces)
        ),
    )
    clusters = [trace.problem_id for trace in traces]
    exact_times = exact_optimal_stop_times(model, 0.05, traces)
    report = headroom_report(traces, 0.05, seed=8, exact_stop_times=exact_times)
    panel = build_panel(traces, 0.05)
    times, policies, folds = cross_fit_lsm(panel, seed=8)
    differences = np.empty(len(traces))
    dual_values = np.empty(len(traces))
    for fold, policy in enumerate(policies):
        indices = np.flatnonzero(folds == fold)
        held = panel.subset(indices)
        dual = dual_upper_bound(held, policy.value)
        dual_values[indices] = dual.per_trace
        differences[indices] = dual.per_trace - held.payoff[np.arange(len(indices)), times[indices]]
        increments = martingale_increments(held, policy.value)
        martingale = np.column_stack(
            (np.zeros(len(indices)), np.cumsum(increments[:, :-1], axis=1))
        )
        assert np.all(
            differences[indices] >= -martingale[np.arange(len(indices)), times[indices]] - 1e-12
        )
    assert report.dual_lsm_upper == pytest.approx(dual_values.mean())
    assert report.dual_lsm_upper_se == cluster_standard_error(dual_values, clusters)
    lsm_row = next(row for row in report.rows if row.method == "dual-upper-cross-fit")
    assert lsm_row.value_ci == cluster_bootstrap_mean_ci(dual_values, clusters, seed=8)
    anchors = [
        [
            ("lsm_value", policy.value),
            ("lsm_payoff", policy.payoff_estimate),
            ("lsm_continuation", policy.continuation),
        ]
        for policy in policies
    ]
    result, _ = cross_fit_pathwise_dual(panel, folds, anchors)
    dual_values = result.per_trace
    differences = dual_values - panel.payoff[np.arange(len(traces)), times]
    assert report.headroom_paired == pytest.approx(differences.mean())
    assert report.headroom_paired == pytest.approx(report.dual_upper - report.lsm_lower)
    assert report.headroom_paired_ci == cluster_bootstrap_mean_ci(differences, clusters, seed=8)
    assert report.dual_upper_se == cluster_standard_error(dual_values, clusters)
    assert report.rows[-1].value_ci == cluster_bootstrap_mean_ci(dual_values, clusters, seed=8)
    assert any("clustered by problem" in note for note in report.notes)
    oracle = next(row for row in report.rows if row.method == "exact-optimal")
    assert oracle.value_ci == cluster_bootstrap_mean_ci(
        panel.payoff[np.arange(len(traces)), exact_times], clusters, seed=8
    )
    assert oracle.family == "oracle"
    assert oracle.value == pytest.approx(panel.payoff[np.arange(len(traces)), exact_times].mean())
    data = report_to_json(report)
    assert data["headroom_paired"] == report.headroom_paired
    assert data["headroom_paired_ci"] == list(report.headroom_paired_ci)
    assert "Paired headroom:" in report_to_markdown(report)


def test_noisy_pathwise_headline():
    traces = sample_traceset(overthinking_model(feature_noise=0.6), 2000, 8, 0)
    report = headroom_report(traces, 0.05, pathwise_config=PathwiseConfig())
    se = report.dual_upper_se
    assert report.dual_upper <= report.hindsight_value + 2 * se
    assert report.dual_upper <= report.dual_lsm_upper + 2 * se
    assert report.dual_upper >= report.lsm_lower - 3 * se
    assert report.irreducible_gap >= -2 * se
    assert report.irreducible_gap == report.hindsight_value - report.dual_upper
    data = report_to_json(report)
    markdown = report_to_markdown(report)
    for method, value in (
        ("dual-upper-cross-fit", report.dual_lsm_upper),
        ("dual-pathwise-cross-fit", report.dual_upper),
    ):
        row = next(row for row in data["rows"] if row["method"] == method)
        assert row["family"] == "bound"
        assert row["value"] == value
        assert method in markdown
    assert data["dual_lsm_upper"] == report.dual_lsm_upper
    assert "Pathwise dual upper:" in markdown
    assert "LSM dual upper:" in markdown
