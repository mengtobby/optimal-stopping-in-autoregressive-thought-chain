"""Headroom decomposition and CLI integration checks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from osc.analysis.headroom import (
    HeadroomError,
    headroom_report,
    report_to_json,
    report_to_markdown,
)
from osc.stopping.synthetic import (
    default_model,
    exact_optimal_stop_times,
    exact_value,
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
    from osc.stopping.payoff import bootstrap_mean_ci, build_panel

    model = default_model()
    traces = sample_traceset(model, 300, 8, 8)
    exact_times = exact_optimal_stop_times(model, 0.05, traces)
    report = headroom_report(traces, 0.05, seed=8, exact_stop_times=exact_times)
    panel = build_panel(traces, 0.05)
    times, policies, folds = cross_fit_lsm(panel, seed=8)
    differences = np.empty(len(traces))
    for fold, policy in enumerate(policies):
        indices = np.flatnonzero(folds == fold)
        held = panel.subset(indices)
        dual = dual_upper_bound(held, policy.value)
        differences[indices] = dual.per_trace - held.payoff[np.arange(len(indices)), times[indices]]
        increments = martingale_increments(held, policy.value)
        martingale = np.column_stack(
            (np.zeros(len(indices)), np.cumsum(increments[:, :-1], axis=1))
        )
        assert np.all(
            differences[indices] >= -martingale[np.arange(len(indices)), times[indices]] - 1e-12
        )
    assert report.headroom_paired == pytest.approx(differences.mean())
    assert report.headroom_paired == pytest.approx(report.dual_upper - report.lsm_lower)
    assert report.headroom_paired_ci == bootstrap_mean_ci(differences, seed=8)
    oracle = next(row for row in report.rows if row.method == "exact-optimal")
    assert oracle.family == "oracle"
    assert oracle.value == pytest.approx(panel.payoff[np.arange(len(traces)), exact_times].mean())
    data = report_to_json(report)
    assert data["headroom_paired"] == report.headroom_paired
    assert data["headroom_paired_ci"] == list(report.headroom_paired_ci)
    assert "Paired headroom:" in report_to_markdown(report)
