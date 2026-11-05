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
    paired_comparison,
    report_to_json,
    report_to_markdown,
)
from osc.stopping.lsm import LSMConfig
from osc.stopping.pathwise import PathwiseConfig, cross_fit_pathwise_dual
from osc.stopping.synthetic import (
    default_model,
    exact_optimal_stop_times,
    exact_value,
    overthinking_model,
    sample_traceset,
)

# Small grids keep the fast tests fast; the slow tests use the production defaults.
FAST_LSM = LSMConfig(margins=(0.0, float("inf")), ridge_alphas=(10.0,), inner_folds=2)
FAST_PATHWISE = PathwiseConfig(l1_penalties=(0.0, 0.1), inner_folds=2)


def _check_rendering(report, n_traces):
    assert report.hindsight_value >= report.dual_upper - 1e-9
    markdown = report_to_markdown(report)
    assert all(row.method in markdown for row in report.rows)
    data = json.loads(json.dumps(report_to_json(report), allow_nan=False))
    assert data["n_traces"] == n_traces
    assert data["rows"][-1]["accuracy"] is None
    best = max(r.value for r in report.rows if r.family in ("heuristic", "learned"))
    assert report.headroom == report.upper_bound - best
    assert len(report.problem_ids) == n_traces
    for row in report.rows:
        assert len(row.payoffs) == n_traces
        assert np.isclose(np.mean(row.payoffs), row.value)
        assert len(row.stop_times) == (0 if row.family == "bound" else n_traces)
        assert all(type(value) is float for value in row.payoffs)
        assert all(type(time) is int for time in row.stop_times)
    comparison = paired_comparison(report, "full-trace", "full-trace")
    assert comparison.mean_difference == 0
    assert comparison.ci == (0, 0)
    assert paired_comparison(data, "full-trace", "full-trace") == comparison
    assert paired_comparison(data, "correctness-probe-cross-fit", "full-trace") == (
        paired_comparison(report, "correctness-probe-cross-fit", "full-trace")
    )
    assert data["problem_ids"] == list(report.problem_ids)
    assert "## Paired differences vs full-trace" in markdown
    assert "| correctness-probe-cross-fit |" in markdown
    legacy = replace(report, rows=tuple(replace(row, payoffs=()) for row in report.rows))
    assert "## Paired differences vs full-trace" not in report_to_markdown(legacy)


def test_small_bounds_and_rendering():
    traces = sample_traceset(default_model(), 200, 2, 0)
    report = headroom_report(traces, 0.05, lsm_config=FAST_LSM, pathwise_config=FAST_PATHWISE)
    _check_rendering(report, 200)


def test_paired_comparison_validation_and_clusters():
    from osc.stopping.stats import cluster_bootstrap_mean_ci

    data = {
        "n_traces": 3,
        "problem_ids": ["a", "a", "b"],
        "rows": [
            {"method": "a", "payoffs": [1.0, 2.0, 6.0]},
            {"method": "b", "payoffs": [0.0, 1.0, 2.0]},
        ],
    }
    comparison = paired_comparison(data, "a", "b", n_boot=100, seed=7)
    assert comparison.mean_difference == 2
    assert comparison.n_traces == 3
    assert comparison.ci == cluster_bootstrap_mean_ci(
        np.array([1.0, 1.0, 4.0]), data["problem_ids"], n_boot=100, seed=7
    )
    with pytest.raises(HeadroomError, match="unknown method"):
        paired_comparison(data, "missing", "b")
    for update, message in (
        ({"problem_ids": []}, "missing problem_ids"),
        ({"problem_ids": ["a"]}, "length mismatch"),
        ({"n_traces": 4}, "length mismatch"),
        ({"rows": [{"method": "a"}]}, "no payoffs"),
        ({"rows": [{"method": "a", "payoffs": [None, 1, 2]}]}, "finite"),
        ({"rows": [{"method": "a", "payoffs": ["bad", 1, 2]}]}, "invalid payoffs"),
    ):
        with pytest.raises(HeadroomError, match=message):
            paired_comparison({**data, **update}, "a", "b")


def test_signal_policies_preserve_plain_duals():
    from osc.stopping.dual import cross_fit_shrunk_dual
    from osc.stopping.lsm import cross_fit_lsm, fit_lsm
    from osc.stopping.payoff import build_panel
    from osc.stopping.signals import with_signals

    traces = sample_traceset(default_model(), 30, 2, 11)
    report = headroom_report(
        traces, 0.05, n_folds=2, seed=11, lsm_config=FAST_LSM, pathwise_config=FAST_PATHWISE
    )
    panel = build_panel(traces, 0.05)
    _, names = with_signals(panel)
    assert report.signal_names == names
    assert names
    for method in ("LSM-signals-cross-fit", "correctness-probe-signals-cross-fit"):
        row = next(row for row in report.rows if row.method == method)
        assert row.family == "learned"
        assert np.isfinite([row.value, *row.value_ci, row.accuracy, row.mean_tokens]).all()
    times, policies, folds = cross_fit_lsm(panel, FAST_LSM, 2, 11)
    assert report.lsm_lower == pytest.approx(panel.payoff[np.arange(len(traces)), times].mean())
    dual, shrink = cross_fit_shrunk_dual(
        panel,
        folds,
        [policy.value for policy in policies],
        fit_value_fn=[
            lambda train, alpha=policy.payoff_model.alpha: (
                fit_lsm(train, replace(FAST_LSM, ridge_alpha=alpha)).value
            )
            for policy in policies
        ],
        seed=11,
    )
    assert report.dual_lsm_upper == pytest.approx(dual.per_trace.mean())
    assert report.lsm_shrink == shrink
    anchors = [
        [
            ("lsm_value", policy.value),
            ("lsm_payoff", policy.payoff_estimate),
            ("lsm_continuation", policy.continuation),
        ]
        for policy in policies
    ]
    pathwise, _ = cross_fit_pathwise_dual(panel, folds, anchors, FAST_PATHWISE)
    assert report.dual_upper == pytest.approx(pathwise.per_trace.mean())
    best = max(row.value for row in report.rows if row.family in ("heuristic", "learned"))
    assert report.headroom == report.upper_bound - best
    data = json.loads(json.dumps(report_to_json(report), allow_nan=False))
    assert data["signal_names"] == list(names)
    markdown = report_to_markdown(report)
    assert all(name in markdown for name in names)
    assert "sibling states carry no answers" in markdown


def test_signal_columns_survive_nested_fits(monkeypatch):
    import osc.analysis.headroom as headroom
    import osc.stopping.lsm as lsm
    from osc.stopping.payoff import build_panel

    traces = sample_traceset(default_model(), 16, 0, 19)
    traces = replace(
        traces,
        traces=tuple(replace(trace, features=trace.features[:, :1]) for trace in traces),
    )
    panel = build_panel(traces, 0.05)
    sig_panel, names = headroom._with_policy_signals(panel)
    expected_dim = 1 + len(names)
    indices = [0, 2, 5]
    subset = sig_panel.subset(indices)
    np.testing.assert_array_equal(subset.features, sig_panel.features[indices, : subset.t_max])
    nested = subset.subset([1])
    np.testing.assert_array_equal(nested.features, subset.features[[1], : nested.t_max])
    assert nested.features.shape[-1] == expected_dim
    assert not nested.features.flags.writeable
    assert nested.t_max == nested.traces.max_steps
    fitted_lsm_dims = []
    fitted_probe_dims = []
    original_lsm = lsm.fit_lsm
    original_probe = headroom.fit_correctness_probe

    def checked_lsm(train, config):
        policy = original_lsm(train, config)
        if config.passthrough:
            fitted_lsm_dims.append(policy.feature_dim)
            assert config.passthrough == len(names)
        return policy

    def checked_probe(train, **kwargs):
        probe = original_probe(train, **kwargs)
        if kwargs.get("passthrough"):
            fitted_probe_dims.append(probe.feature_dim)
            assert kwargs["passthrough"] == len(names)
        return probe

    monkeypatch.setattr(lsm, "fit_lsm", checked_lsm)
    monkeypatch.setattr(headroom, "fit_correctness_probe", checked_probe)
    report = headroom_report(traces, 0.05, n_folds=2, lsm_config=FAST_LSM)
    assert report.signal_names == names
    # Two outer fits and their inner fits must all retain the appended columns.
    assert len(fitted_lsm_dims) > 2
    assert set(fitted_lsm_dims) == {expected_dim}
    assert fitted_probe_dims == [expected_dim, expected_dim]


@pytest.mark.slow
def test_synthetic_bounds_and_rendering():
    model = default_model()
    report = headroom_report(sample_traceset(model, 1500, 4, 0), 0.05)
    _check_rendering(report, 1500)
    assert report.dual_upper >= report.lsm_lower - 3 * report.dual_upper_se
    assert report.dual_upper >= exact_value(model, 0.05) - 3 * report.dual_upper_se


def test_no_siblings_and_validation():
    traces = sample_traceset(default_model(), 30, 0, 3)
    report = headroom_report(traces, 0.05, lsm_config=FAST_LSM)
    assert np.isnan(report.headroom_paired)
    assert report_to_json(report)["headroom_paired_ci"] == [None, None]
    assert np.isnan(report.dual_upper)
    assert report.upper_bound == report.hindsight_value
    assert report.upper_bound_source == "hindsight-prophet"
    assert report.lsm_shrink == ()
    assert report.irreducible_gap == 0.0
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


def test_cli(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "headroom.py"
    spec = importlib.util.spec_from_file_location("headroom_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fast_report(*args, **kwargs):
        config = kwargs["pathwise_config"]
        return headroom_report(
            *args,
            **{
                **kwargs,
                "lsm_config": FAST_LSM,
                "pathwise_config": replace(
                    config, l1_penalties=FAST_PATHWISE.l1_penalties, inner_folds=2
                ),
            },
        )

    monkeypatch.setattr(module, "headroom_report", fast_report)
    assert module.main(["--synthetic", "120", "--lam", "0.05", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "report_lam0.05.json").read_text())
    assert data["exact_value"] == exact_value(default_model(), 0.05)
    assert (tmp_path / "report_lam0.05.md").is_file()
    assert any(row["method"] == "exact-optimal" for row in data["rows"])
    for noise in ("0", "0.6"):
        assert (
            module.main(
                [
                    "--synthetic",
                    "60",
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
    from osc.stopping.dual import cross_fit_shrunk_dual, martingale_increments, scaled_value_fn
    from osc.stopping.lsm import cross_fit_lsm, fit_lsm
    from osc.stopping.payoff import build_panel
    from osc.stopping.stats import cluster_bootstrap_mean_ci, cluster_standard_error

    model = default_model()
    traces = sample_traceset(model, 150, 8, 8)
    traces = replace(
        traces,
        traces=tuple(
            replace(trace, problem_id=f"problem-{i // 3}") for i, trace in enumerate(traces)
        ),
    )
    clusters = [trace.problem_id for trace in traces]
    exact_times = exact_optimal_stop_times(model, 0.05, traces)
    report = headroom_report(
        traces,
        0.05,
        seed=8,
        exact_stop_times=exact_times,
        lsm_config=FAST_LSM,
        pathwise_config=FAST_PATHWISE,
    )
    panel = build_panel(traces, 0.05)
    times, policies, folds = cross_fit_lsm(panel, FAST_LSM, seed=8)
    differences = np.empty(len(traces))
    shrunk, shrink = cross_fit_shrunk_dual(
        panel,
        folds,
        [p.value for p in policies],
        fit_value_fn=lambda train: fit_lsm(train, FAST_LSM).value,
        seed=8,
    )
    assert report.lsm_shrink == shrink
    dual_values = shrunk.per_trace
    for fold, policy in enumerate(policies):
        indices = np.flatnonzero(folds == fold)
        held = panel.subset(indices)
        differences[indices] = (
            dual_values[indices] - held.payoff[np.arange(len(indices)), times[indices]]
        )
        increments = martingale_increments(held, scaled_value_fn(policy.value, shrink[fold]))
        martingale = np.column_stack(
            (np.zeros(len(indices)), np.cumsum(increments[:, :-1], axis=1))
        )
        assert np.all(
            differences[indices] >= -martingale[np.arange(len(indices)), times[indices]] - 1e-12
        )
    assert report.dual_lsm_upper == pytest.approx(dual_values.mean())
    assert report.dual_lsm_upper_se == cluster_standard_error(dual_values, clusters)
    lsm_row = next(row for row in report.rows if row.method == "dual-lsm-shrunk-cross-fit")
    assert lsm_row.value_ci == cluster_bootstrap_mean_ci(dual_values, clusters, seed=8)
    anchors = [
        [
            ("lsm_value", policy.value),
            ("lsm_payoff", policy.payoff_estimate),
            ("lsm_continuation", policy.continuation),
        ]
        for policy in policies
    ]
    result, _ = cross_fit_pathwise_dual(panel, folds, anchors, FAST_PATHWISE)
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


def _check_headline(report):
    assert report.upper_bound <= report.hindsight_value
    assert report.upper_bound == min(
        report.hindsight_value, report.dual_upper, report.dual_lsm_upper
    )
    assert len(report.lsm_shrink) == 5
    assert report.upper_bound == next(
        row.value for row in report.rows if row.method == report.upper_bound_source
    )
    assert report.irreducible_gap == report.hindsight_value - report.upper_bound
    data = report_to_json(report)
    json.dumps(data, allow_nan=False)
    assert data["lsm_shrink"] == list(report.lsm_shrink)
    assert data["upper_bound"] == report.upper_bound
    assert data["upper_bound_source"] == report.upper_bound_source
    markdown = report_to_markdown(report)
    assert f"Source: {report.upper_bound_source}." in markdown
    assert "LSM shrink factors by fold:" in markdown
    for method, value in (
        ("dual-lsm-shrunk-cross-fit", report.dual_lsm_upper),
        ("dual-pathwise-l1-cross-fit", report.dual_upper),
    ):
        row = next(row for row in data["rows"] if row["method"] == method)
        assert row["family"] == "bound"
        assert row["value"] == value
        assert method in markdown
    assert data["dual_lsm_upper"] == report.dual_lsm_upper
    assert "Pathwise dual upper:" in markdown
    assert "LSM dual upper:" in markdown


def test_small_noisy_headline():
    traces = sample_traceset(overthinking_model(feature_noise=0.6), 200, 2, 0)
    report = headroom_report(traces, 0.05, lsm_config=FAST_LSM, pathwise_config=FAST_PATHWISE)
    _check_headline(report)


@pytest.mark.slow
def test_noisy_pathwise_headline():
    traces = sample_traceset(overthinking_model(feature_noise=0.6), 2000, 8, 0)
    report = headroom_report(traces, 0.05, pathwise_config=PathwiseConfig())
    _check_headline(report)
    se = report.dual_upper_se
    assert report.dual_upper <= report.hindsight_value + 2 * se
    assert report.dual_upper <= report.dual_lsm_upper + 2 * se
    assert report.dual_upper >= report.lsm_lower - 3 * se
    assert report.irreducible_gap >= -2 * se
