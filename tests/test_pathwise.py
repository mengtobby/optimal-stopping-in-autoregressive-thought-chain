from types import SimpleNamespace

import numpy as np
import pytest

import osc.stopping.pathwise as pathwise
from osc.stopping.dual import dual_upper_bound
from osc.stopping.lsm import LSMConfig, cross_fit_lsm
from osc.stopping.pathwise import (
    PathwiseBasis,
    PathwiseConfig,
    PathwiseError,
    basis_martingales,
    build_basis,
    cross_fit_pathwise_dual,
    fit_pathwise_beta,
    pathwise_dual,
)
from osc.stopping.payoff import build_panel
from osc.stopping.synthetic import (
    default_model,
    exact_state_value_fn,
    exact_value,
    overthinking_model,
    sample_traceset,
)


@pytest.fixture
def small_panel():
    return build_panel(sample_traceset(default_model(), 40, 8, 7), 0.05)


def test_empty_basis_is_hindsight(small_panel):
    basis = PathwiseBasis((), ())
    beta = fit_pathwise_beta(small_panel, basis)
    assert beta.shape == (0,)
    assert basis_martingales(small_panel, basis).shape == (
        0,
        small_panel.n_traces,
        small_panel.t_max,
    )
    result = pathwise_dual(small_panel, basis, beta)
    np.testing.assert_array_equal(result.per_trace, np.nanmax(small_panel.payoff, axis=1))
    assert result.upper == result.per_trace.mean()
    assert result.mean_abs_increment == 0
    assert result.martingale_coverage == 1
    assert not result.per_trace.flags.writeable


def test_basis_scaling_interactions_and_nonfinite(small_panel):
    config = PathwiseConfig(n_components=0, n_time_bins=1, include_tokens=False)
    basis = build_basis(
        small_panel,
        [
            ("bad", lambda t, x, c: np.full(len(x), -np.inf)),
            ("constant", lambda t, x, c: np.ones(len(x))),
            ("state", lambda t, x, c: x[:, 1]),
        ],
        config,
    )
    assert basis.names == ("state",)
    values = np.concatenate(
        [
            basis.value_fns[0](
                t,
                small_panel.features[small_panel.alive[:, t], t],
                small_panel.cum_tokens[small_panel.alive[:, t], t],
            )
            for t in range(small_panel.t_max)
        ]
    )
    assert values.std() == pytest.approx(1)
    interacted = build_basis(small_panel, config=PathwiseConfig(n_components=None))
    assert any(name.startswith("pca-") for name in interacted.names)
    assert any(name.startswith("tokens:") for name in interacted.names)
    martingales = basis_martingales(small_panel, interacted)
    assert np.all(martingales[:, :, 0] == 0)
    assert np.isfinite(martingales).all()
    single = small_panel.subset([0])
    result = pathwise_dual(single, basis, np.zeros(1))
    assert result.upper_se == 0


@pytest.mark.parametrize(
    ("n_traces", "tolerance"),
    [(400, 0.05), pytest.param(3000, 0.02, marks=pytest.mark.slow)],
)
def test_exact_anchor_cross_fit(n_traces, tolerance):
    model = overthinking_model()
    panel = build_panel(sample_traceset(model, n_traces, 8, 42), 0.05)
    folds = np.arange(panel.n_traces) % 3
    anchor = [("exact", exact_state_value_fn(model, 0.05))]
    result, betas = cross_fit_pathwise_dual(
        panel, folds, [anchor] * 3, PathwiseConfig(0, 1, include_tokens=False)
    )
    truth = exact_value(model, 0.05)
    assert abs(result.upper - truth) < tolerance
    assert result.upper >= truth - 3 * result.upper_se
    assert len(betas) == 3


def _noisy_lsm_anchor_bound(n_traces, k_siblings, lsm_config, pathwise_config=None):
    model = overthinking_model(feature_noise=0.6)
    panel = build_panel(sample_traceset(model, n_traces, k_siblings, 21), 0.05)
    _, policies, folds = cross_fit_lsm(panel, lsm_config, n_folds=3, seed=3)
    anchors = [
        [("value", p.value), ("payoff", p.payoff_estimate), ("continuation", p.continuation)]
        for p in policies
    ]
    result, _ = cross_fit_pathwise_dual(panel, folds, anchors, pathwise_config or PathwiseConfig())
    lsm_values = np.empty(panel.n_traces)
    for f, policy in enumerate(policies):
        indices = np.flatnonzero(folds == f)
        lsm_values[indices] = dual_upper_bound(panel.subset(indices), policy.value).per_trace
    return result, lsm_values, np.nanmax(panel.payoff, axis=1), exact_value(model, 0.05)


def test_small_noisy_lsm_anchors_cross_fit():
    fast_lsm = LSMConfig(n_components=5, margins=(0.0,), ridge_alphas=(10.0,), inner_folds=2)
    fast_pathwise = PathwiseConfig(l1_penalties=(0.0, 0.1), inner_folds=2)
    result, _, hindsight, truth = _noisy_lsm_anchor_bound(240, 2, fast_lsm, fast_pathwise)
    assert result.per_trace.shape == hindsight.shape
    assert np.all(np.isfinite(result.per_trace))
    assert result.upper >= truth - 3 * result.upper_se


@pytest.mark.slow
def test_noisy_lsm_anchors_cross_fit():
    result, lsm_values, hindsight, truth = _noisy_lsm_anchor_bound(
        3000, 8, LSMConfig(n_components=5)
    )
    assert result.upper <= hindsight.mean() + 2 * result.upper_se
    assert result.upper <= lsm_values.mean() + 2 * result.upper_se
    assert result.upper >= truth - 3 * result.upper_se


def test_training_subsets_and_order(small_panel, monkeypatch):
    folds = np.arange(small_panel.n_traces) % 2
    seen = []
    original = pathwise.fit_pathwise_beta

    def spy(panel, basis, config, l1_penalty=0.0):
        seen.append([trace.problem_id for trace in panel.traces])
        return original(panel, basis, config, l1_penalty)

    monkeypatch.setattr(pathwise, "fit_pathwise_beta", spy)
    result, _ = cross_fit_pathwise_dual(
        small_panel, folds, [(), ()], PathwiseConfig(0, 1, include_tokens=False)
    )
    for f, ids in enumerate(seen):
        assert ids == [
            trace.problem_id for i, trace in enumerate(small_panel.traces) if folds[i] != f
        ]
    np.testing.assert_array_equal(result.per_trace, np.nanmax(small_panel.payoff, axis=1))


def test_solver_failure(small_panel, monkeypatch):
    basis = build_basis(small_panel)
    monkeypatch.setattr(
        pathwise,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(success=False, x=None, message="failure"),
    )
    with pytest.raises(PathwiseError, match="LP failed"):
        fit_pathwise_beta(small_panel, basis)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": -1},
        {"n_components": True},
        {"n_time_bins": 0},
        {"beta_bound": np.inf},
        {"include_tokens": 1},
        {"l1_penalties": ()},
        {"l1_penalties": (-1,)},
        {"inner_folds": 1},
        {"inner_folds": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(PathwiseError):
        PathwiseConfig(**kwargs)


def test_invalid_inputs(small_panel):
    with pytest.raises(PathwiseError):
        PathwiseBasis(("x",), ())
    with pytest.raises(PathwiseError):
        build_basis(small_panel, [("x", lambda t, x, c: np.ones((len(x), 1)))])
    with pytest.raises(PathwiseError):
        build_basis(small_panel, [("x", None)])
    with pytest.raises(PathwiseError):
        pathwise_dual(small_panel, PathwiseBasis((), ()), np.ones(1))
    with pytest.raises(PathwiseError):
        cross_fit_pathwise_dual(small_panel, np.zeros(small_panel.n_traces), [(), ()])


def test_no_transitions():
    from dataclasses import replace

    panel = build_panel(sample_traceset(replace(default_model(), t_max=1), 2, 0, 1), 0.05)
    basis = build_basis(panel)
    result = pathwise_dual(panel, basis, fit_pathwise_beta(panel, basis))
    assert result.upper == 0.2
    assert result.mean_abs_increment == result.martingale_coverage == 0


def test_large_l1_penalty_is_hindsight(small_panel):
    basis = build_basis(small_panel)
    beta = fit_pathwise_beta(small_panel, basis, l1_penalty=1e6)
    np.testing.assert_array_equal(beta, np.zeros(len(basis.value_fns)))
    np.testing.assert_array_equal(
        pathwise_dual(small_panel, basis, beta).per_trace,
        np.nanmax(small_panel.payoff, axis=1),
    )
    for penalty in [-1, np.inf, np.nan]:
        with pytest.raises(PathwiseError):
            fit_pathwise_beta(small_panel, basis, l1_penalty=penalty)


def test_regularized_noisy_small_panel():
    from osc.stopping.lsm import fit_lsm

    panel = build_panel(sample_traceset(overthinking_model(feature_noise=0.6), 120, 2, 21), 0.05)
    folds = np.arange(panel.n_traces) % 3
    anchors = [
        [("value", fit_lsm(panel.subset(np.flatnonzero(folds != f))).value)] for f in range(3)
    ]
    result, _ = cross_fit_pathwise_dual(panel, folds, anchors)
    hindsight = np.nanmax(panel.payoff, axis=1).mean()
    assert result.upper <= hindsight + 2 * result.upper_se
    assert result.upper <= 1 + 2 * result.upper_se
    old, _ = cross_fit_pathwise_dual(
        panel, folds, anchors, PathwiseConfig(beta_bound=10, l1_penalties=(0.0,))
    )
    assert old.upper > hindsight + 2 * old.upper_se


def test_group_validation(small_panel):
    from dataclasses import replace

    from osc.data.schema import TraceSet

    traces = list(small_panel.traces)
    traces[1] = replace(traces[1], problem_id=traces[0].problem_id)
    panel = build_panel(TraceSet(tuple(traces)), 0.05)
    with pytest.raises(PathwiseError, match="problem"):
        cross_fit_pathwise_dual(panel, np.arange(panel.n_traces) % 2, [(), ()])
    with pytest.raises(PathwiseError, match="contiguous"):
        cross_fit_pathwise_dual(panel, np.arange(panel.n_traces) % 2 * 2, [(), ()])


def test_penalty_selection_uses_only_outer_training(small_panel, monkeypatch):
    folds = np.arange(small_panel.n_traces) % 2
    original_build = pathwise.build_basis
    original_fit = pathwise._fit_beta
    builds, fits = [], []

    def build(train, anchors, config):
        builds.append({trace.problem_id for trace in train.traces})
        return original_build(train, anchors, config)

    def fit(train, martingales, config, penalty):
        fits.append(({trace.problem_id for trace in train.traces}, penalty))
        return original_fit(train, martingales, config, penalty)

    monkeypatch.setattr(pathwise, "build_basis", build)
    monkeypatch.setattr(pathwise, "_fit_beta", fit)
    config = PathwiseConfig(n_components=1, n_time_bins=1, l1_penalties=(0.0, 1.0))
    cross_fit_pathwise_dual(small_panel, folds, [(), ()], config)
    per_fold_builds, per_fold_fits = 1 + config.inner_folds, 2 * config.inner_folds + 1
    assert len(builds) == 2 * per_fold_builds and len(fits) == 2 * per_fold_fits
    for f in range(2):
        expected = {trace.problem_id for i, trace in enumerate(small_panel.traces) if folds[i] != f}
        fold_builds = builds[f * per_fold_builds : (f + 1) * per_fold_builds]
        fold_fits = fits[f * per_fold_fits : (f + 1) * per_fold_fits]
        # Outer basis first, then one basis per inner fold rebuilt without its validation.
        assert fold_builds[0] == expected
        assert all(ids < expected for ids in fold_builds[1:])
        assert fold_fits[-1][0] == expected
        assert all(ids < expected for ids, _ in fold_fits[:-1])


def test_last_input_cache_reuses_only_identical_arrays():
    calls = []
    cached = pathwise._last_input_cache(lambda x: calls.append(1) or x * 2)
    first, second = np.ones((2, 3)), np.ones((2, 3))
    assert cached(first) is cached(first)
    np.testing.assert_array_equal(cached(second), second * 2)
    cached(first)
    assert len(calls) == 3
