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


def test_exact_anchor_cross_fit():
    model = overthinking_model()
    panel = build_panel(sample_traceset(model, 3000, 8, 42), 0.05)
    folds = np.arange(panel.n_traces) % 3
    anchor = [("exact", exact_state_value_fn(model, 0.05))]
    result, betas = cross_fit_pathwise_dual(
        panel, folds, [anchor] * 3, PathwiseConfig(0, 1, include_tokens=False)
    )
    truth = exact_value(model, 0.05)
    assert abs(result.upper - truth) < 0.02
    assert result.upper >= truth - 3 * result.upper_se
    assert len(betas) == 3


def test_noisy_lsm_anchors_cross_fit():
    model = overthinking_model(feature_noise=0.6)
    panel = build_panel(sample_traceset(model, 3000, 8, 21), 0.05)
    _, policies, folds = cross_fit_lsm(panel, LSMConfig(n_components=5), n_folds=3, seed=3)
    anchors = [
        [("value", p.value), ("payoff", p.payoff_estimate), ("continuation", p.continuation)]
        for p in policies
    ]
    result, _ = cross_fit_pathwise_dual(panel, folds, anchors)
    lsm_values = np.empty(panel.n_traces)
    for f, policy in enumerate(policies):
        indices = np.flatnonzero(folds == f)
        lsm_values[indices] = dual_upper_bound(panel.subset(indices), policy.value).per_trace
    assert result.upper <= np.nanmax(panel.payoff, axis=1).mean() + 2 * result.upper_se
    assert result.upper <= lsm_values.mean() + 2 * result.upper_se
    assert result.upper >= exact_value(model, 0.05) - 3 * result.upper_se


def test_training_subsets_and_order(small_panel, monkeypatch):
    folds = np.arange(small_panel.n_traces) % 2
    seen = []
    original = pathwise.fit_pathwise_beta

    def spy(panel, basis, config):
        seen.append([trace.problem_id for trace in panel.traces])
        return original(panel, basis, config)

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
