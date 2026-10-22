from dataclasses import replace

import numpy as np
import pytest

from osc.stopping.payoff import build_panel
from osc.stopping.synthetic import (
    SyntheticError,
    default_model,
    exact_optimal_stop_times,
    exact_state_value_fn,
    exact_value,
    exact_value_function,
    overthinking_model,
    sample_traceset,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"transition": np.ones((4, 4))},
        {"transition": np.eye(3)},
        {"accuracy": [-0.1, 0, 0, 0]},
        {"accuracy": []},
        {"accuracy": [[0, 0, 0, 0]]},
        {"stop_prob": [0, 0, 0, 1.1]},
        {"initial": [0, 0, 0, 0]},
        {"initial": [1]},
        {"feature_noise": np.nan},
        {"feature_noise": -1},
        {"tokens_per_step": 0},
        {"t_max": 1.5},
    ],
)
def test_model_validation(changes):
    with pytest.raises(SyntheticError):
        replace(default_model(), **changes)


def test_sampling_and_optimal_rollout():
    model = default_model()
    traces = sample_traceset(model, 6000, 0, 12)
    panel = build_panel(traces, 0.05)
    values = exact_value_function(model, 0.05)
    outcomes = []
    for trace in traces:
        for t in range(trace.n_steps):
            state = trace.features[t].argmax()
            payoff = model.accuracy[state] - 0.01 * t
            if t == trace.n_steps - 1 or payoff >= model.transition[state] @ values[t + 1]:
                outcomes.append(payoff)
                break
        assert trace.step_texts[0] == ""
        assert trace.step_tokens[0] == 0
        assert np.all(trace.step_tokens[1:] == 200)
    outcomes = np.array(outcomes)
    truth = exact_value(model, 0.05)
    assert abs(outcomes.mean() - truth) < 3 * outcomes.std(ddof=1) / np.sqrt(len(outcomes))
    assert truth > panel.payoff[:, 0].mean() + 0.05
    assert truth > np.mean([panel.payoff[i, h - 1] for i, h in enumerate(panel.horizons)]) + 0.05
    with pytest.raises(ValueError):
        model.accuracy[0] = 1


def test_sampling_siblings_noise_and_termination():
    model = replace(default_model(), feature_noise=0.01)
    a = sample_traceset(model, 20, 3, 8, sibling_stride=2)
    b = sample_traceset(model, 20, 3, 8, sibling_stride=2)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.features, y.features)
        for t, sibling in enumerate(x.siblings):
            assert (sibling is not None) == (t < x.n_steps - 1 and t % 2 == 0)
            if sibling is not None:
                assert sibling.k == 3
                assert np.all(sibling.step_tokens == 200)
    terminal = replace(model, stop_prob=np.ones(4))
    assert sample_traceset(terminal, 1, 2, 0)[0].natural_stop
    capped = replace(model, t_max=1, stop_prob=np.zeros(4))
    assert not sample_traceset(capped, 1, 2, 0)[0].natural_stop
    np.testing.assert_allclose(exact_value_function(capped, 0.05), [capped.accuracy])
    for args in [(0, 2, 0, 1), (1, -1, 0, 1), (1, 2, -1, 1), (1, 2, 0, 0)]:
        with pytest.raises(SyntheticError):
            sample_traceset(model, *args)


def test_exact_values_validation_and_decoder():
    model = default_model()
    for lam, scale in [(-1, 1000), (np.nan, 1000), (0, 0), (0, np.inf)]:
        with pytest.raises(SyntheticError):
            exact_value_function(model, lam, scale)
    fn = exact_state_value_fn(model, 0.05)
    np.testing.assert_allclose(fn(0, np.eye(4), np.zeros(4)), exact_value_function(model, 0.05)[0])
    for t, features, tokens in [
        (-1, np.eye(4), np.zeros(4)),
        (0, np.ones((1, 2)), np.zeros(1)),
        (0, np.eye(4), np.zeros(1)),
        (0, np.full((1, 4), np.nan), np.zeros(1)),
    ]:
        with pytest.raises(SyntheticError):
            fn(t, features, tokens)


@pytest.mark.parametrize("noise", [0.0, 0.6])
def test_overthinking_state_dependence_and_partial_observability(noise):
    from osc.stopping.baselines import answer_convergence_stop_times
    from osc.stopping.dual import dual_upper_bound
    from osc.stopping.lsm import cross_fit_lsm

    model = overthinking_model(noise)
    traces = sample_traceset(model, 3000, 8, 42)
    panel = build_panel(traces, 0.05)
    rows = np.arange(len(traces))
    times, policies, folds = cross_fit_lsm(panel)
    learned = panel.payoff[rows, times].mean()
    exact = exact_value(model, 0.05)
    dual = np.empty(len(traces))
    for fold, policy in enumerate(policies):
        indices = np.flatnonzero(folds == fold)
        dual[indices] = dual_upper_bound(panel.subset(indices), policy.value).per_trace
    assert dual.mean() >= exact - 3 * dual.std(ddof=1) / np.sqrt(len(traces))
    assert panel.payoff[rows, panel.horizons - 1].mean() < learned
    for k in (2, 3, 4):
        assert panel.payoff[rows, answer_convergence_stop_times(panel, k)].mean() < learned
    if noise == 0:
        optimal = exact_optimal_stop_times(model, 0.05, traces)
        outcomes = panel.payoff[rows, optimal]
        assert abs(outcomes.mean() - exact) < 3 * outcomes.std(ddof=1) / np.sqrt(len(traces))
        assert optimal.std() >= 1.0
        fixed = max(
            panel.payoff[rows, np.minimum(step, panel.horizons - 1)].mean()
            for step in range(model.t_max)
        )
        assert exact >= fixed + 0.03
        assert exact >= panel.payoff[rows, panel.horizons - 1].mean() + 0.05
        assert abs(learned - exact) < 0.03
    else:
        assert learned <= exact - 0.01
        with pytest.raises(SyntheticError, match="feature_noise"):
            exact_optimal_stop_times(model, 0.05, traces)


def test_exact_stop_times_validation_and_observables():
    from osc.data.schema import TraceSet

    model = default_model()
    traces = sample_traceset(model, 20, 0, 7)
    changed = TraceSet(tuple(replace(t, forced_correct=1 - t.forced_correct) for t in traces))
    np.testing.assert_array_equal(
        exact_optimal_stop_times(model, 0.05, traces),
        exact_optimal_stop_times(model, 0.05, changed),
    )
    for invalid_model in (replace(model, t_max=1), overthinking_model()):
        with pytest.raises(SyntheticError, match="dimensions"):
            exact_optimal_stop_times(invalid_model, 0.05, traces)
    noisy = sample_traceset(replace(model, feature_noise=0.01), 10, 0, 7)
    with pytest.raises(SyntheticError, match="one-hot"):
        exact_optimal_stop_times(model, 0.05, noisy)
