import numpy as np
import pytest

from osc.stopping.dual import (
    DualError,
    cross_fit_shrunk_dual,
    dual_upper_bound,
    martingale_increments,
    scaled_value_fn,
    zero_value_fn,
)
from osc.stopping.payoff import build_panel
from osc.stopping.synthetic import (
    default_model,
    exact_state_value_fn,
    exact_value,
    sample_traceset,
)


@pytest.fixture(scope="module")
def panel():
    return build_panel(sample_traceset(default_model(), 6000, 8, 42), 0.05)


def test_zero_recovers_prophet(panel):
    result = dual_upper_bound(panel, zero_value_fn)
    assert result.upper == np.nanmax(panel.payoff, axis=1).mean()
    assert result.upper >= exact_value(default_model(), 0.05)
    assert result.martingale_coverage == 1
    assert result.mean_abs_increment == 0
    with pytest.raises(ValueError):
        result.per_trace[0] = 0


def test_exact_dual_and_centered_increments(panel):
    model = default_model()
    fn = exact_state_value_fn(model, 0.05)
    result = dual_upper_bound(panel, fn)
    truth = exact_value(model, 0.05)
    assert result.upper >= truth - 3 * result.upper_se
    assert abs(result.upper - truth) < 0.03
    increments = martingale_increments(panel, fn)
    for t in range(panel.t_max - 1):
        values = increments[panel.horizons > t + 1, t]
        se = values.std(ddof=1) / np.sqrt(len(values))
        assert abs(values.mean()) <= 3 * se + 1e-12
    assert np.all(increments[~panel.alive] == 0)
    assert result.mean_abs_increment > 0


def test_missing_siblings_and_singleton():
    traces = sample_traceset(default_model(), 10, 0, 2)
    panel = build_panel(traces, 0.05)
    result = dual_upper_bound(panel, exact_state_value_fn(default_model(), 0.05))
    assert result.martingale_coverage == 0
    assert result.upper == np.nanmax(panel.payoff, axis=1).mean()
    partial = build_panel(sample_traceset(default_model(), 20, 2, 1, 2), 0.05)
    assert 0 < dual_upper_bound(partial, zero_value_fn).martingale_coverage < 1
    singleton = build_panel(traces.subset([0]), 0.05)
    assert dual_upper_bound(singleton, zero_value_fn).upper_se == 0


def test_bad_value_output(panel):
    for fn in [lambda t, f, c: np.nan * np.ones(len(f)), lambda t, f, c: np.zeros((len(f), 1))]:
        with pytest.raises(DualError):
            martingale_increments(panel, fn)


def test_no_transitions():
    from dataclasses import replace

    model = replace(default_model(), t_max=1)
    result = dual_upper_bound(build_panel(sample_traceset(model, 1, 8, 0), 0.05), zero_value_fn)
    assert result.upper == 0.2
    assert result.martingale_coverage == result.mean_abs_increment == 0


def test_shrinkage_noisy_panel_and_training_isolation():
    from osc.stopping.lsm import LSMConfig, fit_lsm
    from osc.stopping.synthetic import overthinking_model

    panel = build_panel(sample_traceset(overthinking_model(feature_noise=0.6), 120, 2, 21), 0.05)
    folds = np.arange(panel.n_traces) % 3
    seen = []

    def fit(train):
        seen.append({trace.problem_id for trace in train.traces})
        return fit_lsm(train, LSMConfig(n_components=5)).value

    functions = [fit(panel.subset(np.flatnonzero(folds != f))) for f in range(3)]
    seen.clear()
    result, scales = cross_fit_shrunk_dual(panel, folds, functions, fit)
    for f in range(3):
        held_out = {trace.problem_id for i, trace in enumerate(panel.traces) if folds[i] == f}
        assert all(not held_out.intersection(ids) for ids in seen[3 * f : 3 * f + 3])
    hindsight = np.nanmax(panel.payoff, axis=1)
    assert len(scales) == 3
    assert result.upper <= hindsight.mean() + 2 * result.upper_se
    assert result.upper <= 1 + 2 * result.upper_se
    old, _ = cross_fit_shrunk_dual(panel, folds, functions, fit, grid=(1.0,))
    assert old.upper > hindsight.mean() + 2 * old.upper_se
    zero, scales = cross_fit_shrunk_dual(panel, folds, functions, fit, grid=(0.0,))
    np.testing.assert_array_equal(zero.per_trace, hindsight)
    assert scales == (0.0, 0.0, 0.0)


def test_shrink_validation_and_small_training_fallback():
    from dataclasses import replace

    from osc.data.schema import TraceSet

    panel = build_panel(sample_traceset(default_model(), 4, 2, 7), 0.05)
    folds = np.arange(4) % 2

    def unexpected_fit(train):
        raise AssertionError("too few training problems must use zero")

    result, scales = cross_fit_shrunk_dual(panel, folds, [zero_value_fn] * 2, unexpected_fit)
    np.testing.assert_array_equal(result.per_trace, np.nanmax(panel.payoff, axis=1))
    assert scales == (0.0, 0.0)
    for bad in [np.zeros(4), np.array([0, 0, 2, 2]), np.zeros(3, dtype=int)]:
        with pytest.raises(DualError):
            cross_fit_shrunk_dual(panel, bad, [zero_value_fn] * 2, unexpected_fit)
    for kwargs in [{"grid": ()}, {"grid": (-1,)}, {"inner_folds": 1}]:
        with pytest.raises(DualError):
            cross_fit_shrunk_dual(panel, folds, [zero_value_fn] * 2, unexpected_fit, **kwargs)
    traces = list(panel.traces)
    traces[1] = replace(traces[1], problem_id=traces[0].problem_id)
    grouped = build_panel(TraceSet(tuple(traces)), 0.05)
    with pytest.raises(DualError, match="problem"):
        cross_fit_shrunk_dual(grouped, folds, [zero_value_fn] * 2, unexpected_fit)
    for c in [-1, np.inf, np.nan]:
        with pytest.raises(DualError):
            scaled_value_fn(zero_value_fn, c)
    fn = scaled_value_fn(lambda t, x, c: x[:, 0], 0.25)
    np.testing.assert_array_equal(fn(0, np.ones((2, 1)), np.zeros(2)), [0.25, 0.25])
    zero = scaled_value_fn(lambda *args: (_ for _ in ()).throw(AssertionError()), 0)
    np.testing.assert_array_equal(zero(0, np.ones((2, 1)), np.zeros(2)), [0, 0])


def test_per_fold_refitters_are_used_for_their_own_fold():
    from osc.stopping.synthetic import overthinking_model

    panel = build_panel(sample_traceset(overthinking_model(), 30, 2, 9), 0.05)
    folds = np.arange(panel.n_traces) % 2
    calls = []

    def refitter(f):
        def fit(train):
            calls.append(f)
            return zero_value_fn

        return fit

    _, scales = cross_fit_shrunk_dual(panel, folds, [zero_value_fn] * 2, [refitter(0), refitter(1)])
    assert calls == [0, 0, 0, 1, 1, 1]
    assert len(scales) == 2
    for bad in [[refitter(0)], [refitter(0), "not callable"]]:
        with pytest.raises(DualError, match="one callable per fold"):
            cross_fit_shrunk_dual(panel, folds, [zero_value_fn] * 2, bad)
