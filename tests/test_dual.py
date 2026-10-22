import numpy as np
import pytest

from osc.stopping.dual import (
    DualError,
    dual_upper_bound,
    martingale_increments,
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
