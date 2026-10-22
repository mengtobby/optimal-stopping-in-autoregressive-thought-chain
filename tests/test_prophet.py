import numpy as np
import pytest

from osc.data.schema import Trace, TraceSet
from osc.stopping.payoff import build_panel, evaluate_stop_times
from osc.stopping.prophet import (
    ProphetError,
    hindsight_prophet,
    independent_dp_thresholds,
    kleinberg_weinberg_threshold,
    samuel_cahn_threshold,
    threshold_stop_times,
    time_threshold_stop_times,
)


def make_panel(values):
    traces = []
    for i, row in enumerate(values):
        h = len(row)
        traces.append(
            Trace(
                str(i),
                tuple([""] + ["x"] * (h - 1)),
                np.array([0] + [1] * (h - 1)),
                tuple([None] + ["a"] * (h - 1)),
                np.asarray(row),
                np.column_stack((row, np.arange(h))),
                True,
                float(row[-1]),
            )
        )
    return build_panel(TraceSet(tuple(traces)), lam=0)


def test_prophet_ties_ragged_and_thresholds():
    panel = make_panel([[0.2, 0.8, 0.8], [0.4]])
    result = hindsight_prophet(panel)
    assert result.stop_times.tolist() == [1, 0]
    assert result.value == pytest.approx(0.6)
    assert threshold_stop_times(panel, 0.9).tolist() == [2, 0]
    assert threshold_stop_times(panel, -np.inf).tolist() == [0, 0]
    np.testing.assert_allclose(independent_dp_thresholds(panel), [0.8, 0.8, -np.inf])
    assert time_threshold_stop_times(panel, independent_dp_thresholds(panel)).tolist() == [1, 0]
    for bad in [np.array([1]), np.array([0, np.nan, 1])]:
        with pytest.raises(ProphetError):
            time_threshold_stop_times(panel, bad)
    with pytest.raises(ProphetError):
        threshold_stop_times(panel, np.nan)


def test_seeded_iid_prophet_inequalities():
    panel = make_panel(np.random.default_rng(12).uniform(size=(3000, 8)))
    prophet = hindsight_prophet(panel).value
    for threshold in [samuel_cahn_threshold(panel), kleinberg_weinberg_threshold(panel)]:
        value = evaluate_stop_times(panel, threshold_stop_times(panel, threshold), "oracle-iid")
        assert 0.5 * prophet <= value.value <= prophet
    rng = np.random.default_rng(3)
    for _ in range(20):
        times = rng.integers(panel.horizons)
        assert evaluate_stop_times(panel, times, "random").value <= prophet


def test_dp_uses_next_marginal():
    panel = make_panel([[0, 0, 0.2], [0, 1, 0.6]])
    np.testing.assert_allclose(independent_dp_thresholds(panel), [0.7, 0.4, -np.inf])
