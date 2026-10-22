import numpy as np
import pytest
from test_flips import make_trace

from osc.data.schema import TraceSet
from osc.pilots.elicit import (
    ElicitationError,
    consistency_stats,
    elicit_remaining,
    parse_int,
    run_elicitation_pilot,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("need 12 steps", 12),
        ("0", 0),
        ("+10000", 10000),
        ("-1 then 2", None),
        ("10001 then 2", None),
        ("none", None),
        ("9" * 5000, None),
        ("0000012", 12),
    ],
)
def test_parse(text, expected):
    assert parse_int(text) == expected


class Backend:
    def __init__(self):
        self.prefixes = []

    def complete(self, prompt, thinking_prefix, probe, max_tokens):
        self.prefixes.append(thinking_prefix)
        return "2" if not thinking_prefix else "unknown"


def test_elicit_and_runner():
    trace = make_trace()
    backend = Backend()
    result = elicit_remaining(backend, "prompt", trace, every=2)
    np.testing.assert_allclose(result, [2, np.nan, np.nan, np.nan], equal_nan=True)
    assert backend.prefixes == ["", "answer 42. " * 2]
    stats = run_elicitation_pilot(Backend(), TraceSet((trace,)), {"p": "prompt"})
    assert stats.n_elicitations == 1
    assert stats.parse_rate == 0.25
    with pytest.raises(ElicitationError):
        run_elicitation_pilot(Backend(), TraceSet((trace,)), {})
    with pytest.raises(ElicitationError):
        run_elicitation_pilot(Backend(), TraceSet((trace,)), {}, every=0)
    with pytest.raises(ElicitationError):
        elicit_remaining(Backend(), "", trace, max_tokens=0)


def test_calibrated_and_receding_plans():
    calibrated = np.array([[4, 3, 2, 1, 0], [2, 1, 0, np.nan, np.nan]])
    stats = consistency_stats(calibrated, [5, 3])
    assert stats.receding_slope == 0
    assert stats.receding_slope_se == 0
    assert stats.mean_bias == 0
    assert stats.spearman == pytest.approx(1)
    assert stats.reversal_rate == 0
    assert stats.parse_rate == 1
    np.testing.assert_array_equal(stats.per_trace_slopes, [0, 0])
    receding = consistency_stats(np.full((2, 5), 1), [5, 5])
    assert receding.receding_slope == 1
    assert receding.reversal_rate == 0.4
    assert np.isnan(receding.spearman)
    missing = consistency_stats([[np.nan, 1, np.nan]], [3])
    assert missing.parse_rate == pytest.approx(1 / 3)
    assert np.isnan(missing.receding_slope)
    assert np.isnan(consistency_stats([[np.nan]], [1]).mean_bias)


@pytest.mark.parametrize(
    "planned,horizons,tolerance",
    [
        ([1], [1], 1),
        ([[1]], [], 1),
        ([[1]], [1.5], 1),
        ([[1]], [2], 1),
        ([[1]], [1], -1),
        ([[np.inf]], [1], 1),
        ([[-1]], [1], 1),
    ],
)
def test_invalid_stats(planned, horizons, tolerance):
    with pytest.raises(ElicitationError):
        consistency_stats(planned, horizons, tolerance)
