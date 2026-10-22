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


def test_censored_exclusions_and_override():
    from dataclasses import replace

    from osc.pilots.elicit import ConsistencyStats

    natural = make_trace((0, 1, 1))
    censored = replace(natural, natural_stop=False)
    capped = make_trace()
    traces = TraceSet((natural, censored, capped))
    backend = Backend()
    stats = run_elicitation_pilot(backend, traces, {"p": "prompt"}, max_steps=3)
    assert (stats.n_traces, stats.n_excluded) == (1, 2)
    assert len(backend.prefixes) == 3
    included = run_elicitation_pilot(
        Backend(), traces, {"p": "prompt"}, max_steps=3, include_censored=True
    )
    assert (included.n_traces, included.n_excluded) == (3, 0)
    excluded = run_elicitation_pilot(Backend(), TraceSet((censored,)), {})
    assert (excluded.n_traces, excluded.n_excluded, excluded.n_elicitations) == (0, 1, 0)
    assert np.isnan(excluded.parse_rate)
    stats = consistency_stats([[2, 1, 0], [0, 0, 0]], [3, 3], censored=[False, True])
    assert stats.mean_bias == 0
    assert stats.n_excluded == 1
    assert stats.per_trace_slopes.shape == (1,)
    assert ConsistencyStats(0, 0, 0, 0, 0, 0, 0, 0, np.empty(0)).n_excluded == 0


@pytest.mark.parametrize("mask", [[True, False], [1], [[True]]])
def test_invalid_censor_mask(mask):
    with pytest.raises(ElicitationError, match="censored"):
        consistency_stats([[0]], [1], censored=mask)


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True])
def test_invalid_max_steps(max_steps):
    with pytest.raises(ElicitationError, match="max_steps"):
        run_elicitation_pilot(Backend(), TraceSet((make_trace(),)), {}, max_steps=max_steps)


def test_collection_limit_boundary_and_metadata_fallback():
    traces = TraceSet(
        (make_trace((0, 1, 1)), make_trace()), metadata={"collect_config": {"max_steps": 3}}
    )
    for max_steps in (None, 3):
        backend = Backend()
        stats = run_elicitation_pilot(backend, traces, {"p": "prompt"}, max_steps=max_steps)
        assert (stats.n_traces, stats.n_excluded) == (1, 1)
        assert len(backend.prefixes) == 3
    overridden = run_elicitation_pilot(Backend(), traces, {"p": "prompt"}, max_steps=4)
    assert (overridden.n_traces, overridden.n_excluded) == (2, 0)


@pytest.mark.parametrize("config", [None, [], {"max_steps": True}, {"max_steps": 0}])
def test_invalid_collection_metadata(config):
    traces = TraceSet((make_trace(),), metadata={"collect_config": config})
    with pytest.raises(ElicitationError):
        run_elicitation_pilot(Backend(), traces, {"p": "prompt"})
