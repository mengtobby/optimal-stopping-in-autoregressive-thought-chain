import numpy as np
import pytest
from test_flips import make_trace

from osc.data.schema import TraceSet
from osc.pilots.elicit import (
    ElicitationError,
    consistency_stats,
    elicit_remaining,
    elicit_remaining_distribution,
    parse_int,
    parse_leading_int,
    run_elicitation_pilot,
    run_elicitation_pilot_detailed,
)


def run_elicitation_pilot_greedy(*args, **kwargs):
    return run_elicitation_pilot(*args, method="greedy", **kwargs)


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
    stats = run_elicitation_pilot_greedy(Backend(), TraceSet((trace,)), {"p": "prompt"})
    assert stats.n_elicitations == 1
    assert stats.parse_rate == 0.25
    with pytest.raises(ElicitationError):
        run_elicitation_pilot_greedy(Backend(), TraceSet((trace,)), {})
    with pytest.raises(ElicitationError):
        run_elicitation_pilot_greedy(Backend(), TraceSet((trace,)), {}, every=0)
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
    stats = run_elicitation_pilot_greedy(backend, traces, {"p": "prompt"}, max_steps=3)
    assert (stats.n_traces, stats.n_excluded) == (1, 2)
    assert len(backend.prefixes) == 3
    included = run_elicitation_pilot_greedy(
        Backend(), traces, {"p": "prompt"}, max_steps=3, include_censored=True
    )
    assert (included.n_traces, included.n_excluded) == (3, 0)
    excluded = run_elicitation_pilot_greedy(Backend(), TraceSet((censored,)), {})
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
        run_elicitation_pilot_greedy(Backend(), TraceSet((make_trace(),)), {}, max_steps=max_steps)


def test_collection_limit_boundary_and_metadata_fallback():
    traces = TraceSet(
        (make_trace((0, 1, 1)), make_trace()), metadata={"collect_config": {"max_steps": 3}}
    )
    for max_steps in (None, 3):
        backend = Backend()
        stats = run_elicitation_pilot_greedy(backend, traces, {"p": "prompt"}, max_steps=max_steps)
        assert (stats.n_traces, stats.n_excluded) == (1, 1)
        assert len(backend.prefixes) == 3
    overridden = run_elicitation_pilot_greedy(Backend(), traces, {"p": "prompt"}, max_steps=4)
    assert (overridden.n_traces, overridden.n_excluded) == (2, 0)


@pytest.mark.parametrize("config", [None, [], {"max_steps": True}, {"max_steps": 0}])
def test_invalid_collection_metadata(config):
    traces = TraceSet((make_trace(),), metadata={"collect_config": config})
    with pytest.raises(ElicitationError):
        run_elicitation_pilot_greedy(Backend(), traces, {"p": "prompt"})


@pytest.mark.parametrize(
    "text,expected",
    [("Alright, so 48 plus", None), (" 3 steps", 3), ("-1", None), ("+12", 12)],
)
def test_parse_leading(text, expected):
    assert parse_leading_int(text) == expected


class ScoringBackend:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_continuations(
        self, prompt, thinking_prefix, probe, candidates, require_non_digit_next=False
    ):
        self.calls.append((prompt, thinking_prefix, probe, candidates, require_non_digit_next))
        return self.scores


def test_distribution_and_filter():
    backend = ScoringBackend(np.log([0.1, 0.2, 0.3]))
    trace = make_trace()
    result = elicit_remaining_distribution(backend, "prompt", trace, every=2, max_k=2)
    np.testing.assert_allclose(result.expected, [4 / 3, np.nan, 4 / 3, np.nan])
    np.testing.assert_allclose(result.mode, [2, np.nan, 2, np.nan])
    np.testing.assert_allclose(result.mass, [0.6, np.nan, 0.6, np.nan])
    np.testing.assert_allclose(result.probs[0], [1 / 6, 1 / 3, 1 / 2])
    assert np.isnan(result.probs[1]).all()
    assert backend.calls[0][1] == ""
    assert backend.calls[1][1] == "answer 42. " * 2
    assert backend.calls[0][3:] == ([" 0", " 1", " 2"], True)
    stats, details = run_elicitation_pilot_detailed(
        backend, TraceSet((trace,)), {"p": "prompt"}, max_k=2, min_mass=0.7
    )
    assert stats.n_elicitations == 0
    assert np.isnan(details[0].expected).all()
    np.testing.assert_allclose(details[0].mass, 0.6)
    stats = run_elicitation_pilot(backend, TraceSet((trace,)), {"p": "prompt"}, max_k=2)
    assert stats.n_elicitations == 4
    assert stats.mean_bias == pytest.approx(-1 / 6)


@pytest.mark.parametrize("scores", [[0], [0, np.nan], [0, 0.1], [0, np.inf]])
def test_invalid_distribution_scores(scores):
    with pytest.raises(ElicitationError, match="scores"):
        elicit_remaining_distribution(ScoringBackend(scores), "", make_trace(), max_k=1)


def test_zero_probability_distribution():
    result = elicit_remaining_distribution(ScoringBackend([-np.inf]), "", make_trace(), max_k=0)
    assert np.isnan(result.expected).all()
    assert np.isnan(result.probs).all()
    np.testing.assert_array_equal(result.mass, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"every": 0},
        {"max_k": -1},
        {"max_k": 1.5},
        {"max_k": True},
        {"method": "invalid"},
        {"min_mass": -1},
        {"min_mass": np.nan},
    ],
)
def test_invalid_distribution_settings(kwargs):
    with pytest.raises(ElicitationError):
        run_elicitation_pilot(ScoringBackend([0]), TraceSet((make_trace(),)), {}, **kwargs)


def test_greedy_rejects_reasoning_numbers():
    class ReasoningBackend:
        def complete(self, *args):
            return "Alright, so 48 plus"

    assert np.isnan(elicit_remaining(ReasoningBackend(), "", make_trace())).all()
