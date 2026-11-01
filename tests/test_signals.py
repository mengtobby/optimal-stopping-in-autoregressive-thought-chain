from dataclasses import replace

import numpy as np
import pytest

from osc.data.schema import Trace, TraceSet
from osc.stopping.payoff import build_panel
from osc.stopping.signals import answer_signals, semantic_answer_ids, with_signals


def _trace(answers: tuple[str | None, ...]) -> Trace:
    n = len(answers)
    return Trace(
        problem_id="a",
        step_texts=("",) * n,
        step_tokens=np.ones(n, dtype=int),
        forced_answers=answers,
        forced_correct=np.zeros(n),
        features=np.ones((n, 2)),
        natural_stop=True,
        final_correct=0.0,
    )


def test_semantic_ids():
    assert semantic_answer_ids(
        (None, " ", "[(0.5)]", r"\frac{1}{2}", "(1)+(2)", "1)+(2", "[3)")
    ) == (-1, -1, 0, 0, 1, 2, 3)
    assert semantic_answer_ids(("((", "()", "([3)]")) == (0, -1, 1)
    with pytest.raises(TypeError):
        semantic_answer_ids((3,))


def test_signal_values_and_padding():
    panel = build_panel(TraceSet((_trace((None, "1", "1", "2", "1", None)), _trace(("3",)))), 0)
    signals, names = answer_signals(panel)
    assert names == ("stable_run", "agree_past", "n_distinct", "has_answer")
    assert signals.dtype == np.float32
    np.testing.assert_allclose(
        signals[0],
        [
            [0, 0, 0, 0],
            [0, 1 / 2, 1 / 8, 1],
            [1 / 8, 2 / 3, 1 / 8, 1],
            [0, 1 / 4, 2 / 8, 1],
            [0, 3 / 5, 2 / 8, 1],
            [0, 0, 2 / 8, 0],
        ],
    )
    assert not signals[1, 1:].any()
    long, _ = answer_signals(build_panel(TraceSet((_trace(("1",) * 12),)), 0))
    assert long[0, -1, 0] == 1
    distinct, _ = answer_signals(
        build_panel(TraceSet((_trace(tuple(str(i) for i in range(12))),)), 0)
    )
    assert distinct[0, -1, 2] == 1


def test_all_signals_are_causal_and_ignore_correctness():
    trace = replace(
        _trace(("1", "1", "2", "3")),
        forced_agreement=np.array([0.5, 1, 0.5, 1]),
        forced_confidence=np.array([np.nan, 0.2, 0.8, 1]),
    )
    original, _ = answer_signals(build_panel(TraceSet((trace,)), 0))
    for t in range(trace.n_steps):
        changed = replace(
            trace,
            forced_answers=trace.forced_answers[: t + 1] + ("99",) * (3 - t),
            forced_agreement=np.concatenate((trace.forced_agreement[: t + 1], np.zeros(3 - t))),
            forced_confidence=np.concatenate((trace.forced_confidence[: t + 1], np.zeros(3 - t))),
            forced_correct=np.ones(4),
            final_correct=1,
        )
        actual, _ = answer_signals(build_panel(TraceSet((changed,)), 0))
        np.testing.assert_array_equal(actual[:, : t + 1], original[:, : t + 1])


@pytest.mark.parametrize("field", ["forced_agreement", "forced_confidence"])
def test_optional_columns_require_every_trace(field):
    trace = _trace(("1", "2"))
    complete = replace(trace, **{field: np.array([0.5, 1.0])})
    signals, names = answer_signals(build_panel(TraceSet((complete, complete)), 0))
    assert names[-1] == field
    np.testing.assert_array_equal(signals[0, :, -1], [0.5, 1])
    _, names = answer_signals(build_panel(TraceSet((complete, trace)), 0))
    assert field not in names


def test_with_signals_preserves_outcomes_and_appends():
    trace = replace(_trace(("1", "1")), forced_confidence=np.array([np.nan, 0.5]))
    panel = build_panel(TraceSet((trace,)), 0.2)
    augmented, names = with_signals(panel)
    signals, _ = answer_signals(panel)
    np.testing.assert_array_equal(augmented.features[:, :, :2], panel.features)
    np.testing.assert_array_equal(augmented.features[:, :, -len(names) :], signals)
    assert augmented.payoff is panel.payoff
    assert augmented.accuracy is panel.accuracy
    assert augmented.traces is panel.traces
    assert not augmented.features.flags.writeable
    assert signals[0, 0, -1] == 0
