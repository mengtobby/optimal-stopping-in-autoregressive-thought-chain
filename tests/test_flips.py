import numpy as np
import pytest

from osc.data.schema import Trace, TraceSet
from osc.pilots.flips import (
    AnchorError,
    find_anchor_events,
    locate_answer_span,
    step_ends,
    thinking_text,
)


def make_trace(correct=(0, 1, 1, 0), problem_id="p", answers=None):
    n = len(correct)
    return Trace(
        problem_id,
        tuple([""] + ["answer 42. "] * (n - 1)),
        np.r_[0, np.ones(n - 1)],
        tuple(answers) if answers is not None else tuple([None] + ["42"] * (n - 1)),
        np.array(correct),
        np.zeros((n, 1)),
        True,
        float(correct[-1]),
    )


def test_anchor_classification():
    traces = TraceSet(
        (make_trace(), make_trace((0, 1, 0, 1)), make_trace((0, 1, 1, 1)), make_trace((0, 0, 0, 0)))
    )
    events = find_anchor_events(traces)
    assert len(events) == 3
    assert [(e.anchor_step, e.flip_step, e.persistent) for e in events] == [
        (1, 3, True),
        (1, 2, False),
        (1, None, False),
    ]
    assert events[0].span == (7, 9)
    assert thinking_text(traces[0]) == "answer 42. " * 3
    np.testing.assert_array_equal(step_ends(traces[0]), [0, 11, 22, 33])


def test_span_and_empty_answers():
    text = r"42 then \boxed{42}"
    assert locate_answer_span(text, 0, len(text), "42") == (8, len(text))
    assert locate_answer_span("42 then 42", 3, 10, "42") == (8, 10)
    assert locate_answer_span("abc", 1, 3, "42") == (1, 3)
    event = find_anchor_events(TraceSet((make_trace((1, 1, 1), answers=("", None, "42")),)))[0]
    assert event.anchor_step == 2
    for threshold in (-1, 2, np.nan):
        with pytest.raises(AnchorError):
            find_anchor_events(TraceSet((make_trace(),)), threshold)
    with pytest.raises(AnchorError):
        locate_answer_span("abc", 2, 1, "a")
    with pytest.raises(AnchorError):
        locate_answer_span("abc", 0, 1, " ")


def test_checked_spans_and_empty_prefix_anchor():
    from osc.pilots.flips import AnchorEvent, locate_answer_span_checked

    assert locate_answer_span_checked("", 0, 0, "42") == ((0, 0), False)
    assert locate_answer_span_checked("abc", 0, 3, "42") == ((0, 3), False)
    text = r"then \boxed{42}"
    assert locate_answer_span_checked(text, 0, len(text), "42") == ((5, len(text)), True)
    trace = make_trace((1, 1, 0), answers=("42", "42", "42"))
    event = find_anchor_events(TraceSet((trace,)))[0]
    assert event.anchor_step == 1
    assert event.span_found
    missing = make_trace((1, 1), answers=("99", "99"))
    event = find_anchor_events(TraceSet((missing,)))[0]
    assert event.anchor_step == 0
    assert not event.span_found
    assert AnchorEvent("p", 0, 0, "42", (0, 0), None, False).span_found
