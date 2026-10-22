"""Retrospective answer anchors and correctness flips (not stopping policies)."""

from dataclasses import dataclass

import numpy as np

from osc.data.schema import Trace, TraceSet


class AnchorError(ValueError):
    """Invalid anchor analysis input."""


def thinking_text(trace: Trace) -> str:
    return "".join(trace.step_texts)


def step_ends(trace: Trace) -> np.ndarray:
    return np.cumsum([len(text) for text in trace.step_texts], dtype=np.int64)


@dataclass(frozen=True)
class AnchorEvent:
    problem_id: str
    trace_index: int
    anchor_step: int
    answer: str
    span: tuple[int, int]
    flip_step: int | None
    persistent: bool
    span_found: bool = True


def locate_answer_span(
    thinking: str, step_start: int, step_end: int, answer: str
) -> tuple[int, int]:
    return locate_answer_span_checked(thinking, step_start, step_end, answer)[0]


def locate_answer_span_checked(
    thinking: str, step_start: int, step_end: int, answer: str
) -> tuple[tuple[int, int], bool]:
    """Locate an answer in the prefix, flagging the fallback step span as missing."""
    if not 0 <= step_start <= step_end <= len(thinking):
        raise AnchorError("step offsets must satisfy 0 <= start <= end <= len(thinking)")
    if not answer.strip():
        raise AnchorError("answer must be non-empty")
    prefix = thinking[:step_end]
    boxed = "\\boxed{" + answer + "}"
    candidates = [(prefix.rfind(value), value) for value in (answer, boxed)]
    # A boxed occurrence covers the whole expression, including its answer substring.
    candidates = [(pos, value) for pos, value in candidates if pos >= 0]
    if not candidates:
        return (step_start, step_end), False
    pos, value = max(candidates, key=lambda item: (item[0] + len(item[1]), len(item[1])))
    return (pos, pos + len(value)), True


def find_anchor_events(traces: TraceSet, threshold: float = 0.5) -> tuple[AnchorEvent, ...]:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise AnchorError("threshold must be finite and in [0, 1]")
    events = []
    for i, trace in enumerate(traces):
        anchors = [
            t
            for t, answer in enumerate(trace.forced_answers)
            if trace.forced_correct[t] >= threshold and answer is not None and answer.strip()
        ]
        if not anchors:
            continue
        t = anchors[0]
        ends = step_ends(trace)
        thinking = thinking_text(trace)
        for candidate in anchors:
            candidate_answer = trace.forced_answers[candidate].strip()
            _, found = locate_answer_span_checked(
                thinking,
                int(ends[candidate - 1]) if candidate else 0,
                int(ends[candidate]),
                candidate_answer,
            )
            if found:
                t = candidate
                break
        answer = trace.forced_answers[t].strip()
        later = np.flatnonzero(trace.forced_correct[t + 1 :] < threshold)
        flip = t + 1 + int(later[0]) if later.size else None
        span, found = locate_answer_span_checked(
            thinking, int(ends[t - 1]) if t else 0, int(ends[t]), answer
        )
        events.append(
            AnchorEvent(
                trace.problem_id,
                i,
                t,
                answer,
                span,
                flip,
                bool(flip is not None and np.all(trace.forced_correct[flip:] < threshold)),
                found,
            )
        )
    return tuple(events)
