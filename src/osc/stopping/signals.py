"""Observable answer features for stopping policies, never for dual anchors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from osc.collect.answers import is_equivalent

if TYPE_CHECKING:
    from osc.stopping.payoff import PayoffPanel


def _has_top_level_comma(text: str) -> bool:
    depth = 0
    for char in text:
        depth += (char in "([{") - (char in ")]}")
        if char == "," and depth == 0:
            return True
    return False


def _strip_outer(answer: str) -> str:
    """Drop redundant wrapping brackets; tuples and intervals keep theirs, so (0,1) != [0,1]."""
    answer = answer.strip()
    pairs = {"(": ")", "[": "]"}
    while answer and answer[0] in pairs and answer[-1] == pairs[answer[0]]:
        stack: list[str] = []
        for i, char in enumerate(answer):
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values():
                if not stack or stack.pop() != char:
                    return answer
            if not stack and i < len(answer) - 1:
                return answer
        if stack or _has_top_level_comma(answer[1:-1]):
            return answer
        answer = answer[1:-1].strip()
    return answer


def _canonical(answer: str) -> str:
    """Bare comma lists read as tuples: '3, x' matches '(3, x)' but not '[3, x]'."""
    stripped = _strip_outer(answer)
    return _strip_outer(f"({stripped})") if _has_top_level_comma(stripped) else stripped


def semantic_answer_ids(answers: Sequence[str | None]) -> tuple[int, ...]:
    """Assign first-seen equivalence classes within one trace; missing answers are -1."""
    representatives: list[str] = []
    ids: list[int] = []
    for answer in answers:
        if answer is not None and not isinstance(answer, str):
            raise TypeError("answers must contain strings or None")
        normalized = _canonical(answer) if answer is not None else ""
        if not normalized:
            ids.append(-1)
            continue
        for code, representative in enumerate(representatives):
            if is_equivalent(normalized, representative):
                ids.append(code)
                break
        else:
            ids.append(len(representatives))
            representatives.append(normalized)
    return tuple(ids)


def answer_signals(panel: PayoffPanel) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build causal features from answers and available sample statistics only.

    ``stable_run`` counts matching consecutive previous states, excluding the current
    state. Optional statistics are included separately when present on every trace.
    """
    names = ("stable_run", "agree_past", "n_distinct", "has_answer")
    optional = tuple(
        name
        for name in ("forced_agreement", "forced_confidence")
        if all(getattr(trace, name) is not None for trace in panel.traces)
    )
    names += optional
    signals = np.zeros((panel.n_traces, panel.t_max, len(names)), dtype=np.float32)
    for i, trace in enumerate(panel.traces):
        counts: dict[int, int] = {}
        previous = -1
        run = 0
        for t, code in enumerate(semantic_answer_ids(trace.forced_answers)):
            run = run + 1 if code >= 0 and code == previous else 0
            if code >= 0:
                counts[code] = counts.get(code, 0) + 1
                signals[i, t, 0] = min(run, 8) / 8
                signals[i, t, 1] = counts[code] / (t + 1)
                signals[i, t, 3] = 1
            signals[i, t, 2] = min(len(counts), 8) / 8
            previous = code
        for j, name in enumerate(optional, start=4):
            signals[i, : trace.n_steps, j] = np.nan_to_num(getattr(trace, name), nan=0.0)
    return signals, names


def with_signals(panel: PayoffPanel) -> tuple[PayoffPanel, tuple[str, ...]]:
    """Append policy-only signals as the final feature columns, preserving outcomes."""
    signals, names = answer_signals(panel)
    features = np.concatenate((panel.features, signals), axis=2)
    features.setflags(write=False)
    return replace(panel, features=features), names
