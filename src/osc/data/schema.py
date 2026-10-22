"""Immutable trace records shared by collection, stopping and pilot code.

A reasoning trace is split into steps. Index ``t`` denotes the state *after* step ``t``
(by convention the collection pipeline makes step 0 the empty prefix, i.e. answering
without any thinking, with ``step_tokens[0] == 0``). Stopping at ``t`` means forcing a
final answer from the prefix ending at step ``t``; ``forced_correct[t]`` is the
probability (mean over forced-answer samples) that this answer is correct.
Step texts are exact, contiguous slices of the thinking text (delimiters stay attached
to the end of their step), so ``"".join(step_texts)`` reconstructs the thinking.

Only ``forced_correct`` / ``final_correct`` need ground truth. Everything else (texts,
forced answers, token counts, features) is observable at decision time, so stopping
policies may use it but must never read correctness.

``siblings[t]`` (for ``t < T - 1``) holds ``K`` alternative next steps sampled
independently from the prefix ending at step ``t``: one-step continuations used for
nested (Andersen-Broadie) martingale estimates in the dual bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


class SchemaError(ValueError):
    """Raised when a trace record violates the schema."""


def _frozen(array: Any, dtype: Any, name: str, ndim: int) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True)
    if out.ndim != ndim:
        raise SchemaError(f"{name} must be {ndim}-D, got shape {out.shape}")
    out.setflags(write=False)
    return out


def _check_prob(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise SchemaError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class SiblingSet:
    """K independent one-step continuations sampled from the prefix ending at step t."""

    step_tokens: np.ndarray  # (K,) int, tokens in each sibling step
    forced_correct: np.ndarray  # (K,) float in [0, 1]
    features: np.ndarray  # (K, d) float32

    def __post_init__(self) -> None:
        tokens = _frozen(self.step_tokens, np.int64, "siblings.step_tokens", 1)
        correct = _frozen(self.forced_correct, np.float64, "siblings.forced_correct", 1)
        feats = _frozen(self.features, np.float32, "siblings.features", 2)
        k = tokens.shape[0]
        if k < 2:
            raise SchemaError("a sibling set needs at least 2 continuations")
        if correct.shape != (k,) or feats.shape[0] != k:
            raise SchemaError("sibling arrays must share their first dimension")
        if np.any(tokens < 0):
            raise SchemaError("siblings.step_tokens must be non-negative")
        _check_prob(correct, "siblings.forced_correct")
        object.__setattr__(self, "step_tokens", tokens)
        object.__setattr__(self, "forced_correct", correct)
        object.__setattr__(self, "features", feats)

    @property
    def k(self) -> int:
        return int(self.step_tokens.shape[0])


@dataclass(frozen=True)
class Trace:
    """One reasoning trace with per-step stop payload."""

    problem_id: str
    step_texts: tuple[str, ...]
    step_tokens: np.ndarray  # (T,) int
    forced_answers: tuple[str | None, ...]
    forced_correct: np.ndarray  # (T,) float in [0, 1]
    features: np.ndarray  # (T, d) float32
    natural_stop: bool  # True if the model closed its thinking itself after step T-1
    final_correct: float  # correctness of the model's own final answer, in [0, 1]
    siblings: tuple[SiblingSet | None, ...] = ()  # empty, or length T (last entry None)

    def __post_init__(self) -> None:
        tokens = _frozen(self.step_tokens, np.int64, "step_tokens", 1)
        correct = _frozen(self.forced_correct, np.float64, "forced_correct", 1)
        feats = _frozen(self.features, np.float32, "features", 2)
        t = tokens.shape[0]
        if t < 1:
            raise SchemaError("a trace needs at least one step")
        if len(self.step_texts) != t or len(self.forced_answers) != t:
            raise SchemaError("step_texts and forced_answers must have length T")
        if correct.shape != (t,) or feats.shape[0] != t:
            raise SchemaError("forced_correct and features must have length T")
        if np.any(tokens < 0):
            raise SchemaError("step_tokens must be non-negative")
        _check_prob(correct, "forced_correct")
        _check_prob(np.array([self.final_correct], dtype=np.float64), "final_correct")
        siblings = tuple(self.siblings)
        if siblings:
            if len(siblings) != t:
                raise SchemaError("siblings must be empty or have length T")
            if siblings[-1] is not None:
                raise SchemaError("the last step cannot have siblings")
            d = feats.shape[1]
            for s in siblings:
                if s is not None and s.features.shape[1] != d:
                    raise SchemaError("sibling feature dimension must match trace features")
        object.__setattr__(self, "step_texts", tuple(self.step_texts))
        object.__setattr__(self, "forced_answers", tuple(self.forced_answers))
        object.__setattr__(self, "step_tokens", tokens)
        object.__setattr__(self, "forced_correct", correct)
        object.__setattr__(self, "features", feats)
        object.__setattr__(self, "siblings", siblings)
        object.__setattr__(self, "natural_stop", bool(self.natural_stop))
        object.__setattr__(self, "final_correct", float(self.final_correct))

    @property
    def n_steps(self) -> int:
        return int(self.step_tokens.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def cum_tokens(self) -> np.ndarray:
        """Tokens generated up to and including step t, shape (T,)."""
        out = np.cumsum(self.step_tokens)
        out.setflags(write=False)
        return out

    @property
    def has_siblings(self) -> bool:
        return bool(self.siblings) and any(s is not None for s in self.siblings)


@dataclass(frozen=True)
class TraceSet:
    """An immutable collection of traces sharing a feature dimension."""

    traces: tuple[Trace, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        traces = tuple(self.traces)
        if not traces:
            raise SchemaError("a TraceSet needs at least one trace")
        dims = {tr.feature_dim for tr in traces}
        if len(dims) != 1:
            raise SchemaError(f"all traces must share a feature dimension, got {sorted(dims)}")
        ids = [tr.problem_id for tr in traces]
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "_ids", tuple(ids))

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self):
        return iter(self.traces)

    def __getitem__(self, index: int) -> Trace:
        return self.traces[index]

    @property
    def feature_dim(self) -> int:
        return self.traces[0].feature_dim

    @property
    def max_steps(self) -> int:
        return max(tr.n_steps for tr in self.traces)

    def subset(self, indices: Sequence[int]) -> TraceSet:
        """Return a new TraceSet with the traces at ``indices`` (order preserved)."""
        return TraceSet(tuple(self.traces[i] for i in indices), self.metadata)

    def with_metadata(self, **updates: Any) -> TraceSet:
        return TraceSet(self.traces, {**self.metadata, **updates})
