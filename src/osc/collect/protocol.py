"""The model-facing seam. Collection and pilot code depend only on this Protocol.

``prompt`` is the raw problem statement; the backend owns chat templates and the
``<think>`` wrapper. ``thinking`` is reasoning text produced inside the think block.
Character offsets always index into ``thinking``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ReasoningBackend(Protocol):
    def generate_thinking(self, prompt: str, max_tokens: int) -> tuple[str, bool]:
        """Sample a full thinking block. Returns (thinking, natural_stop): natural_stop is
        True when the model closed its thinking itself before ``max_tokens``."""
        ...

    def count_tokens(self, text: str) -> int:
        """Number of tokens ``text`` encodes to (no special tokens)."""
        ...

    def force_answers(self, prompt: str, thinking_prefix: str, n_samples: int) -> list[str]:
        """Close the think block after ``thinking_prefix`` and sample ``n_samples`` short
        final-answer completions (raw text, to be parsed by the caller)."""
        ...

    def continue_steps(
        self, prompt: str, thinking_prefix: str, k: int, max_tokens: int
    ) -> list[str]:
        """Sample ``k`` independent raw continuations of ``thinking_prefix`` (up to
        ``max_tokens`` each, stopping early if the model closes its thinking), using the
        same sampling parameters as ``generate_thinking``. The caller cuts each into one
        step with the same causal rule used to segment full traces, so sibling steps
        share the law of the trace's actual next step (required by the dual bound)."""
        ...

    def complete(self, prompt: str, thinking_prefix: str, probe: str, max_tokens: int) -> str:
        """Append ``probe`` (e.g. a self-report question) after ``thinking_prefix`` inside
        the think block and greedily decode up to ``max_tokens``. Does not alter state."""
        ...

    def hidden_features(
        self, prompt: str, thinking: str, char_positions: Sequence[int]
    ) -> np.ndarray:
        """One forward pass over ``thinking``; returns the feature vector at the last token
        ending at or before each char position. Shape (len(char_positions), d)."""
        ...

    def attention_to_span(
        self,
        prompt: str,
        thinking: str,
        span: tuple[int, int],
        query_char_positions: Sequence[int],
    ) -> np.ndarray:
        """Attention mass each query token puts on the key tokens covering the character
        span [start, end) of ``thinking``. Shape (n_queries, n_layers, n_heads), values in
        [0, 1]."""
        ...
