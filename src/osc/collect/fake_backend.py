"""Deterministic scripted backend for collection and pilot experiments."""

from collections.abc import Mapping, Sequence

import numpy as np


class FakeBackend:
    def __init__(
        self,
        thinking: str | Mapping[str, str] = "First.\n\nWait, verify.\n\nDone.",
        answers: Sequence[str] | Mapping[str, Sequence[str]] = ("\\boxed{1}",),
        continuations: Sequence[str] = ("Alternative.\n\nDone.",),
        natural_stop: bool = True,
        feature_dim: int = 3,
        completion: str = "1",
        scores: Mapping[str, float] | None = None,
    ) -> None:
        if feature_dim < 1 or not continuations:
            raise ValueError("feature_dim must be positive and continuations nonempty")
        self.thinking = thinking
        self.answers = answers
        self.continuations = tuple(continuations)
        self.natural_stop = natural_stop
        self.feature_dim = feature_dim
        self.completion = completion
        self.hidden_features_calls = 0
        self.scores = dict(scores) if scores is not None else None
        self.score_calls: list = []

    def score_continuations(
        self,
        prompt: str,
        thinking_prefix: str,
        probe: str,
        candidates: Sequence[str],
        require_non_digit_next: bool = False,
    ) -> np.ndarray:
        self.score_calls.append(
            (prompt, thinking_prefix, probe, tuple(candidates), require_non_digit_next)
        )
        if self.scores is not None:
            missing = set(candidates) - self.scores.keys()
            if missing:
                raise ValueError(f"missing scripted scores: {sorted(missing)}")
        return np.array(
            [self.scores[c] if self.scores is not None else -float(len(c)) for c in candidates],
            dtype=float,
        )

    def generate_thinking(self, prompt: str, max_tokens: int) -> tuple[str, bool]:
        text = self.thinking[prompt] if isinstance(self.thinking, Mapping) else self.thinking
        return text[:max_tokens], self.natural_stop and len(text) <= max_tokens

    def count_tokens(self, text: str) -> int:
        return len(text)

    def force_answers(self, prompt: str, thinking_prefix: str, n_samples: int) -> list[str]:
        samples = (
            self.answers[thinking_prefix] if isinstance(self.answers, Mapping) else self.answers
        )
        if not samples:
            raise ValueError("scripted answers must be nonempty")
        return [samples[i % len(samples)] for i in range(n_samples)]

    def continue_steps(
        self, prompt: str, thinking_prefix: str, k: int, max_tokens: int
    ) -> list[str]:
        return [self.continuations[i % len(self.continuations)][:max_tokens] for i in range(k)]

    def complete(self, prompt: str, thinking_prefix: str, probe: str, max_tokens: int) -> str:
        return self.completion[:max_tokens]

    def hidden_features(
        self, prompt: str, thinking: str, char_positions: Sequence[int]
    ) -> np.ndarray:
        self.hidden_features_calls += 1
        return np.array(
            [[p + j for j in range(self.feature_dim)] for p in char_positions], dtype=np.float32
        ).reshape(len(char_positions), self.feature_dim)

    def attention_to_span(
        self, prompt: str, thinking: str, span: tuple[int, int], query_char_positions: Sequence[int]
    ) -> np.ndarray:
        start, end = span
        if not 0 <= start <= end <= len(thinking):
            raise ValueError("span must be within thinking")
        return np.array(
            [max(0, min(end, p) - start) / max(1, p) for p in query_char_positions],
            dtype=np.float32,
        ).reshape(-1, 1, 1)
