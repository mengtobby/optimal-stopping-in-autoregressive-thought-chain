"""Trace collection orchestration, independent of model libraries."""

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from osc.collect.answers import grade_samples
from osc.collect.datasets import Problem
from osc.collect.protocol import ReasoningBackend
from osc.collect.segment import first_step, segment_thinking
from osc.data.io import load_traceset, save_traceset
from osc.data.schema import SiblingSet, Trace, TraceSet

logger = logging.getLogger(__name__)


class CollectionError(ValueError):
    """Invalid collection configuration or backend output."""


@dataclass(frozen=True)
class CollectConfig:
    max_thinking_tokens: int = 8192
    n_forced: int = 4
    k_siblings: int = 4
    sibling_stride: int = 1
    sibling_max_tokens: int = 512
    segment_mode: str = "paragraph"
    min_chars: int = 40
    max_steps: int = 64

    def __post_init__(self) -> None:
        for name in (
            "max_thinking_tokens",
            "n_forced",
            "sibling_stride",
            "sibling_max_tokens",
            "max_steps",
        ):
            if getattr(self, name) < 1:
                raise CollectionError(f"{name} must be positive")
        if self.k_siblings < 0 or self.k_siblings == 1:
            raise CollectionError("k_siblings must be zero or at least two")
        segment_thinking("", self.segment_mode, self.min_chars, self.max_steps)


def collect_trace(backend: ReasoningBackend, problem: Problem, config: CollectConfig) -> Trace:
    thinking, natural = backend.generate_thinking(problem.prompt, config.max_thinking_tokens)
    spans = segment_thinking(thinking, config.segment_mode, config.min_chars, config.max_steps)
    texts = ("", *(thinking[a:b] for a, b in spans))
    ends = [0, *(b for _, b in spans)]
    answers, correct = [], []
    for end in ends:
        answer, score = grade_samples(
            backend.force_answers(problem.prompt, thinking[:end], config.n_forced), problem.answer
        )
        answers.append(answer)
        correct.append(score)
    features = backend.hidden_features(problem.prompt, thinking, ends)
    siblings: list[SiblingSet | None] = [None] * len(texts)
    for t in range(len(texts) - 2):
        if not config.k_siblings or t % config.sibling_stride:
            continue
        prefix = thinking[: ends[t]]
        raw = backend.continue_steps(
            problem.prompt, prefix, config.k_siblings, config.sibling_max_tokens
        )
        if len(raw) != config.k_siblings:
            raise CollectionError("backend returned an incorrect number of siblings")
        steps = [first_step(s, config.segment_mode, config.min_chars) for s in raw]
        scores, feats = [], []
        for step in steps:
            full = prefix + step
            scores.append(
                grade_samples(
                    backend.force_answers(problem.prompt, full, config.n_forced), problem.answer
                )[1]
            )
            feats.append(backend.hidden_features(problem.prompt, full, [len(full)])[0])
        siblings[t] = SiblingSet(
            np.array([backend.count_tokens(s) for s in steps]), np.array(scores), np.array(feats)
        )
    return Trace(
        problem.problem_id,
        texts,
        np.array([backend.count_tokens(s) for s in texts]),
        tuple(answers),
        np.array(correct),
        features,
        natural,
        correct[-1],
        tuple(siblings),
    )


def collect_dataset(backend, problems, config, out_dir, resume: bool = True) -> TraceSet:
    out = Path(out_dir)
    traces = []
    for problem in problems:
        shard = out / "shards" / hashlib.sha256(problem.problem_id.encode()).hexdigest()
        try:
            if resume and shard.exists():
                trace = load_traceset(shard)[0]
                if trace.problem_id != problem.problem_id:
                    raise CollectionError("shard problem ID mismatch")
            else:
                trace = collect_trace(backend, problem, config)
                save_traceset(TraceSet((trace,)), shard)
            traces.append(trace)
        except Exception:
            logger.exception("collection failed for problem %s", problem.problem_id)
    if not traces:
        raise CollectionError("no problems were collected successfully")
    result = TraceSet(tuple(traces), {"collect_config": asdict(config)})
    save_traceset(result, out / "merged")
    return result
