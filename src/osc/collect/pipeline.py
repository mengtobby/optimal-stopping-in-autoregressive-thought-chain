"""Trace collection orchestration, independent of model libraries."""

import hashlib
import logging
import os
import shutil
import tempfile
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


@dataclass(frozen=True)
class SiblingStats:
    n_transitions: int
    n_with_siblings: int
    n_draws: int
    n_extended: int
    n_ended_thinking: int
    n_budget_truncated: int


def collect_trace(backend: ReasoningBackend, problem: Problem, config: CollectConfig) -> Trace:
    return collect_trace_with_stats(backend, problem, config)[0]


def collect_trace_with_stats(
    backend: ReasoningBackend, problem: Problem, config: CollectConfig
) -> tuple[Trace, SiblingStats]:
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
    tokens = np.array([backend.count_tokens(s) for s in texts])
    cumulative = np.cumsum(tokens)
    merged = len(segment_thinking(thinking, config.segment_mode, config.min_chars, None)) > len(
        spans
    )
    n_with = n_draws = n_extended = n_ended = n_truncated = 0
    for t in range(len(texts) - 1):
        if merged and t == len(texts) - 2:
            continue
        if not config.k_siblings or t % config.sibling_stride:
            continue
        prefix = thinking[: ends[t]]
        remaining = config.max_thinking_tokens - int(cumulative[t])
        cap = min(config.sibling_max_tokens, remaining)
        if cap <= 0:
            raise CollectionError("outer transition has no remaining thinking budget")
        raw = backend.continue_steps(problem.prompt, prefix, config.k_siblings, cap)
        if len(raw) != config.k_siblings:
            raise CollectionError("backend returned an incorrect number of siblings")
        n_with += 1
        n_draws += len(raw)
        steps = []
        for partial in raw:
            used = backend.count_tokens(partial)
            extended = False
            while True:
                if used > cap:
                    raise CollectionError("backend continuation exceeds its token cap")
                # A short draw ended thinking; apply the same causal boundary as the outer step.
                if used < cap:
                    n_ended += 1
                    step = first_step(partial, config.segment_mode, config.min_chars)
                    break
                step = first_step(partial, config.segment_mode, config.min_chars)
                if len(step) < len(partial):
                    break
                left = remaining - backend.count_tokens(partial)
                if left <= 0:
                    n_truncated += 1
                    break
                cap = min(config.sibling_max_tokens, left)
                more = backend.continue_steps(problem.prompt, prefix + partial, 1, cap)
                if len(more) != 1:
                    raise CollectionError("backend returned an incorrect number of siblings")
                partial += more[0]
                used = backend.count_tokens(more[0])
                extended = True
            n_extended += int(extended)
            steps.append(step)
            cap = min(config.sibling_max_tokens, remaining)
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
    trace = Trace(
        problem.problem_id,
        texts,
        tokens,
        tuple(answers),
        np.array(correct),
        features,
        natural,
        correct[-1],
        tuple(siblings),
    )
    return trace, SiblingStats(len(texts) - 1, n_with, n_draws, n_extended, n_ended, n_truncated)


def collect_dataset(backend, problems, config, out_dir, resume: bool = True) -> TraceSet:
    """Collect unconditional siblings given each observed prefix and remaining budget.

    Stride selects prefixes only; draws are never filtered by termination or length.
    Transitions into max_steps-merged steps are excluded because their law differs.
    """
    out = Path(out_dir)
    traces = []
    totals = dict.fromkeys(SiblingStats.__dataclass_fields__, 0)
    for problem in problems:
        shard = out / "shards" / hashlib.sha256(problem.problem_id.encode()).hexdigest()
        try:
            cached = None
            if resume and shard.exists():
                try:
                    cached = load_traceset(shard)
                    if len(cached) != 1 or cached[0].problem_id != problem.problem_id:
                        raise CollectionError("shard problem ID mismatch")
                    stats = SiblingStats(**cached.metadata["sibling_stats"])
                except Exception:
                    logger.warning(
                        "invalid shard for %s; re-collecting", problem.problem_id, exc_info=True
                    )
                    shutil.rmtree(shard)
                    cached = None
            if cached is not None:
                trace = cached[0]
            else:
                trace, stats = collect_trace_with_stats(backend, problem, config)
                shard.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=shard.name + ".tmp-", dir=shard.parent))
                try:
                    save_traceset(TraceSet((trace,), {"sibling_stats": asdict(stats)}), temporary)
                    if shard.exists():
                        shutil.rmtree(shard)
                    os.replace(temporary, shard)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            traces.append(trace)
            for name, value in asdict(stats).items():
                totals[name] += value
        except Exception:
            logger.exception("collection failed for problem %s", problem.problem_id)
    if not traces:
        raise CollectionError("no problems were collected successfully")
    result = TraceSet(tuple(traces), {"collect_config": asdict(config), "sibling_stats": totals})
    save_traceset(result, out / "merged")
    return result
