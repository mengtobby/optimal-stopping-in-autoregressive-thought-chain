"""Attention to earlier answers, matched by distance from the answer anchor."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from osc.collect.protocol import ReasoningBackend
from osc.data.schema import Trace, TraceSet
from osc.pilots.flips import AnchorEvent, find_anchor_events, step_ends, thinking_text


class AttentionError(ValueError):
    """Invalid attention pilot input."""


def event_query_positions(trace: Trace, event: AnchorEvent, max_after: int) -> list[int]:
    if max_after < 1:
        raise AttentionError("max_after must be positive")
    if not 0 <= event.anchor_step < trace.n_steps:
        raise AttentionError("anchor step is outside trace")
    last = trace.n_steps - 1 if event.flip_step is None else event.flip_step - 1
    return step_ends(trace)[
        event.anchor_step + 1 : min(last, event.anchor_step + max_after) + 1
    ].tolist()


def attention_trajectory(
    backend: ReasoningBackend,
    prompt: str,
    thinking: str,
    span: tuple[int, int],
    query_char_positions: Sequence[int],
    layers: Sequence[int] | None = None,
    head_reduce: str = "mean",
) -> np.ndarray:
    if head_reduce not in ("mean", "max"):
        raise AttentionError("head_reduce must be 'mean' or 'max'")
    if not 0 <= span[0] <= span[1] <= len(thinking):
        raise AttentionError("span must lie within thinking")
    if any(pos < 0 or pos > len(thinking) for pos in query_char_positions):
        raise AttentionError("query positions must lie within thinking")
    if not len(query_char_positions):
        return np.empty(0)
    values = np.asarray(backend.attention_to_span(prompt, thinking, span, query_char_positions))
    if values.ndim != 3 or values.shape[0] != len(query_char_positions):
        raise AttentionError("backend attention must have shape (n_queries, layers, heads)")
    if 0 in values.shape or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise AttentionError("backend attention must be non-empty, finite and in [0, 1]")
    if layers is not None:
        if not len(layers) or any(i < -values.shape[1] or i >= values.shape[1] for i in layers):
            raise AttentionError("selected layers must be non-empty and within backend layers")
        values = values[:, layers, :]
    reduced = values.mean(axis=2) if head_reduce == "mean" else values.max(axis=2)
    return reduced.mean(axis=1)


@dataclass(frozen=True)
class AttentionPilotResult:
    flip_curves: np.ndarray
    control_curves: np.ndarray
    auroc: float
    perm_p: float
    n_flip: int
    n_control: int


def compare_flip_vs_control(
    flip_curves: np.ndarray,
    control_curves: np.ndarray,
    flip_offsets: Sequence[int],
    n_perm: int = 2000,
    seed: int = 0,
) -> AttentionPilotResult:
    """Offsets are one-based steps since anchor; permutation test is one-sided.

    Each eligible flip contributes one case and controls observed at that offset.
    Controls can consequently appear more than once in the matched pooled sample.
    """
    flips, controls = np.array(flip_curves, float), np.array(control_curves, float)
    offsets = np.asarray(flip_offsets)
    if flips.ndim != 2 or controls.ndim != 2 or flips.shape[1] != controls.shape[1]:
        raise AttentionError("curves must be 2-D with equal padded widths")
    if offsets.shape != (len(flips),) or not np.all(np.isfinite(offsets)):
        raise AttentionError("flip_offsets must contain one finite offset per flip")
    if np.any(offsets != np.floor(offsets)) or np.any(offsets < 0):
        raise AttentionError("flip offsets must be non-negative integers")
    if n_perm < 1:
        raise AttentionError("n_perm must be positive")
    if np.any(np.isinf(flips)) or np.any(np.isinf(controls)):
        raise AttentionError("curves may contain NaN padding but not infinity")
    cases, references = [], []
    for row, offset in zip(flips, offsets, strict=True):
        j = int(offset) - 1
        if not 0 <= j < flips.shape[1] or not np.isfinite(row[j]):
            continue
        matched = controls[:, j]
        matched = matched[np.isfinite(matched)]
        if matched.size:
            cases.append(row[j])
            references.extend(matched.tolist())
    auc = p = float("nan")
    if cases and references:
        labels = np.r_[np.ones(len(cases)), np.zeros(len(references))]
        scores = -np.r_[cases, references]
        auc = float(roc_auc_score(labels, scores))
        rng = np.random.default_rng(seed)
        exceed = sum(roc_auc_score(rng.permutation(labels), scores) >= auc for _ in range(n_perm))
        p = (1 + exceed) / (n_perm + 1)
    flips.setflags(write=False)
    controls.setflags(write=False)
    return AttentionPilotResult(flips, controls, auc, p, len(flips), len(controls))


def run_attention_pilot(
    backend: ReasoningBackend,
    traces: TraceSet,
    prompts: Mapping[str, str],
    max_after: int = 20,
    layers: Sequence[int] | None = None,
) -> AttentionPilotResult:
    if max_after < 1:
        raise AttentionError("max_after must be positive")
    flips, controls, offsets = [], [], []
    for event in find_anchor_events(traces):
        trace = traces[event.trace_index]
        if trace.problem_id not in prompts:
            raise AttentionError(f"missing prompt for {trace.problem_id!r}")
        positions = event_query_positions(trace, event, max_after)
        curve = np.full(max_after, np.nan)
        curve[: len(positions)] = attention_trajectory(
            backend, prompts[trace.problem_id], thinking_text(trace), event.span, positions, layers
        )
        if event.flip_step is None:
            controls.append(curve)
        else:
            flips.append(curve)
            offsets.append(event.flip_step - event.anchor_step - 1)
    return compare_flip_vs_control(
        np.array(flips).reshape(-1, max_after), np.array(controls).reshape(-1, max_after), offsets
    )
