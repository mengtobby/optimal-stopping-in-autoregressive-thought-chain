"""Elicit remaining steps and measure whether predicted stopping times recede."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import spearmanr

from osc.collect.protocol import ReasoningBackend
from osc.data.schema import Trace, TraceSet
from osc.pilots.flips import step_ends, thinking_text

LEGACY_PROBE = (
    "How many more reasoning steps do you need before giving your final answer? "
    "Reply with one integer from 0 to 10000 only."
)
DEFAULT_PROBE = (
    "\n\nBefore continuing, I estimate the number of further reasoning steps I need before "
    "writing the final answer. Number of further steps:"
)


class ElicitationError(ValueError):
    """Invalid elicitation input."""


def parse_int(text: str) -> int | None:
    """Parse the first signed integer; an out-of-range first integer is invalid."""
    match = re.search(r"[+-]?\d+", text)
    if match is None:
        return None
    digits = match.group().lstrip("+-").lstrip("0") or "0"
    if len(digits) > 5:
        return None
    value = int(match.group()[0] + digits) if match.group()[0] in "+-" else int(digits)
    return value if 0 <= value <= 10000 else None


def parse_leading_int(text: str) -> int | None:
    """Accept a step count only at the start of the completion."""
    match = re.match(r"\s*([+-]?\d+)", text)
    return None if match is None else parse_int(match.group(1))


@dataclass(frozen=True)
class ElicitedPlans:
    """Per-step plans; skipped rows are NaN and mass retains probe diagnostics."""

    expected: np.ndarray
    mode: np.ndarray
    probs: np.ndarray
    mass: np.ndarray


def _empty_plans(n_steps: int, max_k: int) -> ElicitedPlans:
    return ElicitedPlans(
        np.full(n_steps, np.nan),
        np.full(n_steps, np.nan),
        np.full((n_steps, max_k + 1), np.nan),
        np.full(n_steps, np.nan),
    )


def _validate_distribution(every: int, max_k: int) -> None:
    if isinstance(every, bool) or not isinstance(every, (int, np.integer)) or every < 1:
        raise ElicitationError("every must be a positive integer")
    if (
        isinstance(max_k, bool)
        or not isinstance(max_k, (int, np.integer))
        or not 0 <= max_k <= 10000
    ):
        raise ElicitationError("max_k must be an integer in [0, 10000]")


def elicit_remaining_distribution(
    backend: ReasoningBackend,
    prompt: str,
    trace: Trace,
    every: int = 1,
    probe: str = DEFAULT_PROBE,
    max_k: int = 30,
) -> ElicitedPlans:
    """Score bounded integer counts, retaining their unnormalized probability mass."""
    _validate_distribution(every, max_k)
    result = _empty_plans(trace.n_steps, max_k)
    thinking, ends = thinking_text(trace), step_ends(trace)
    counts = np.arange(max_k + 1)
    candidates = [f" {k}" for k in counts]
    for t in range(0, trace.n_steps, every):
        scores = np.asarray(
            backend.score_continuations(
                prompt, thinking[: ends[t]], probe, candidates, require_non_digit_next=True
            ),
            dtype=float,
        )
        if scores.shape != (max_k + 1,) or np.any(np.isnan(scores)) or np.any(scores > 0):
            raise ElicitationError("scores must be one non-positive log-probability per candidate")
        result.mass[t] = np.exp(scores).sum()
        peak = scores.max()
        if np.isneginf(peak):
            continue
        weights = np.exp(scores - peak)
        result.probs[t] = weights / weights.sum()
        result.expected[t] = np.dot(counts, result.probs[t])
        result.mode[t] = np.argmax(scores)
    return result


def elicit_remaining(
    backend: ReasoningBackend,
    prompt: str,
    trace: Trace,
    every: int = 1,
    probe: str = DEFAULT_PROBE,
    max_tokens: int = 8,
) -> np.ndarray:
    if every < 1 or max_tokens < 1:
        raise ElicitationError("every and max_tokens must be positive")
    planned = np.full(trace.n_steps, np.nan)
    thinking = thinking_text(trace)
    ends = step_ends(trace)
    for t in range(0, trace.n_steps, every):
        value = parse_leading_int(backend.complete(prompt, thinking[: ends[t]], probe, max_tokens))
        if value is not None:
            planned[t] = value
    return planned


@dataclass(frozen=True)
class ConsistencyStats:
    n_traces: int
    n_elicitations: int
    parse_rate: float
    mean_bias: float
    spearman: float
    receding_slope: float
    receding_slope_se: float
    reversal_rate: float
    per_trace_slopes: np.ndarray
    n_excluded: int = 0


def consistency_stats(
    planned: np.ndarray,
    horizons: Sequence[int],
    tolerance: float = 1,
    censored: Sequence[bool] | None = None,
) -> ConsistencyStats:
    """Analyze NaN-padded plans; parse_rate counts all in-horizon slots.

    Skipped and unparseable probes are indistinguishable in the public array format.
    The slope SE is the usual homoskedastic OLS SE with trace intercepts removed.
    Censored rows are excluded from all statistics, including per_trace_slopes.
    """
    values = np.asarray(planned, float)
    h = np.asarray(horizons)
    if values.ndim != 2 or h.shape != (len(values),):
        raise ElicitationError("planned must be 2-D and horizons must have one entry per trace")
    if not np.all(np.isfinite(h)) or np.any(h != np.floor(h)):
        raise ElicitationError("horizons must be finite integers")
    if np.any(h < 1) or np.any(h > values.shape[1]):
        raise ElicitationError("horizons must lie within the planned width")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ElicitationError("tolerance must be finite and non-negative")
    alive = np.arange(values.shape[1])[None, :] < h[:, None]
    if np.any(np.isinf(values)) or np.any((values[alive] < 0) | (values[alive] > 10000)):
        raise ElicitationError("plans must be NaN or finite values in [0, 10000]")
    excluded = np.zeros(len(values), dtype=bool) if censored is None else np.asarray(censored)
    if excluded.shape != (len(values),) or excluded.dtype.kind != "b":
        raise ElicitationError("censored must contain one boolean per trace")
    n_excluded = int(excluded.sum())
    values, h, alive = values[~excluded], h[~excluded], alive[~excluded]
    valid = alive & np.isfinite(values)
    rows, times = np.nonzero(valid)
    observed = values[valid]
    realized = h[rows] - 1 - times
    slopes = np.full(len(values), np.nan)
    xs, ys = [], []
    groups = 0
    for i in range(len(values)):
        t = np.flatnonzero(valid[i]).astype(float)
        if t.size:
            groups += 1
            stop = t + values[i, valid[i]]
            x, y = t - t.mean(), stop - stop.mean()
            xs.extend(x.tolist())
            ys.extend(y.tolist())
            if t.size > 1:
                slopes[i] = np.dot(x, y) / np.dot(x, x)
    x, y = np.array(xs), np.array(ys)
    slope = se = float("nan")
    ss = float(np.dot(x, x))
    if ss > 0:
        slope = float(np.dot(x, y) / ss)
        dof = len(x) - groups - 1
        if dof > 0:
            se = float(np.sqrt(np.sum((y - slope * x) ** 2) / dof / ss))
    correlation = float("nan")
    if observed.size > 1 and np.ptp(observed) > 0 and np.ptp(realized) > 0:
        correlation = float(spearmanr(observed, realized).statistic)
    slopes.setflags(write=False)
    return ConsistencyStats(
        len(values),
        len(observed),
        float(valid.sum() / alive.sum()) if alive.any() else float("nan"),
        float(np.mean(observed - realized)) if observed.size else float("nan"),
        correlation,
        slope,
        se,
        float(np.mean(observed + tolerance < realized)) if observed.size else float("nan"),
        slopes,
        n_excluded,
    )


def run_elicitation_pilot_detailed(
    backend: ReasoningBackend,
    traces: TraceSet,
    prompts: Mapping[str, str],
    every: int = 1,
    include_censored: bool = False,
    max_steps: int | None = None,
    method: Literal["distribution", "greedy"] = "distribution",
    min_mass: float = 0.0,
    max_k: int = 30,
) -> tuple[ConsistencyStats, tuple[ElicitedPlans, ...]]:
    """Probe completed traces; optionally include traces censored by collection limits.

    max_steps is CollectConfig.max_steps, defaulting to collect_config in metadata.
    The empty prefix counts as a step: M steps are kept, M + 1 steps are censored.
    Traces without a natural stop are also censored.
    Details follow input trace order, including NaN rows for excluded traces.
    min_mass filters expected values; mode, probabilities and mass remain diagnostic.
    Greedy details contain counts in expected/mode and NaN probabilities/mass.
    """
    _validate_distribution(every, max_k)
    if method not in {"distribution", "greedy"}:
        raise ElicitationError("method must be distribution or greedy")
    if not np.isfinite(min_mass) or not 0 <= min_mass <= 1:
        raise ElicitationError("min_mass must be finite and in [0, 1]")
    if max_steps is None:
        collect_config = traces.metadata.get("collect_config", {})
        if not isinstance(collect_config, Mapping):
            raise ElicitationError("metadata collect_config must be a mapping")
        max_steps = collect_config.get("max_steps")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1
    ):
        raise ElicitationError("max_steps must be a positive integer")
    censored = [
        not include_censored
        and (not trace.natural_stop or (max_steps is not None and trace.n_steps > max_steps))
        for trace in traces
    ]
    planned = np.full((len(traces), traces.max_steps), np.nan)
    details = []
    for i, trace in enumerate(traces):
        detail = _empty_plans(trace.n_steps, max_k)
        details.append(detail)
        if censored[i]:
            continue
        if trace.problem_id not in prompts:
            raise ElicitationError(f"missing prompt for {trace.problem_id!r}")
        if method == "distribution":
            detail = elicit_remaining_distribution(
                backend, prompts[trace.problem_id], trace, every, max_k=max_k
            )
            details[-1] = detail
            detail.expected[detail.mass < min_mass] = np.nan
        else:
            detail.expected[:] = elicit_remaining(backend, prompts[trace.problem_id], trace, every)
            detail.mode[:] = detail.expected
        planned[i, : trace.n_steps] = detail.expected
    stats = consistency_stats(planned, [trace.n_steps for trace in traces], censored=censored)
    return stats, tuple(details)


def run_elicitation_pilot(
    backend: ReasoningBackend,
    traces: TraceSet,
    prompts: Mapping[str, str],
    every: int = 1,
    include_censored: bool = False,
    max_steps: int | None = None,
    method: Literal["distribution", "greedy"] = "distribution",
    min_mass: float = 0.0,
    max_k: int = 30,
) -> ConsistencyStats:
    """Return consistency statistics for distributional or greedy step forecasts."""
    return run_elicitation_pilot_detailed(
        backend, traces, prompts, every, include_censored, max_steps, method, min_mass, max_k
    )[0]
