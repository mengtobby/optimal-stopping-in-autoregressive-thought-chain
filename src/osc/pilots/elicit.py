"""Elicit remaining steps and measure whether predicted stopping times recede."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from osc.collect.protocol import ReasoningBackend
from osc.data.schema import Trace, TraceSet
from osc.pilots.flips import step_ends, thinking_text

DEFAULT_PROBE = (
    "How many more reasoning steps do you need before giving your final answer? "
    "Reply with one integer from 0 to 10000 only."
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
        value = parse_int(backend.complete(prompt, thinking[: ends[t]], probe, max_tokens))
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


def run_elicitation_pilot(
    backend: ReasoningBackend,
    traces: TraceSet,
    prompts: Mapping[str, str],
    every: int = 1,
    include_censored: bool = False,
    max_steps: int | None = None,
) -> ConsistencyStats:
    """Probe completed traces; optionally include traces censored by collection limits."""
    if every < 1:
        raise ElicitationError("every must be positive")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1
    ):
        raise ElicitationError("max_steps must be a positive integer")
    censored = [
        not include_censored
        and (not trace.natural_stop or (max_steps is not None and trace.n_steps >= max_steps))
        for trace in traces
    ]
    planned = np.full((len(traces), traces.max_steps), np.nan)
    for i, trace in enumerate(traces):
        if censored[i]:
            continue
        if trace.problem_id not in prompts:
            raise ElicitationError(f"missing prompt for {trace.problem_id!r}")
        planned[i, : trace.n_steps] = elicit_remaining(
            backend, prompts[trace.problem_id], trace, every
        )
    return consistency_stats(planned, [trace.n_steps for trace in traces], censored=censored)
