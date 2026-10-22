"""Nested-simulation martingale upper bounds for reasoning stopping problems."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from osc.stopping.payoff import PayoffPanel

ValueFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


class DualError(ValueError):
    """Invalid value-function output."""


@dataclass(frozen=True)
class DualResult:
    upper: float
    upper_se: float
    per_trace: np.ndarray
    martingale_coverage: float
    mean_abs_increment: float


def zero_value_fn(t: int, features: np.ndarray, cum_tokens: np.ndarray) -> np.ndarray:
    """Zero martingale, recovering the hindsight prophet bound."""
    return np.zeros(len(features))


def _values(value_fn: ValueFn, t: int, features: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(value_fn(t, features, tokens), dtype=float)
    if values.shape != (len(features),) or not np.all(np.isfinite(values)):
        raise DualError("value_fn must return one finite value per input row")
    return values


def martingale_increments(panel: PayoffPanel, value_fn: ValueFn) -> np.ndarray:
    """Return centered next-state values, batched by time.

    For any fixed integrable value_fn depending only on the state at t+1,
    outer and independent sibling draws share the next-step law. Their difference
    has conditional mean zero. Missing siblings give zero increments and only
    loosen the bound (in expectation); they do not invalidate it. The function
    must not be fitted using these evaluation paths or sibling draws.
    """
    increments = np.zeros_like(panel.payoff)
    for t in range(panel.t_max - 1):
        rows, siblings = [], []
        for i, trace in enumerate(panel.traces):
            if t < trace.n_steps - 1 and trace.siblings and trace.siblings[t] is not None:
                rows.append(i)
                siblings.append(trace.siblings[t])
        if not rows:
            continue
        outer = _values(value_fn, t + 1, panel.features[rows, t + 1], panel.cum_tokens[rows, t + 1])
        features = np.concatenate([s.features for s in siblings])
        tokens = np.concatenate(
            [panel.cum_tokens[i, t] + s.step_tokens for i, s in zip(rows, siblings, strict=True)]
        )
        inner = _values(value_fn, t + 1, features, tokens)
        offset = 0
        for j, (i, sibling) in enumerate(zip(rows, siblings, strict=True)):
            increments[i, t] = outer[j] - inner[offset : offset + sibling.k].mean()
            offset += sibling.k
    return increments


def dual_upper_bound(panel: PayoffPanel, value_fn: ValueFn) -> DualResult:
    """Estimate E max(Z-M), an upper bound for every observable stopping policy.

    Any fixed integrable next-state value_fn is valid because siblings share the
    outer next-step law, making M a martingale. Missing siblings only loosen the
    bound in expectation. Finite-sample estimates can fall below the true value.
    """
    increments = martingale_increments(panel, value_fn)
    martingale = np.zeros_like(increments)
    martingale[:, 1:] = np.cumsum(increments[:, :-1], axis=1)
    per_trace = np.nanmax(panel.payoff - martingale, axis=1)
    transitions = int(np.sum(panel.horizons - 1))
    covered = sum(s is not None for trace in panel.traces for s in trace.siblings[:-1])
    per_trace.setflags(write=False)
    return DualResult(
        upper=float(per_trace.mean()),
        upper_se=float(per_trace.std(ddof=1) / np.sqrt(len(per_trace)))
        if len(per_trace) > 1
        else 0.0,
        per_trace=per_trace,
        martingale_coverage=covered / transitions if transitions else 0.0,
        mean_abs_increment=float(np.abs(increments).sum() / transitions) if transitions else 0.0,
    )
