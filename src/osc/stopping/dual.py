"""Nested-simulation martingale upper bounds for reasoning stopping problems."""

from collections.abc import Callable, Sequence
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


def scaled_value_fn(value_fn: ValueFn, c: float) -> ValueFn:
    """Scale a fixed value function; zero recovers the hindsight martingale."""
    if not callable(value_fn) or not np.isfinite(c) or c < 0:
        raise DualError("value_fn must be callable and c finite and non-negative")
    if c == 0:
        return zero_value_fn

    def scaled(t: int, features: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        return c * _values(value_fn, t, features, tokens)

    return scaled


def _validate_folds(panel: PayoffPanel, folds: np.ndarray, k: int) -> np.ndarray:
    folds = np.asarray(folds)
    if (
        folds.shape != (panel.n_traces,)
        or folds.dtype.kind not in "iu"
        or k < 2
        or not np.array_equal(np.unique(folds), np.arange(k))
    ):
        raise DualError("folds must assign every trace to contiguous nonempty folds 0..K-1")
    groups: dict[str, int] = {}
    for trace, fold in zip(panel.traces, folds, strict=True):
        if groups.setdefault(trace.problem_id, int(fold)) != fold:
            raise DualError("all traces from a problem must share a fold")
    return folds


def _inner_assignments(panel: PayoffPanel, k: int, seed: int) -> np.ndarray | None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 2:
        raise DualError("inner_folds must be an integer at least 2")
    ids = list(dict.fromkeys(trace.problem_id for trace in panel.traces))
    if len(ids) < k:
        return None
    order = np.random.default_rng(seed).permutation(len(ids))
    assignments = {ids[index]: rank % k for rank, index in enumerate(order)}
    return np.array([assignments[trace.problem_id] for trace in panel.traces])


def cross_fit_shrunk_dual(
    panel: PayoffPanel,
    folds: np.ndarray,
    value_fns_by_fold: Sequence[ValueFn],
    fit_value_fn: Callable[[PayoffPanel], ValueFn] | Sequence[Callable[[PayoffPanel], ValueFn]],
    grid: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    inner_folds: int = 3,
    seed: int = 0,
) -> tuple[DualResult, tuple[float, ...]]:
    """Select shrinkage on grouped inner folds, then evaluate independent outer folds.

    Each supplied outer value function must exclude that fold and its siblings.
    Inner value functions are refitted without their validation problems. Pass one
    fit_value_fn per fold to refit with that fold's own (outer-train) hyperparameters.
    """
    folds = _validate_folds(panel, folds, len(value_fns_by_fold))
    candidates = sorted(set(grid))
    if not candidates or any(not np.isfinite(c) or c < 0 for c in candidates):
        raise DualError("grid must contain finite non-negative shrink factors")
    fitters = (
        [fit_value_fn] * len(value_fns_by_fold) if callable(fit_value_fn) else list(fit_value_fn)
    )
    if len(fitters) != len(value_fns_by_fold) or not all(map(callable, fitters)):
        raise DualError("fit_value_fn must be callable or one callable per fold")
    if not all(map(callable, value_fns_by_fold)):
        raise DualError("value functions and fit_value_fn must be callable")
    values, chosen = np.empty(panel.n_traces), []
    absolute_increments = 0.0
    for f, vf in enumerate(value_fns_by_fold):
        train = panel.subset(np.flatnonzero(folds != f))
        inner = _inner_assignments(train, inner_folds, seed + f)
        c = 0.0
        if inner is not None:
            scores = np.zeros(len(candidates))
            for g in range(inner_folds):
                fitted = fitters[f](train.subset(np.flatnonzero(inner != g)))
                validation = train.subset(np.flatnonzero(inner == g))
                for j, candidate in enumerate(candidates):
                    scores[j] += dual_upper_bound(
                        validation, scaled_value_fn(fitted, candidate)
                    ).per_trace.sum()
            c = float(candidates[int(np.argmin(scores))])
        indices = np.flatnonzero(folds == f)
        test = panel.subset(indices)
        result = dual_upper_bound(test, scaled_value_fn(vf, c))
        values[indices] = result.per_trace
        absolute_increments += result.mean_abs_increment * np.sum(test.horizons - 1)
        chosen.append(c)
    baseline = dual_upper_bound(panel, zero_value_fn)
    values.setflags(write=False)
    transitions = int(np.sum(panel.horizons - 1))
    return DualResult(
        float(values.mean()),
        float(values.std(ddof=1) / np.sqrt(len(values))),
        values,
        baseline.martingale_coverage,
        float(absolute_increments / transitions) if transitions else 0.0,
    ), tuple(chosen)


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
