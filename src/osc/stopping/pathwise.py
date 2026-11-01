"""Cross-fitted pathwise LP optimization over sibling-centered martingales.

Fixed basis functions and weights independent of evaluation paths give a
martingale M, so E max(Z - M) bounds every observable stopping rule. The
in-sample LP objective is optimistic; only held-out values should be reported.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from osc.stopping.dual import (
    DualError,
    DualResult,
    ValueFn,
    _inner_assignments,
    _validate_folds,
    martingale_increments,
)
from osc.stopping.payoff import PayoffPanel


class PathwiseError(ValueError):
    """Invalid pathwise inputs or unsuccessful LP optimization."""


@dataclass(frozen=True)
class PathwiseConfig:
    n_components: int | None = 8
    n_time_bins: int = 4
    beta_bound: float = 1.0
    include_tokens: bool = True
    l1_penalties: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
    inner_folds: int = 3

    def __post_init__(self) -> None:
        if self.n_components is not None and (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, (int, np.integer))
            or self.n_components < 0
        ):
            raise PathwiseError("n_components must be a non-negative integer or None")
        if (
            isinstance(self.n_time_bins, bool)
            or not isinstance(self.n_time_bins, (int, np.integer))
            or self.n_time_bins < 1
        ):
            raise PathwiseError("n_time_bins must be a positive integer")
        if not np.isfinite(self.beta_bound) or self.beta_bound <= 0:
            raise PathwiseError("beta_bound must be finite and positive")
        if not self.l1_penalties or any(not np.isfinite(p) or p < 0 for p in self.l1_penalties):
            raise PathwiseError("l1_penalties must contain finite non-negative penalties")
        if (
            isinstance(self.inner_folds, bool)
            or not isinstance(self.inner_folds, (int, np.integer))
            or self.inner_folds < 2
        ):
            raise PathwiseError("inner_folds must be an integer at least 2")
        if not isinstance(self.include_tokens, bool):
            raise PathwiseError("include_tokens must be boolean")


@dataclass(frozen=True)
class PathwiseBasis:
    names: tuple[str, ...]
    value_fns: tuple[ValueFn, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.value_fns) or not all(map(callable, self.value_fns)):
            raise PathwiseError("basis names and callable value functions must match")


def _finite_values(fn: ValueFn, t: int, x: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(fn(t, x, tokens), dtype=float)
    if values.shape != (len(x),):
        raise PathwiseError("basis function must return one value per row")
    return np.where(np.isfinite(values), values, 0.0)


_DEFAULT_CONFIG = PathwiseConfig()


def build_basis(
    panel_train: PayoffPanel,
    anchors: Sequence[tuple[str, ValueFn]] = (),
    config: PathwiseConfig = _DEFAULT_CONFIG,
) -> PathwiseBasis:
    """Fit observable PCA and basis scales using only training alive states.

    None retains all available PCA components; zero disables PCA terms. Anchors
    must themselves have been fitted independently of the evaluation fold.
    """
    x = panel_train.features[panel_train.alive].astype(float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise PathwiseError("training panel must contain finite alive features")
    raw = list(anchors)
    count = min(x.shape[1], len(x) - 1)
    if config.n_components is not None:
        count = min(count, config.n_components)
    if not np.any(np.var(x, axis=0) > 0):
        count = 0
    if count:
        scaler = StandardScaler().fit(x)
        pca = PCA(n_components=count, svd_solver="full").fit(scaler.transform(x))
        for j in range(count):

            def score(t: int, features: np.ndarray, tokens: np.ndarray, j: int = j) -> np.ndarray:
                return pca.transform(scaler.transform(features))[:, j]

            raw.append((f"pca-{j}", score))
    if config.include_tokens:
        raw.append(("tokens", lambda t, features, tokens: np.asarray(tokens) / 1000.0))
    names, functions = [], []
    for name, fn in raw:
        if not isinstance(name, str) or not callable(fn):
            raise PathwiseError("anchors must contain names and callable value functions")
        for time_bin in range(config.n_time_bins):

            def interacted(
                t: int,
                features: np.ndarray,
                tokens: np.ndarray,
                fn: ValueFn = fn,
                time_bin: int = time_bin,
            ) -> np.ndarray:
                bucket = min(t * config.n_time_bins // panel_train.t_max, config.n_time_bins - 1)
                if bucket != time_bin:
                    return np.zeros(len(features))
                return _finite_values(fn, t, features, tokens)

            values = np.concatenate(
                [
                    interacted(
                        t,
                        panel_train.features[panel_train.alive[:, t], t],
                        panel_train.cum_tokens[panel_train.alive[:, t], t],
                    )
                    for t in range(panel_train.t_max)
                ]
            )
            scale = float(values.std())
            if scale == 0:
                continue
            if not np.isfinite(scale):
                raise PathwiseError("basis scale must be finite")

            def scaled(
                t: int,
                features: np.ndarray,
                tokens: np.ndarray,
                fn: ValueFn = interacted,
                scale: float = scale,
            ) -> np.ndarray:
                return fn(t, features, tokens) / scale

            names.append(name if config.n_time_bins == 1 else f"{name}:bin-{time_bin}")
            functions.append(scaled)
    return PathwiseBasis(tuple(names), tuple(functions))


def basis_martingales(panel: PayoffPanel, basis: PathwiseBasis) -> np.ndarray:
    """Return cumulative martingales of shape (J, N, T), starting at zero."""
    result = np.zeros((len(basis.value_fns), panel.n_traces, panel.t_max))
    for j, fn in enumerate(basis.value_fns):
        increments = martingale_increments(panel, fn)
        result[j, :, 1:] = np.cumsum(increments[:, :-1], axis=1)
    return result


def fit_pathwise_beta(
    panel_train: PayoffPanel,
    basis: PathwiseBasis,
    config: PathwiseConfig = _DEFAULT_CONFIG,
    l1_penalty: float = 0.0,
) -> np.ndarray:
    """Minimize mean training max-payoff plus an L1 penalty using a sparse LP."""
    if not np.isfinite(l1_penalty) or l1_penalty < 0:
        raise PathwiseError("l1_penalty must be finite and non-negative")
    j = len(basis.value_fns)
    if not j:
        return np.empty(0)
    return _fit_beta(panel_train, basis_martingales(panel_train, basis), config, l1_penalty)


def _fit_beta(
    panel_train: PayoffPanel,
    martingales: np.ndarray,
    config: PathwiseConfig,
    l1_penalty: float,
) -> np.ndarray:
    j = len(martingales)
    rows, times = np.nonzero(panel_train.alive)
    n, constraints = panel_train.n_traces, len(rows)
    u = coo_matrix((-np.ones(constraints), (np.arange(constraints), rows)), (constraints, n))
    m = csr_matrix(martingales[:, rows, times].T)
    matrix = hstack((u, -m, m), format="csr")
    result = linprog(
        np.r_[np.full(n, 1 / n), np.full(2 * j, l1_penalty)],
        A_ub=matrix,
        b_ub=-panel_train.payoff[rows, times],
        bounds=[(None, None)] * n + [(0, config.beta_bound)] * (2 * j),
        method="highs",
    )
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        raise PathwiseError(f"pathwise LP failed: {result.message}")
    return np.asarray(result.x[n : n + j] - result.x[n + j :]).copy()


def _result(panel: PayoffPanel, values: np.ndarray, absolute_increments: float) -> DualResult:
    transitions = int(np.sum(panel.horizons - 1))
    covered = sum(s is not None for trace in panel.traces for s in trace.siblings[:-1])
    values.setflags(write=False)
    return DualResult(
        float(values.mean()),
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        values,
        covered / transitions if transitions else 0.0,
        absolute_increments / transitions if transitions else 0.0,
    )


def pathwise_dual(panel_eval: PayoffPanel, basis: PathwiseBasis, beta: np.ndarray) -> DualResult:
    """Evaluate fixed independent basis/weights; E max(Z-M) is a valid upper bound."""
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (len(basis.value_fns),) or not np.all(np.isfinite(beta)):
        raise PathwiseError("beta must be a finite vector with one entry per basis function")
    martingale = np.einsum("j,jnt->nt", beta, basis_martingales(panel_eval, basis))
    values = np.nanmax(panel_eval.payoff - martingale, axis=1)
    return _result(panel_eval, values, float(np.abs(np.diff(martingale, axis=1)).sum()))


def cross_fit_pathwise_dual(
    panel: PayoffPanel,
    folds: np.ndarray,
    anchors_by_fold: Sequence[Sequence[tuple[str, ValueFn]]],
    config: PathwiseConfig = _DEFAULT_CONFIG,
) -> tuple[DualResult, tuple[np.ndarray, ...]]:
    """Fit basis and beta without each held-out fold and assemble panel-order values.

    Callers must fit anchors_by_fold[f] without fold f, including its siblings.
    Problem groups must stay within a fold. Only out-of-fold objectives are
    reported because optimizing the in-sample objective is optimistic.
    """
    try:
        folds = _validate_folds(panel, folds, len(anchors_by_fold))
    except DualError as exc:
        raise PathwiseError(str(exc)) from exc
    values, betas = np.empty(panel.n_traces), []
    absolute_increments = 0.0
    for fold, anchors in enumerate(anchors_by_fold):
        train = panel.subset(np.flatnonzero(folds != fold))
        indices = np.flatnonzero(folds == fold)
        test = panel.subset(indices)
        basis = build_basis(train, anchors, config)
        inner = _inner_assignments(train, config.inner_folds, seed=fold)
        penalties = sorted(set(config.l1_penalties))
        penalty = max(penalties)
        if inner is not None and len(penalties) > 1 and basis.value_fns:
            scores = np.zeros(len(penalties))
            for g in range(config.inner_folds):
                inner_train = train.subset(np.flatnonzero(inner != g))
                validation = train.subset(np.flatnonzero(inner == g))
                # Rebuild PCA/scales without validation paths. Anchors were fitted on the
                # whole outer-train fold, which only biases selection, never validity.
                inner_basis = build_basis(inner_train, anchors, config)
                if not inner_basis.value_fns:
                    continue
                train_martingales = basis_martingales(inner_train, inner_basis)
                validation_martingales = basis_martingales(validation, inner_basis)
                for j, candidate in enumerate(penalties):
                    beta = _fit_beta(inner_train, train_martingales, config, candidate)
                    m = np.einsum("j,jnt->nt", beta, validation_martingales)
                    scores[j] += np.nanmax(
                        validation.payoff - m[:, : validation.t_max], axis=1
                    ).sum()
            penalty = penalties[int(np.argmin(scores))]
        beta = fit_pathwise_beta(train, basis, config, penalty)
        result = pathwise_dual(test, basis, beta)
        values[indices] = result.per_trace
        absolute_increments += result.mean_abs_increment * np.sum(test.horizons - 1)
        betas.append(beta)
    return _result(panel, values, float(absolute_increments)), tuple(betas)
