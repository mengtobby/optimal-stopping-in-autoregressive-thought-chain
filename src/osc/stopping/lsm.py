"""Longstaff--Schwartz stopping with problem-grouped cross-fitting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from osc.stopping.payoff import PayoffPanel, evaluate_stop_times


class LSMError(ValueError):
    """Invalid LSM configuration or observable inputs."""


@dataclass(frozen=True)
class LSMConfig:
    n_components: int | None = 32
    ridge_alpha: float = 1.0
    min_samples: int = 20
    margins: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2, inf)
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0)
    inner_folds: int = 3
    passthrough: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.passthrough, int)
            or isinstance(self.passthrough, bool)
            or self.passthrough < 0
        ):
            raise LSMError("passthrough must be a non-negative integer")
        if self.n_components is not None and (
            not isinstance(self.n_components, int) or self.n_components < 1
        ):
            raise LSMError("n_components must be a positive integer or None")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise LSMError("ridge_alpha must be finite and non-negative")
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise LSMError("min_samples must be a positive integer")
        if not self.margins or any(np.isnan(m) or m < 0 for m in self.margins):
            raise LSMError("margins must be non-empty and non-negative (infinity is allowed)")
        if not self.ridge_alphas or any(not np.isfinite(a) or a < 0 for a in self.ridge_alphas):
            raise LSMError("ridge_alphas must be non-empty, finite and non-negative")
        if not isinstance(self.inner_folds, int) or self.inner_folds < 2:
            raise LSMError("inner_folds must be an integer >= 2")


@dataclass(frozen=True)
class LSMPolicy:
    scaler: StandardScaler | None
    pca: PCA | None
    payoff_model: Ridge
    continuation_models: tuple[Ridge | float, ...]
    feature_dim: int
    t_max_train: int
    lam: float
    token_scale: float
    margin: float = 0.0
    passthrough: int = 0
    passthrough_scaler: StandardScaler | None = None

    def _basis(self, t: int, features: np.ndarray, cum_tokens: np.ndarray) -> np.ndarray:
        if not isinstance(t, (int, np.integer)) or t < 0:
            raise LSMError("t must be a non-negative integer")
        x = np.asarray(features, dtype=float)
        tokens = np.asarray(cum_tokens, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.feature_dim or tokens.shape != (len(x),):
            raise LSMError("features must have shape (n, d) and cum_tokens shape (n,)")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(tokens)):
            raise LSMError("features and cum_tokens must be finite")
        if np.any(tokens < 0):
            raise LSMError("cum_tokens must be non-negative")
        signals = x[:, self.feature_dim - self.passthrough :]
        x = x[:, : self.feature_dim - self.passthrough]
        if len(x) and self.scaler is not None:
            x = self.scaler.transform(x)
            if self.pca is not None:
                x = self.pca.transform(x)
        elif self.pca is not None:
            x = np.empty((0, self.pca.n_components_))
        if self.passthrough_scaler is not None:
            signals = self.passthrough_scaler.transform(signals) if len(x) else signals
            x = np.column_stack((x, signals))
        return np.column_stack(
            (x, np.full(len(x), t / self.t_max_train), tokens / self.token_scale)
        )

    def payoff_estimate(self, t: int, features: np.ndarray, cum_tokens: np.ndarray) -> np.ndarray:
        basis = self._basis(t, features, cum_tokens)
        if not len(basis):
            return np.empty(0)
        return np.clip(self.payoff_model.predict(basis), 0, 1) - (
            self.lam * np.asarray(cum_tokens) / self.token_scale
        )

    def continuation(self, t: int, features: np.ndarray, cum_tokens: np.ndarray) -> np.ndarray:
        basis = self._basis(t, features, cum_tokens)
        if t >= len(self.continuation_models):
            return np.full(len(basis), -np.inf)
        model = self.continuation_models[t]
        if isinstance(model, float):
            return np.full(len(basis), model)
        return model.predict(basis) if len(basis) else np.empty(0)

    def value(self, t: int, features: np.ndarray, cum_tokens: np.ndarray) -> np.ndarray:
        return np.maximum(
            self.payoff_estimate(t, features, cum_tokens),
            self.continuation(t, features, cum_tokens),
        )

    def stop_times(self, panel: PayoffPanel) -> np.ndarray:
        taus = panel.horizons.copy() - 1
        if self.margin == inf:
            return taus
        for t in range(panel.t_max - 1):
            rows = np.flatnonzero((taus > t) & (panel.horizons > t + 1))
            x, tokens = panel.features[rows, t], panel.cum_tokens[rows, t]
            stop = self.payoff_estimate(t, x, tokens) >= (
                self.continuation(t, x, tokens) + self.margin
            )
            taus[rows[stop]] = t
        return taus


_DEFAULT_CONFIG = LSMConfig()


def fit_lsm(panel: PayoffPanel, config: LSMConfig = _DEFAULT_CONFIG) -> LSMPolicy:
    """Fit preprocessing and realized-cashflow backward regressions on training traces."""
    alive = panel.alive
    x = panel.features[alive].astype(float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise LSMError("training panel must contain finite alive features")
    if config.passthrough > x.shape[1]:
        raise LSMError("passthrough must not exceed the feature dimension")
    leading = x[:, : x.shape[1] - config.passthrough]
    scaler = StandardScaler().fit(leading) if leading.shape[1] else None
    scaled = scaler.transform(leading) if scaler is not None else leading
    pca = None
    if config.n_components is not None and min(leading.shape[1], len(x) - 1) > 0:
        pca = PCA(
            n_components=min(config.n_components, leading.shape[1], len(x) - 1),
            svd_solver="full",
        ).fit(scaled)
        scaled = pca.transform(scaled)
    passthrough_scaler = None
    if config.passthrough:
        signals = x[:, -config.passthrough :]
        passthrough_scaler = StandardScaler().fit(signals)
        scaled = np.column_stack((scaled, passthrough_scaler.transform(signals)))
    times = np.broadcast_to(np.arange(panel.t_max), alive.shape)[alive]
    basis = np.column_stack(
        (scaled, times / panel.t_max, panel.cum_tokens[alive] / panel.token_scale)
    )
    head = Ridge(alpha=config.ridge_alpha).fit(basis, panel.accuracy[alive])
    policy = LSMPolicy(
        scaler,
        pca,
        head,
        (),
        x.shape[1],
        panel.t_max,
        panel.lam,
        panel.token_scale,
        passthrough=config.passthrough,
        passthrough_scaler=passthrough_scaler,
    )
    cashflow = panel.payoff[np.arange(panel.n_traces), panel.horizons - 1].copy()
    models: list[Ridge | float] = []
    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    for t in range(panel.t_max - 2, -1, -1):
        rows = np.flatnonzero(panel.horizons > t + 1)
        current = policy._basis(t, panel.features[rows, t], panel.cum_tokens[rows, t])
        if len(rows):
            pooled_x.append(current)
            pooled_y.append(cashflow[rows].copy())
        train_x, train_y = current, cashflow[rows]
        if len(rows) < config.min_samples and pooled_x:
            train_x, train_y = np.concatenate(pooled_x), np.concatenate(pooled_y)
        model = (
            Ridge(alpha=config.ridge_alpha).fit(train_x, train_y)
            if len(train_y)
            else float(cashflow.mean())
        )
        models.append(model)
        continuation = model.predict(current) if isinstance(model, Ridge) else model
        estimate = policy.payoff_estimate(t, panel.features[rows, t], panel.cum_tokens[rows, t])
        stopped = rows[estimate >= continuation]
        cashflow[stopped] = panel.payoff[stopped, t]
    return LSMPolicy(
        scaler,
        pca,
        head,
        tuple(reversed(models)),
        x.shape[1],
        panel.t_max,
        panel.lam,
        panel.token_scale,
        passthrough=config.passthrough,
        passthrough_scaler=passthrough_scaler,
    )


def _problem_folds(panel: PayoffPanel, n_folds: int, seed: int) -> np.ndarray:
    problem_ids = [trace.problem_id for trace in panel.traces]
    unique_ids = list(dict.fromkeys(problem_ids))
    if not isinstance(n_folds, int) or not 2 <= n_folds <= len(unique_ids):
        raise LSMError("n_folds must be an integer between 2 and the number of problems")
    permutation = np.random.default_rng(seed).permutation(len(unique_ids))
    assignments = {unique_ids[index]: rank % n_folds for rank, index in enumerate(permutation)}
    return np.array([assignments[problem_id] for problem_id in problem_ids], dtype=np.int64)


def select_lsm_hyperparams(
    panel: PayoffPanel, config: LSMConfig = _DEFAULT_CONFIG, seed: int = 0
) -> tuple[float, float]:
    """Select alpha and stopping margin using only problem-grouped inner held-out paths."""
    if len({trace.problem_id for trace in panel.traces}) < config.inner_folds:
        return config.ridge_alpha, inf
    folds = _problem_folds(panel, config.inner_folds, seed)
    scores = {(alpha, margin): 0.0 for alpha in config.ridge_alphas for margin in config.margins}
    for fold in range(config.inner_folds):
        train = panel.subset(np.flatnonzero(folds != fold))
        held_out = panel.subset(np.flatnonzero(folds == fold))
        for alpha in config.ridge_alphas:
            policy = fit_lsm(train, replace(config, ridge_alpha=alpha))
            for margin in config.margins:
                candidate = replace(policy, margin=margin)
                result = evaluate_stop_times(held_out, candidate.stop_times(held_out), "lsm")
                scores[alpha, margin] += result.value * held_out.n_traces / panel.n_traces
    return max(scores, key=lambda pair: (scores[pair], pair[1], pair[0]))


def cross_fit_lsm(
    panel: PayoffPanel, config: LSMConfig = _DEFAULT_CONFIG, n_folds: int = 5, seed: int = 0
) -> tuple[np.ndarray, tuple[LSMPolicy, ...], np.ndarray]:
    """Return held-out stop times and policies tuned solely on each outer training fold.

    ridge_alpha and margin are selected from config.ridge_alphas and config.margins;
    config.ridge_alpha is only the fallback when a fold has too few problems to tune.
    """
    folds = _problem_folds(panel, n_folds, seed)
    taus = np.empty(panel.n_traces, dtype=np.int64)
    policies = []
    for fold in range(n_folds):
        train = panel.subset(np.flatnonzero(folds != fold))
        alpha, margin = select_lsm_hyperparams(train, config, seed + fold)
        policy = replace(fit_lsm(train, replace(config, ridge_alpha=alpha)), margin=margin)
        held_out = np.flatnonzero(folds == fold)
        taus[held_out] = policy.stop_times(panel.subset(held_out))
        policies.append(policy)
    return taus, tuple(policies), folds
