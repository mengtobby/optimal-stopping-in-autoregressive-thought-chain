"""Longstaff--Schwartz stopping with trace-level cross-fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from osc.stopping.payoff import PayoffPanel


class LSMError(ValueError):
    """Invalid LSM configuration or observable inputs."""


@dataclass(frozen=True)
class LSMConfig:
    n_components: int | None = 32
    ridge_alpha: float = 1.0
    min_samples: int = 20

    def __post_init__(self) -> None:
        if self.n_components is not None and (
            not isinstance(self.n_components, int) or self.n_components < 1
        ):
            raise LSMError("n_components must be a positive integer or None")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise LSMError("ridge_alpha must be finite and non-negative")
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise LSMError("min_samples must be a positive integer")


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
        if len(x) and self.scaler is not None:
            x = self.scaler.transform(x)
            if self.pca is not None:
                x = self.pca.transform(x)
        elif self.pca is not None:
            x = np.empty((0, self.pca.n_components_))
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
        for t in range(panel.t_max - 1):
            rows = np.flatnonzero((taus > t) & (panel.horizons > t + 1))
            x, tokens = panel.features[rows, t], panel.cum_tokens[rows, t]
            stop = self.payoff_estimate(t, x, tokens) >= self.continuation(t, x, tokens)
            taus[rows[stop]] = t
        return taus


_DEFAULT_CONFIG = LSMConfig()


def fit_lsm(panel: PayoffPanel, config: LSMConfig = _DEFAULT_CONFIG) -> LSMPolicy:
    """Fit preprocessing and realized-cashflow backward regressions on training traces."""
    alive = panel.alive
    x = panel.features[alive].astype(float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise LSMError("training panel must contain finite alive features")
    scaler = StandardScaler().fit(x) if x.shape[1] else None
    scaled = scaler.transform(x) if scaler is not None else x
    pca = None
    if config.n_components is not None and min(x.shape[1], len(x) - 1) > 0:
        pca = PCA(
            n_components=min(config.n_components, x.shape[1], len(x) - 1), svd_solver="full"
        ).fit(scaled)
        scaled = pca.transform(scaled)
    times = np.broadcast_to(np.arange(panel.t_max), alive.shape)[alive]
    basis = np.column_stack(
        (scaled, times / panel.t_max, panel.cum_tokens[alive] / panel.token_scale)
    )
    head = Ridge(alpha=config.ridge_alpha).fit(basis, panel.accuracy[alive])
    policy = LSMPolicy(scaler, pca, head, (), x.shape[1], panel.t_max, panel.lam, panel.token_scale)
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
    )


def cross_fit_lsm(
    panel: PayoffPanel, config: LSMConfig = _DEFAULT_CONFIG, n_folds: int = 5, seed: int = 0
) -> tuple[np.ndarray, tuple[LSMPolicy, ...], np.ndarray]:
    """Return held-out stop times, fold policies, and each trace's fold index."""
    if not isinstance(n_folds, int) or not 2 <= n_folds <= panel.n_traces:
        raise LSMError("n_folds must be an integer between 2 and the number of traces")
    folds = np.empty(panel.n_traces, dtype=np.int64)
    permutation = np.random.default_rng(seed).permutation(panel.n_traces)
    folds[permutation] = np.arange(panel.n_traces) % n_folds
    taus = np.empty(panel.n_traces, dtype=np.int64)
    policies = []
    for fold in range(n_folds):
        policy = fit_lsm(panel.subset(np.flatnonzero(folds != fold)), config)
        held_out = np.flatnonzero(folds == fold)
        taus[held_out] = policy.stop_times(panel.subset(held_out))
        policies.append(policy)
    return taus, tuple(policies), folds
