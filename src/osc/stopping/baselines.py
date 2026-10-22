"""Observable stopping heuristics and supervised correctness probes.

Correctness is used only for fitting and offline parameter selection. Fitted
probe predictions and all heuristic decisions use observable state only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from osc.stopping.payoff import PayoffPanel, evaluate_stop_times


class BaselineError(ValueError):
    """Invalid baseline configuration or observable inputs."""


def full_trace_stop_times(panel: PayoffPanel) -> np.ndarray:
    return panel.horizons - 1


def token_budget_stop_times(panel: PayoffPanel, budget: float) -> np.ndarray:
    """Last affordable state, falling back to the empty prefix at step zero."""
    if not np.isfinite(budget) or budget < 0:
        raise BaselineError("budget must be finite and non-negative")
    affordable = panel.alive & (panel.cum_tokens <= budget)
    return np.maximum(np.where(affordable, np.arange(panel.t_max), -1).max(axis=1), 0)


def answer_convergence_stop_times(panel: PayoffPanel, k: int) -> np.ndarray:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise BaselineError("k must be a positive integer")
    times = full_trace_stop_times(panel)
    for i, h in enumerate(panel.horizons):
        for t in range(k - 1, h):
            window = panel.answer_ids[i, t - k + 1 : t + 1]
            if window[0] != -1 and np.all(window == window[0]):
                times[i] = t
                break
    return times


def _features(panel: PayoffPanel) -> np.ndarray:
    features = panel.features[panel.alive]
    if features.shape[1] == 0 or not np.isfinite(features).all():
        raise BaselineError("alive features must be finite with at least one column")
    return features


@dataclass(frozen=True)
class CorrectnessProbe:
    """Fitted feature-only predictor; constant when training has one class."""

    model: Pipeline | None
    constant: float | None
    feature_dim: int

    def scores(self, panel: PayoffPanel) -> np.ndarray:
        features = _features(panel)
        if features.shape[1] != self.feature_dim:
            raise BaselineError("probe feature dimension does not match panel")
        scores = np.full(panel.alive.shape, np.nan)
        scores[panel.alive] = (
            self.model.predict_proba(features)[:, 1] if self.model is not None else self.constant
        )
        return scores


def fit_correctness_probe(
    panel: PayoffPanel, n_components: int | None = 32, C: float = 1.0
) -> CorrectnessProbe:
    if n_components is not None and (
        not isinstance(n_components, (int, np.integer))
        or isinstance(n_components, bool)
        or n_components < 1
    ):
        raise BaselineError("n_components must be a positive integer or None")
    if not np.isfinite(C) or C <= 0:
        raise BaselineError("C must be finite and positive")
    features = _features(panel)
    labels = (panel.accuracy[panel.alive] >= 0.5).astype(int)
    if np.unique(labels).size == 1:
        return CorrectnessProbe(None, float(labels[0]), features.shape[1])
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(
            ("pca", PCA(n_components=min(n_components, *features.shape), svd_solver="full"))
        )
    steps.append(("logistic", LogisticRegression(C=C, max_iter=1000, random_state=0)))
    model = Pipeline(steps).fit(features, labels)
    return CorrectnessProbe(model, None, features.shape[1])


def cross_fit_probe_scores(panel: PayoffPanel, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions, holding out whole problem IDs to avoid leakage."""
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise BaselineError("n_folds must be an integer at least two")
    ids = np.array([tr.problem_id for tr in panel.traces])
    groups = np.unique(ids)
    if len(groups) < 2:
        raise BaselineError("cross-fitting requires at least two distinct problem IDs")
    rng = np.random.default_rng(seed)
    scores = np.full(panel.alive.shape, np.nan)
    for held_out in np.array_split(rng.permutation(groups), min(n_folds, len(groups))):
        test = np.flatnonzero(np.isin(ids, held_out))
        train = np.flatnonzero(~np.isin(ids, held_out))
        predicted = fit_correctness_probe(panel.subset(train)).scores(panel.subset(test))
        scores[test, : predicted.shape[1]] = predicted
    return scores


def probe_threshold_stop_times(
    panel: PayoffPanel, scores: np.ndarray, threshold: float
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != panel.alive.shape or not np.isfinite(scores[panel.alive]).all():
        raise BaselineError("scores must match the panel shape and be finite on alive states")
    if not np.isfinite(threshold):
        raise BaselineError("threshold must be finite")
    hits = panel.alive & (scores >= threshold)
    hits[np.arange(panel.n_traces), panel.horizons - 1] = True
    return hits.argmax(axis=1)


def select_param(
    panel: PayoffPanel,
    candidates: Sequence[float],
    stop_fn: Callable[[PayoffPanel, float], np.ndarray],
) -> float:
    """Offline validation-set tuning; first candidate wins exact value ties."""
    if len(candidates) == 0 or not np.isfinite(candidates).all():
        raise BaselineError("candidates must be non-empty and finite")
    values = [evaluate_stop_times(panel, stop_fn(panel, c), "validation").value for c in candidates]
    return float(candidates[int(np.argmax(values))])
