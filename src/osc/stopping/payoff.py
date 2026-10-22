"""Padded payoff panel and stop-time evaluation shared by all stopping methods.

Payoff for stopping trace ``i`` at step ``t``:

    Z[i, t] = accuracy[i, t] - lam * cum_tokens[i, t] / token_scale

so ``lam`` is the accuracy a policy is willing to give up per ``token_scale`` tokens.
Arrays are padded to ``T_max``; entries at ``t >= horizons[i]`` are NaN (float arrays),
0 (features) or -1 (answer ids). ``accuracy`` is ground truth: a policy's *decisions*
may only use observables (``features``, ``cum_tokens``, ``answer_ids``, ``t``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osc.data.schema import TraceSet


class PanelError(ValueError):
    """Raised for invalid panel construction or stop times."""


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class PayoffPanel:
    traces: TraceSet
    lam: float
    token_scale: float
    horizons: np.ndarray  # (N,) int
    payoff: np.ndarray  # (N, T_max) float64, NaN-padded
    accuracy: np.ndarray  # (N, T_max) float64, NaN-padded
    cum_tokens: np.ndarray  # (N, T_max) float64, NaN-padded
    features: np.ndarray  # (N, T_max, d) float32, zero-padded
    answer_ids: np.ndarray  # (N, T_max) int64; -1 = no answer / padding

    @property
    def n_traces(self) -> int:
        return int(self.horizons.shape[0])

    @property
    def t_max(self) -> int:
        return int(self.payoff.shape[1])

    @property
    def alive(self) -> np.ndarray:
        """(N, T_max) bool mask: True where t < horizon."""
        return np.arange(self.t_max)[None, :] < self.horizons[:, None]

    def payoff_at(self, accuracy: np.ndarray, cum_tokens: np.ndarray) -> np.ndarray:
        """Payoff for arbitrary (accuracy, cum_tokens) under this panel's lam/token_scale."""
        return np.asarray(accuracy, float) - self.lam * np.asarray(cum_tokens, float) / (
            self.token_scale
        )

    def subset(self, indices: np.ndarray | list[int]) -> PayoffPanel:
        idx = np.asarray(indices, dtype=np.int64)
        return build_panel(self.traces.subset(idx.tolist()), self.lam, self.token_scale)


def build_panel(traces: TraceSet, lam: float, token_scale: float = 1000.0) -> PayoffPanel:
    if not np.isfinite(lam) or lam < 0:
        raise PanelError("lam must be a finite non-negative number")
    if not np.isfinite(token_scale) or token_scale <= 0:
        raise PanelError("token_scale must be positive")
    n, t_max, d = len(traces), traces.max_steps, traces.feature_dim
    horizons = np.array([tr.n_steps for tr in traces], dtype=np.int64)
    accuracy = np.full((n, t_max), np.nan)
    cum_tokens = np.full((n, t_max), np.nan)
    features = np.zeros((n, t_max, d), dtype=np.float32)
    answer_ids = np.full((n, t_max), -1, dtype=np.int64)
    for i, tr in enumerate(traces):
        h = tr.n_steps
        accuracy[i, :h] = tr.forced_correct
        cum_tokens[i, :h] = tr.cum_tokens
        features[i, :h] = tr.features
        codes: dict[str, int] = {}
        for t, ans in enumerate(tr.forced_answers):
            if ans is not None and ans.strip():
                answer_ids[i, t] = codes.setdefault(ans.strip(), len(codes))
    payoff = accuracy - lam * cum_tokens / token_scale
    return PayoffPanel(
        traces=traces,
        lam=float(lam),
        token_scale=float(token_scale),
        horizons=_readonly(horizons),
        payoff=_readonly(payoff),
        accuracy=_readonly(accuracy),
        cum_tokens=_readonly(cum_tokens),
        features=_readonly(features),
        answer_ids=_readonly(answer_ids),
    )


@dataclass(frozen=True)
class PolicyResult:
    name: str
    value: float  # mean payoff at the stop time
    value_se: float  # standard error of ``value`` across traces
    accuracy: float  # mean accuracy at the stop time
    mean_tokens: float  # mean cumulative tokens at the stop time
    stop_times: np.ndarray  # (N,) int


def evaluate_stop_times(panel: PayoffPanel, stop_times: np.ndarray, name: str) -> PolicyResult:
    """Score per-trace stop times; each must satisfy 0 <= tau < horizon."""
    taus = np.asarray(stop_times, dtype=np.int64)
    if taus.shape != (panel.n_traces,):
        raise PanelError(f"stop_times must have shape ({panel.n_traces},), got {taus.shape}")
    if np.any(taus < 0) or np.any(taus >= panel.horizons):
        raise PanelError("every stop time must satisfy 0 <= tau < horizon")
    rows = np.arange(panel.n_traces)
    values = panel.payoff[rows, taus]
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return PolicyResult(
        name=name,
        value=float(values.mean()),
        value_se=se,
        accuracy=float(panel.accuracy[rows, taus].mean()),
        mean_tokens=float(panel.cum_tokens[rows, taus].mean()),
        stop_times=_readonly(taus.copy()),
    )


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise PanelError("values must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
