"""Hindsight and perfect-verifier benchmarks, unavailable to observable policies.

Threshold functions deliberately inspect true payoff: these are oracle-verifier
rules, not implementable early-exit policies. Use an ``oracle-`` result name when
evaluating their stop times.
"""

from __future__ import annotations

import numpy as np

from osc.stopping.payoff import PayoffPanel, PolicyResult, evaluate_stop_times


class ProphetError(ValueError):
    """Invalid oracle threshold input."""


def hindsight_prophet(panel: PayoffPanel) -> PolicyResult:
    """Return the unattainable future-aware optimum, taking earliest ties."""
    times = np.argmax(np.where(panel.alive, panel.payoff, -np.inf), axis=1)
    return evaluate_stop_times(panel, times, "hindsight-prophet")


def threshold_stop_times(panel: PayoffPanel, threshold: float) -> np.ndarray:
    """Oracle: stop at the first true payoff at least threshold, else terminal."""
    if np.isnan(threshold):
        raise ProphetError("threshold must not be NaN")
    return time_threshold_stop_times(panel, np.full(panel.t_max, threshold))


def samuel_cahn_threshold(panel: PayoffPanel) -> float:
    """Median empirical path maximum (the Samuel-Cahn oracle threshold)."""
    return float(np.nanmedian(np.nanmax(panel.payoff, axis=1)))


def kleinberg_weinberg_threshold(panel: PayoffPanel) -> float:
    """Half the empirical prophet value (the Kleinberg-Weinberg threshold)."""
    return float(0.5 * np.nanmax(panel.payoff, axis=1).mean())


def independent_dp_thresholds(panel: PayoffPanel) -> np.ndarray:
    """Backward continuation values using empirical alive-step marginals.

    This is optimal for independent steps with a fixed horizon. For ragged
    traces, each next-step marginal uses only traces alive at that step;
    application always forces stopping at the individual trace's last step.
    """
    thresholds = np.full(panel.t_max, -np.inf)
    for t in range(panel.t_max - 2, -1, -1):
        next_values = panel.payoff[panel.alive[:, t + 1], t + 1]
        thresholds[t] = np.maximum(next_values, thresholds[t + 1]).mean()
    return thresholds


def time_threshold_stop_times(panel: PayoffPanel, thresholds: np.ndarray) -> np.ndarray:
    """Oracle: apply time thresholds, forcing a stop at each trace's horizon."""
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (panel.t_max,) or np.isnan(thresholds).any():
        raise ProphetError(f"thresholds must have shape ({panel.t_max},) without NaNs")
    hits = panel.alive & (panel.payoff >= thresholds[None, :])
    hits[np.arange(panel.n_traces), panel.horizons - 1] = True
    return hits.argmax(axis=1)
