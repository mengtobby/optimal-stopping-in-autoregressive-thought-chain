"""Uncertainty estimates for trace means with dependent problem clusters."""

from collections.abc import Sequence

import numpy as np


def _cluster_totals(
    values: np.ndarray, clusters: Sequence[str] | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    labels = np.asarray(clusters)
    if v.ndim != 1 or not v.size or not np.all(np.isfinite(v)):
        raise ValueError("values must be a non-empty finite 1-D array")
    if labels.ndim != 1 or labels.size != v.size:
        raise ValueError("clusters must be a 1-D array with the same length as values")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("clusters must contain finite labels")
    # Preserve first occurrence order so singleton clusters reproduce the trace bootstrap.
    codes: dict[object, int] = {}
    try:
        inverse = np.array([codes.setdefault(label, len(codes)) for label in labels])
    except TypeError as exc:
        raise ValueError("cluster labels must be hashable") from exc
    totals = np.bincount(inverse, weights=v)
    counts = np.bincount(inverse)
    return v, totals, counts


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    clusters: Sequence[str] | np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI resampling whole clusters, with equal weight per sampled trace.

    Unequal cluster sizes yield varying bootstrap sample sizes; each mean is the
    sum of the sampled cluster totals divided by their total trace count.
    """
    _, totals, counts = _cluster_totals(values, clusters)
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    # Bound temporary memory for large reports.
    batch_size = max(1, 1_000_000 // len(totals))
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        sampled = rng.integers(0, len(totals), size=(stop - start, len(totals)))
        means[start:stop] = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def cluster_standard_error(values: np.ndarray, clusters: Sequence[str] | np.ndarray) -> float:
    """Cluster-robust SE with G/(G-1) correction; zero for a single cluster.

    Singleton clusters give exactly std(ddof=1)/sqrt(N). With only one cluster
    between-cluster variation is unidentifiable; zero follows the singleton
    convention of the existing i.i.d. estimator.
    """
    v, totals, counts = _cluster_totals(values, clusters)
    g = len(totals)
    if g == 1:
        return 0.0
    residual_sums = totals - counts * v.mean()
    return float(np.sqrt(g / (g - 1) * np.sum(residual_sums**2) / v.size**2))
