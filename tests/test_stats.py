"""Problem-clustered uncertainty estimates."""

import numpy as np
import pytest

from osc.stopping.payoff import bootstrap_mean_ci
from osc.stopping.stats import cluster_bootstrap_mean_ci, cluster_standard_error


def test_singletons_match_iid():
    values = np.random.default_rng(4).normal(size=40)
    clusters = [str(i) for i in range(len(values))]
    assert cluster_bootstrap_mean_ci(values, clusters) == bootstrap_mean_ci(values)
    assert cluster_standard_error(values, clusters) == pytest.approx(
        values.std(ddof=1) / np.sqrt(len(values))
    )


def test_triplication_preserves_uncertainty():
    values = np.random.default_rng(3).normal(size=80)
    clusters = np.arange(len(values)).astype(str)
    original = np.ptp(cluster_bootstrap_mean_ci(values, clusters))
    repeated = np.repeat(values, 3)
    labels = np.repeat(clusters, 3)
    assert np.ptp(cluster_bootstrap_mean_ci(repeated, labels)) / original > 0.8
    assert np.ptp(bootstrap_mean_ci(repeated)) / original < 0.8
    assert cluster_standard_error(repeated, labels) == pytest.approx(
        cluster_standard_error(values, clusters)
    )


def test_unequal_clusters_weight_traces():
    values = np.array([0.0, 0.0, 9.0])
    labels = ["a", "a", "b"]
    assert cluster_standard_error(values, labels) == pytest.approx(4.0)
    rng = np.random.default_rng(2)
    draws = rng.integers(0, 2, size=(100, 2))
    means = np.array([0.0, 9.0])[draws].sum(axis=1) / np.array([2, 1])[draws].sum(axis=1)
    assert cluster_bootstrap_mean_ci(values, labels, 100, 0.6, 2) == tuple(
        np.quantile(means, [0.3, 0.7])
    )


@pytest.mark.parametrize(
    "values, labels",
    [
        ([], []),
        ([1], []),
        ([[1]], ["a"]),
        ([np.nan], ["a"]),
        ([np.inf], ["a"]),
        ([1], [["a"]]),
        ([1], [np.nan]),
    ],
)
def test_invalid_inputs(values, labels):
    for estimate in (cluster_bootstrap_mean_ci, cluster_standard_error):
        with pytest.raises(ValueError):
            estimate(np.asarray(values), labels)


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"n_boot": True}, {"alpha": 1}])
def test_invalid_bootstrap_configuration(kwargs):
    with pytest.raises(ValueError):
        cluster_bootstrap_mean_ci(np.array([1.0]), ["a"], **kwargs)


def test_single_cluster():
    assert cluster_bootstrap_mean_ci(np.array([1.0, 3.0]), ["a", "a"]) == (2.0, 2.0)
    assert cluster_standard_error(np.array([1.0, 3.0]), ["a", "a"]) == 0.0
