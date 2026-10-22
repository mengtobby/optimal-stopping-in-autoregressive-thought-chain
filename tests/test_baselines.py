from dataclasses import replace

import numpy as np
import pytest
from test_prophet import make_panel

from osc.stopping.baselines import (
    BaselineError,
    answer_convergence_stop_times,
    cross_fit_probe_scores,
    fit_correctness_probe,
    full_trace_stop_times,
    probe_threshold_stop_times,
    select_param,
    token_budget_stop_times,
)


def test_observable_heuristics():
    panel = make_panel([[0, 1, 0, 1], [1], [1, 0]])
    assert full_trace_stop_times(panel).tolist() == [3, 0, 1]
    for budget, expected in [(0, [0, 0, 0]), (1, [1, 0, 1]), (99, [3, 0, 1])]:
        assert token_budget_stop_times(panel, budget).tolist() == expected
    assert answer_convergence_stop_times(panel, 2).tolist() == [2, 0, 1]
    assert answer_convergence_stop_times(panel, 1).tolist() == [1, 0, 1]
    assert answer_convergence_stop_times(panel, 9).tolist() == [3, 0, 1]
    changed = replace(panel, accuracy=np.zeros_like(panel.accuracy))
    np.testing.assert_array_equal(
        answer_convergence_stop_times(panel, 2), answer_convergence_stop_times(changed, 2)
    )
    for bad in [-1, np.nan, np.inf]:
        with pytest.raises(BaselineError):
            token_budget_stop_times(panel, bad)
    for bad in [0, 1.5, True]:
        with pytest.raises(BaselineError):
            answer_convergence_stop_times(panel, bad)


def test_probe_features_only_and_padding():
    panel = make_panel([[0, 1, 1], [1], [0, 0]])
    for components in [None, 1, 32]:
        probe = fit_correctness_probe(panel, components)
        scores = probe.scores(panel)
        assert np.isnan(scores[~panel.alive]).all()
        changed = replace(panel, accuracy=1 - panel.accuracy, payoff=1 - panel.payoff)
        np.testing.assert_allclose(scores, probe.scores(changed))
        times = probe_threshold_stop_times(panel, scores, 0.5)
        assert np.all((times >= 0) & (times < panel.horizons))
        assert probe_threshold_stop_times(panel, scores, 2).tolist() == [2, 0, 1]
    with pytest.raises(BaselineError):
        probe.scores(replace(panel, features=panel.features[:, :, :1]))
    with pytest.raises(BaselineError):
        fit_correctness_probe(replace(panel, features=panel.features * np.nan))
    for components, c in [(0, 1), (1.5, 1), (1, 0), (1, np.nan)]:
        with pytest.raises(BaselineError):
            fit_correctness_probe(panel, components, c)
    for scores, threshold in [(np.zeros((1, 1)), 0.5), (panel.accuracy, np.nan)]:
        with pytest.raises(BaselineError):
            probe_threshold_stop_times(panel, scores, threshold)


def test_constants_cross_fit_and_selection():
    panel = make_panel([[0, 0], [1], [0, 1]])
    for value in [0, 1]:
        constant = make_panel([[value], [value, value]])
        scores = fit_correctness_probe(constant).scores(constant)
        assert np.all(scores[constant.alive] == value)
    scores = cross_fit_probe_scores(panel, seed=11)
    np.testing.assert_allclose(scores, cross_fit_probe_scores(panel, seed=11))
    assert np.isfinite(scores[panel.alive]).all()
    assert np.isnan(scores[~panel.alive]).all()
    # Two held-out singleton problems must predict only the opposite training label.
    np.testing.assert_allclose(cross_fit_probe_scores(make_panel([[0], [1]])), [[1], [0]])
    with pytest.raises(BaselineError):
        cross_fit_probe_scores(make_panel([[0]]))
    with pytest.raises(BaselineError):
        cross_fit_probe_scores(panel, n_folds=1)
    assert select_param(panel, [0, 1], token_budget_stop_times) == 1
    assert select_param(panel, [10, 20], token_budget_stop_times) == 10
    for candidates in [[], [np.nan]]:
        with pytest.raises(BaselineError):
            select_param(panel, candidates, token_budget_stop_times)
