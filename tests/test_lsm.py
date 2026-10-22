"""Independent finite-state stopping benchmark and LSM contract checks."""

from dataclasses import replace

import numpy as np
import pytest

from osc.data.schema import Trace, TraceSet
from osc.stopping.lsm import LSMConfig, cross_fit_lsm, fit_lsm
from osc.stopping.payoff import build_panel, evaluate_stop_times


def toy(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    rewards = np.array([0.1, 0.55, 0.95])
    transition = np.array([[0.45, 0.35, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    termination = np.array([0.1, 0.2, 0.4])
    horizon, cost = 6, 0.06
    values = rewards.copy()
    for t in range(horizon - 2, -1, -1):
        next_reward = rewards - cost * (t + 1)
        cont = transition @ (termination * next_reward + (1 - termination) * values)
        values = np.maximum(rewards - cost * t, cont)
    traces = []
    for i in range(n):
        states = [int(rng.integers(3))]
        for _ in range(1, horizon):
            state = int(rng.choice(3, p=transition[states[-1]]))
            states.append(state)
            if rng.random() < termination[state]:
                break
        h = len(states)
        traces.append(
            Trace(
                problem_id=str(i),
                step_texts=("",) * h,
                step_tokens=np.array([0] + [1] * (h - 1)),
                forced_answers=(None,) * h,
                forced_correct=rewards[states],
                features=np.eye(3)[states],
                natural_stop=True,
                final_correct=float(rewards[states[-1]]),
            )
        )
    return build_panel(TraceSet(tuple(traces)), cost, token_scale=1), float(values.mean())


def test_cross_fitted_markov_value():
    panel, exact = toy()
    taus, policies, folds = cross_fit_lsm(panel, n_folds=5, seed=7)
    result = evaluate_stop_times(panel, taus, "lsm")
    assert abs(result.value - exact) < 0.03
    assert result.value <= exact + 3 * result.value_se
    assert np.all((taus >= 0) & (taus < panel.horizons))
    assert len(policies) == 5
    for fold, policy in enumerate(policies):
        rows = np.flatnonzero(folds == fold)
        np.testing.assert_array_equal(taus[rows], policy.stop_times(panel.subset(rows)))


def test_observable_decisions_and_values():
    panel, _ = toy(120)
    policy = fit_lsm(panel)
    changed_traces = TraceSet(
        tuple(replace(tr, forced_correct=1 - tr.forced_correct) for tr in panel.traces)
    )
    changed = build_panel(changed_traces, panel.lam, panel.token_scale)
    np.testing.assert_array_equal(policy.stop_times(panel), policy.stop_times(changed))
    x, tokens = panel.features[:, 0], panel.cum_tokens[:, 0]
    np.testing.assert_allclose(
        policy.value(0, x, tokens),
        np.maximum(policy.payoff_estimate(0, x, tokens), policy.continuation(0, x, tokens)),
    )
    assert np.all(np.isneginf(policy.continuation(panel.t_max - 1, x, tokens)))
    assert policy.value(0, x[:0], tokens[:0]).shape == (0,)


def test_sparse_fallback_no_pca_and_single_state():
    panel, _ = toy(5)
    policy = fit_lsm(panel, LSMConfig(n_components=None, min_samples=100))
    assert policy.pca is None
    assert np.all(policy.stop_times(panel) < panel.horizons)
    singleton = build_panel(
        TraceSet(
            (
                replace(
                    panel.traces[0],
                    step_texts=("",),
                    step_tokens=np.array([0]),
                    forced_answers=(None,),
                    forced_correct=np.array([0.5]),
                    features=np.ones((1, 3)),
                ),
            )
        ),
        0,
    )
    policy = fit_lsm(singleton)
    np.testing.assert_array_equal(policy.stop_times(singleton), [0])
    assert policy.pca is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": 0},
        {"n_components": -1},
        {"ridge_alpha": -1},
        {"ridge_alpha": float("nan")},
        {"min_samples": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LSMConfig(**kwargs)


@pytest.mark.parametrize("folds", [1, 11, 2.5])
def test_invalid_folds(folds):
    panel, _ = toy(10)
    with pytest.raises(ValueError, match="n_folds"):
        cross_fit_lsm(panel, n_folds=folds)


@pytest.mark.parametrize(
    "t,x,tokens",
    [
        (-1, np.ones((1, 3)), np.zeros(1)),
        (0, np.ones((1, 2)), np.zeros(1)),
        (0, np.ones((1, 3)), np.array([np.nan])),
        (0, np.ones((1, 3)), np.array([-1])),
    ],
)
def test_invalid_observables(t, x, tokens):
    panel, _ = toy(10)
    with pytest.raises(ValueError):
        fit_lsm(panel).value(t, x, tokens)


def test_seeded_folds():
    panel, _ = toy(30)
    first = cross_fit_lsm(panel, seed=12)
    second = cross_fit_lsm(panel, seed=12)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[2], second[2])
