"""Independent finite-state stopping benchmark and LSM contract checks."""

from dataclasses import replace

import numpy as np
import pytest

from osc.data.schema import Trace, TraceSet
from osc.stopping.lsm import LSMConfig, cross_fit_lsm, fit_lsm, select_lsm_hyperparams
from osc.stopping.payoff import build_panel, evaluate_stop_times
from osc.stopping.synthetic import overthinking_model, sample_traceset


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
        {"margins": ()},
        {"margins": (-0.1,)},
        {"margins": (float("nan"),)},
        {"ridge_alphas": ()},
        {"ridge_alphas": (float("inf"),)},
        {"ridge_alphas": (-1,)},
        {"inner_folds": 1},
        {"inner_folds": 2.5},
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


def test_problem_grouped_folds_and_unique_id_compatibility():
    panel, _ = toy(60)
    _, _, folds = cross_fit_lsm(panel, seed=17)
    expected = np.empty(60, dtype=int)
    expected[np.random.default_rng(17).permutation(60)] = np.arange(60) % 5
    np.testing.assert_array_equal(folds, expected)
    grouped = TraceSet(
        tuple(replace(t, problem_id=str(i // 3)) for i, t in enumerate(panel.traces))
    )
    panel = build_panel(grouped, panel.lam, panel.token_scale)
    _, _, folds = cross_fit_lsm(panel, seed=17)
    for i in range(20):
        assert len(set(folds[3 * i : 3 * i + 3])) == 1
    with pytest.raises(ValueError, match="number of problems"):
        cross_fit_lsm(panel, n_folds=21)


def test_infinite_margin_finishes_without_changing_value():
    panel, _ = toy(30)
    policy = fit_lsm(panel)
    finish = replace(policy, margin=float("inf"))
    np.testing.assert_array_equal(finish.stop_times(panel), panel.horizons - 1)
    x, tokens = panel.features[:, 0], panel.cum_tokens[:, 0]
    np.testing.assert_array_equal(finish.value(0, x, tokens), policy.value(0, x, tokens))


def test_increasing_payoff_recovers_finishing():
    panel, _ = toy(150)
    traces = TraceSet(
        tuple(replace(tr, forced_correct=np.linspace(0, 1, tr.n_steps)) for tr in panel.traces)
    )
    panel = build_panel(traces, 0)
    taus, _, _ = cross_fit_lsm(panel, n_folds=3, seed=4)
    assert evaluate_stop_times(panel, taus, "lsm").value >= (
        evaluate_stop_times(panel, panel.horizons - 1, "finish").value - 0.001
    )


def test_overthinking_selects_early_stopping():
    panel = build_panel(sample_traceset(overthinking_model(0.05), 300, 0, 18), 0.01)
    taus, policies, _ = cross_fit_lsm(panel, n_folds=3, seed=2)
    assert all(np.isfinite(policy.margin) for policy in policies)
    assert evaluate_stop_times(panel, taus, "lsm").value > (
        evaluate_stop_times(panel, panel.horizons - 1, "finish").value + 0.1
    )


def test_selection_fallback_and_conservative_ties():
    panel, _ = toy(12)
    assert select_lsm_hyperparams(panel, LSMConfig(inner_folds=13)) == (1.0, float("inf"))
    constant = build_panel(
        TraceSet(tuple(replace(tr, forced_correct=np.ones(tr.n_steps)) for tr in panel.traces)),
        0,
    )
    assert select_lsm_hyperparams(constant) == (100.0, float("inf"))


def test_nested_selection_uses_only_training_problem_groups(monkeypatch):
    import osc.stopping.lsm as lsm

    panel, _ = toy(36)
    panel = build_panel(
        TraceSet(tuple(replace(tr, problem_id=str(i // 3)) for i, tr in enumerate(panel.traces))),
        panel.lam,
        panel.token_scale,
    )
    original_fit = lsm.fit_lsm
    original_evaluate = lsm.evaluate_stop_times
    training_ids = set()

    def fit(train, config):
        nonlocal training_ids
        training_ids = {tr.problem_id for tr in train.traces}
        return original_fit(train, config)

    def evaluate(held_out, taus, name):
        assert training_ids.isdisjoint(tr.problem_id for tr in held_out.traces)
        return original_evaluate(held_out, taus, name)

    monkeypatch.setattr(lsm, "fit_lsm", fit)
    monkeypatch.setattr(lsm, "evaluate_stop_times", evaluate)
    original_select = lsm.select_lsm_hyperparams
    selected_ids = []

    def select(train, config, seed):
        selected_ids.append({tr.problem_id for tr in train.traces})
        return original_select(train, config, seed)

    monkeypatch.setattr(lsm, "select_lsm_hyperparams", select)
    _, _, folds = cross_fit_lsm(panel, n_folds=3)
    for fold, ids in enumerate(selected_ids):
        assert ids == {tr.problem_id for i, tr in enumerate(panel.traces) if folds[i] != fold}


def test_passthrough_preserves_signal_and_default_behavior():
    panel, _ = toy(80)
    np.testing.assert_array_equal(
        fit_lsm(panel).stop_times(panel),
        fit_lsm(panel, LSMConfig(passthrough=0)).stop_times(panel),
    )
    from test_baselines import signal_panel

    train, test = signal_panel(10), signal_panel(11)
    compressed = fit_lsm(train, LSMConfig(n_components=1))
    preserved = fit_lsm(train, LSMConfig(n_components=1, passthrough=1))
    assert evaluate_stop_times(test, preserved.stop_times(test), "signals").value > (
        evaluate_stop_times(test, compressed.stop_times(test), "pca").value + 0.1
    )
    assert preserved.value(0, test.features[:0, 0], test.cum_tokens[:0, 0]).shape == (0,)
    all_signals = fit_lsm(train, LSMConfig(passthrough=train.features.shape[2]))
    assert all_signals.pca is None
    assert all_signals.scaler is None
    assert np.isfinite(all_signals.value(0, test.features[:, 0], test.cum_tokens[:, 0])).all()
    with pytest.raises(ValueError, match="passthrough"):
        fit_lsm(panel, LSMConfig(passthrough=4))


@pytest.mark.parametrize("passthrough", [-1, 1.5, True])
def test_invalid_passthrough(passthrough):
    with pytest.raises(ValueError, match="passthrough"):
        LSMConfig(passthrough=passthrough)
