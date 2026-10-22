"""Finite-state reasoning traces with an exact pre-termination stopping value."""

from dataclasses import dataclass

import numpy as np

from osc.data.schema import SiblingSet, Trace, TraceSet
from osc.stopping.dual import ValueFn


class SyntheticError(ValueError):
    """Invalid synthetic model or sampling parameters."""


def _integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise SyntheticError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class MarkovReasoningModel:
    transition: np.ndarray
    accuracy: np.ndarray
    stop_prob: np.ndarray
    initial: np.ndarray
    tokens_per_step: int
    t_max: int
    feature_noise: float = 0.0

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("transition", "accuracy", "stop_prob", "initial"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
                raise SyntheticError(f"{name} must contain probabilities in [0, 1]")
            arrays[name] = array
        s = arrays["accuracy"].size
        if s == 0 or arrays["accuracy"].shape != (s,):
            raise SyntheticError("accuracy must be a nonempty vector")
        if arrays["transition"].shape != (s, s):
            raise SyntheticError("transition must have shape (S, S)")
        for name in ("initial", "stop_prob"):
            if arrays[name].shape != (s,):
                raise SyntheticError(f"{name} must have shape (S,)")
        if not np.allclose(arrays["transition"].sum(axis=1), 1, rtol=0, atol=1e-12):
            raise SyntheticError("transition rows must sum to one")
        if not np.isclose(arrays["initial"].sum(), 1, rtol=0, atol=1e-12):
            raise SyntheticError("initial must sum to one")
        _integer(self.tokens_per_step, "tokens_per_step", 1)
        _integer(self.t_max, "t_max", 1)
        if not np.isfinite(self.feature_noise) or self.feature_noise < 0:
            raise SyntheticError("feature_noise must be finite and non-negative")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def default_model() -> MarkovReasoningModel:
    """Explore, find a candidate, flip to a wrong candidate, then loop."""
    return MarkovReasoningModel(
        transition=np.array([[0, 0.85, 0.15, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]),
        accuracy=np.array([0.2, 0.95, 0.1, 0]),
        stop_prob=np.array([0, 0.05, 0.1, 0.5]),
        initial=np.array([1, 0, 0, 0]),
        tokens_per_step=200,
        t_max=8,
    )


def overthinking_model(feature_noise: float = 0.0) -> MarkovReasoningModel:
    """Random discovery followed by correction, doubt, and destructive overthinking.

    States are explore, correct candidate, wrong candidate, doubt, and loop.
    """
    return MarkovReasoningModel(
        transition=np.array(
            [
                [0.72, 0.16, 0.08, 0.03, 0.01],
                [0.00, 0.55, 0.25, 0.15, 0.05],
                [0.05, 0.25, 0.50, 0.15, 0.05],
                [0.20, 0.15, 0.20, 0.35, 0.10],
                [0.00, 0.01, 0.01, 0.03, 0.95],
            ]
        ),
        accuracy=np.array([0.10, 0.98, 0.12, 0.08, 0.01]),
        stop_prob=np.array([0.005, 0.04, 0.03, 0.03, 0.65]),
        initial=np.array([1, 0, 0, 0, 0]),
        tokens_per_step=150,
        t_max=24,
        feature_noise=feature_noise,
    )


def exact_optimal_stop_times(
    model: MarkovReasoningModel,
    lam: float,
    traces: TraceSet,
    token_scale: float = 1000.0,
) -> np.ndarray:
    """Apply the fully observed DP rule to zero-noise model paths."""
    if model.feature_noise != 0:
        raise SyntheticError("exact stop times require feature_noise == 0")
    values = exact_value_function(model, lam, token_scale)
    times = np.empty(len(traces), dtype=np.int64)
    for i, trace in enumerate(traces):
        if trace.n_steps > model.t_max or trace.features.shape[1] != len(model.accuracy):
            raise SyntheticError("trace dimensions must match the model")
        states = trace.features.argmax(axis=1)
        if not np.allclose(trace.features, np.eye(len(model.accuracy))[states]):
            raise SyntheticError("exact stop times require one-hot state features")
        times[i] = trace.n_steps - 1
        for t in range(trace.n_steps - 1):
            state = states[t]
            payoff = model.accuracy[state] - lam * model.tokens_per_step * t / token_scale
            if payoff >= model.transition[state] @ values[t + 1]:
                times[i] = t
                break
    return times


def sample_traceset(
    model: MarkovReasoningModel, n: int, k_siblings: int, seed: int, sibling_stride: int = 1
) -> TraceSet:
    """Sample independent outer paths and independent next-step sibling draws."""
    _integer(n, "n", 1)
    _integer(k_siblings, "k_siblings", 0)
    _integer(sibling_stride, "sibling_stride", 1)
    _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    s = len(model.accuracy)
    eye = np.eye(s)

    def features(states: np.ndarray) -> np.ndarray:
        return eye[states] + rng.normal(0, model.feature_noise, (len(states), s))

    traces = []
    for i in range(n):
        states = [int(rng.choice(s, p=model.initial))]
        natural = False
        for t in range(model.t_max):
            natural = bool(rng.random() < model.stop_prob[states[-1]])
            if natural or t == model.t_max - 1:
                break
            states.append(int(rng.choice(s, p=model.transition[states[-1]])))
        h = len(states)
        siblings: list[SiblingSet | None] = [None] * h
        for t in range(h - 1):
            if k_siblings >= 2 and t % sibling_stride == 0:
                draws = rng.choice(s, size=k_siblings, p=model.transition[states[t]])
                siblings[t] = SiblingSet(
                    np.full(k_siblings, model.tokens_per_step),
                    model.accuracy[draws],
                    features(draws),
                )
        traces.append(
            Trace(
                problem_id=f"synthetic-{i}",
                step_texts=("",) + tuple(f"state {state}; " for state in states[1:]),
                step_tokens=np.array([0] + [model.tokens_per_step] * (h - 1)),
                forced_answers=tuple("A" if model.accuracy[x] >= 0.5 else f"W{x}" for x in states),
                forced_correct=model.accuracy[states],
                features=features(np.array(states)),
                natural_stop=natural,
                final_correct=model.accuracy[states[-1]],
                siblings=tuple(siblings),
            )
        )
    return TraceSet(tuple(traces), {"synthetic": True, "seed": int(seed)})


def exact_value_function(
    model: MarkovReasoningModel, lam: float, token_scale: float = 1000.0
) -> np.ndarray:
    """Values before the natural-termination coin is revealed at each state."""
    if not np.isfinite(lam) or lam < 0:
        raise SyntheticError("lam must be finite and non-negative")
    if not np.isfinite(token_scale) or token_scale <= 0:
        raise SyntheticError("token_scale must be finite and positive")
    values = np.empty((model.t_max, len(model.accuracy)))
    for t in range(model.t_max - 1, -1, -1):
        payoff = model.accuracy - lam * model.tokens_per_step * t / token_scale
        if t == model.t_max - 1:
            values[t] = payoff
        else:
            continuation = model.transition @ values[t + 1]
            q = model.stop_prob
            values[t] = q * payoff + (1 - q) * np.maximum(payoff, continuation)
    return values


def exact_value(model: MarkovReasoningModel, lam: float, token_scale: float = 1000.0) -> float:
    return float(model.initial @ exact_value_function(model, lam, token_scale)[0])


def exact_state_value_fn(
    model: MarkovReasoningModel, lam: float, token_scale: float = 1000.0
) -> ValueFn:
    """Decode observable features by argmax; exact state decoding requires zero noise."""
    values = exact_value_function(model, lam, token_scale)

    def value_fn(t: int, features: np.ndarray, cum_tokens: np.ndarray) -> np.ndarray:
        if not 0 <= t < model.t_max:
            raise SyntheticError("t must be within the model horizon")
        features = np.asarray(features)
        tokens = np.asarray(cum_tokens)
        if features.ndim != 2 or features.shape[1] != len(model.accuracy):
            raise SyntheticError("features must have shape (n, S)")
        if tokens.shape != (len(features),) or not np.all(np.isfinite(tokens)):
            raise SyntheticError("cum_tokens must be a finite vector with n entries")
        if not np.all(np.isfinite(features)):
            raise SyntheticError("features must be finite")
        # Adjust the time-indexed table for the supplied observable token count.
        return (
            values[t, features.argmax(axis=1)]
            + lam * (model.tokens_per_step * t - tokens) / token_scale
        )

    return value_fn
