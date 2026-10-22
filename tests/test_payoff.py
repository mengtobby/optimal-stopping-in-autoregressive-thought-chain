import numpy as np
import pytest

from osc.data.schema import Trace, TraceSet
from osc.stopping.payoff import (
    PanelError,
    bootstrap_mean_ci,
    build_panel,
    evaluate_stop_times,
)


def _trace(pid: str, correct: list[float], answers: list[str | None]) -> Trace:
    t = len(correct)
    return Trace(
        problem_id=pid,
        step_texts=tuple([""] + ["x"] * (t - 1)),
        step_tokens=np.array([0] + [500] * (t - 1)),
        forced_answers=tuple(answers),
        forced_correct=np.array(correct),
        features=np.ones((t, 2)),
        natural_stop=True,
        final_correct=correct[-1],
    )


@pytest.fixture
def panel():
    ts = TraceSet(
        (
            _trace("a", [0.0, 1.0, 1.0], [None, "4", " 4 "]),
            _trace("b", [0.0, 0.0, 1.0, 0.0], ["1", "2", "3", "2"]),
        )
    )
    return build_panel(ts, lam=0.1)


def test_panel_padding_and_payoff(panel):
    assert panel.t_max == 4 and list(panel.horizons) == [3, 4]
    assert np.isnan(panel.payoff[0, 3]) and panel.features[0, 3].sum() == 0
    np.testing.assert_allclose(panel.payoff[0, :3], [0.0, 0.95, 0.9])
    assert list(panel.answer_ids[0]) == [-1, 0, 0, -1]
    assert list(panel.answer_ids[1]) == [0, 1, 2, 1]
    assert panel.alive.sum() == 7
    np.testing.assert_allclose(panel.payoff_at(1.0, 1000.0), 0.9)
    with pytest.raises(ValueError):
        panel.payoff[0, 0] = 1.0


def test_evaluate_and_validation(panel):
    res = evaluate_stop_times(panel, np.array([1, 2]), "x")
    assert res.value == pytest.approx((0.95 + 0.9) / 2)
    assert res.accuracy == 1.0 and res.mean_tokens == 750.0
    with pytest.raises(PanelError):
        evaluate_stop_times(panel, np.array([3, 0]), "bad")
    with pytest.raises(PanelError):
        evaluate_stop_times(panel, np.array([0]), "bad")


def test_build_panel_rejects_bad_lambda(panel):
    with pytest.raises(PanelError):
        build_panel(panel.traces, lam=-1.0)
    with pytest.raises(PanelError):
        build_panel(panel.traces, lam=0.1, token_scale=0.0)


def test_subset_and_bootstrap(panel):
    assert panel.subset([1]).n_traces == 1
    lo, hi = bootstrap_mean_ci(np.array([0.0, 1.0] * 50), seed=1)
    assert lo < 0.5 < hi
    with pytest.raises(PanelError):
        bootstrap_mean_ci(np.array([]))
