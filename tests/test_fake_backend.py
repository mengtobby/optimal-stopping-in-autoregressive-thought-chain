import numpy as np
import pytest

from osc.collect.fake_backend import FakeBackend


def test_scores_and_recorded_calls():
    backend = FakeBackend(scores={" 1": -0.5, " 12": -2.0})
    np.testing.assert_array_equal(
        backend.score_continuations("q", "thinking", "probe", [" 1", " 12"], True),
        [-0.5, -2.0],
    )
    assert backend.score_calls == [("q", "thinking", "probe", (" 1", " 12"), True)]
    with pytest.raises(ValueError, match="missing scripted scores"):
        backend.score_continuations("q", "", "", ["unknown"])


def test_default_scores_and_empty_candidates():
    backend = FakeBackend()
    np.testing.assert_array_equal(backend.score_continuations("q", "", "", ["", " 12"]), [0, -3])
    assert backend.score_continuations("q", "", "", []).shape == (0,)
