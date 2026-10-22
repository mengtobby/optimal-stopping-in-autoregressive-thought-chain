import pytest

from osc.collect.answers import extract_boxed, grade_samples, is_equivalent, normalize_answer


def test_boxes():
    assert extract_boxed(r"\boxed{old} then \fbox{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{unfinished") is None
    assert extract_boxed("none") is None


@pytest.mark.parametrize(
    "a,b",
    [
        (r"\frac{1}{2}", "0.5"),
        ("1/2", ".5"),
        ("1,000", "1000"),
        (r"$ x = \left\dfrac{1}{2}\right $.", "0.5"),
        (r"\text{yes}", "yes"),
    ],
)
def test_equivalence(a, b):
    assert is_equivalent(a, b)


def test_grading():
    assert not is_equivalent(None, "1")
    assert not is_equivalent("1/0", "1")
    assert not is_equivalent("nan", "1")
    assert not is_equivalent("a/b", "1")
    assert normalize_answer(r"30^{\circ}") == "30"
    assert grade_samples([], "1") == (None, 0)
    assert grade_samples(["none"], "1") == (None, 0)
    assert grade_samples([r"\boxed{1/2}", r"\boxed{.5}", r"\boxed{2}", "none"], ".5") == (
        "1/2",
        0.5,
    )
