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


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("1, 2", "12", False),
        ("1, 100", "1100", False),
        ("1, 234", "1234", False),
        ("10, 100", "10100", False),
        ("1,234", "1234", True),
        ("1{,}000", "1000", True),
        ("10{,}000", "10000", True),
        ("10{,}000.5", "10000.5", True),
        (r"1\,000", "1000", True),
        (r"-1\,234\,567.5", "-1234567.5", True),
        ("1, 000", "1000", False),
        ("1,2", "1, 2", True),
        ("1,2", "2,1", False),
        ("1,2.0", "1.0,2", True),
        ("12,34", "1234", False),
        ("-1,234.5", "-1234.5", True),
    ],
)
def test_comma_answers(left, right, expected):
    assert is_equivalent(left, right) is expected
    assert is_equivalent(right, left) is expected


def test_boxed_thousands():
    assert is_equivalent(extract_boxed(r"\boxed{1,000}"), "1000")
