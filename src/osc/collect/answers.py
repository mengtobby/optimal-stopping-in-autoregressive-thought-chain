"""Conservative boxed-answer extraction and numeric grading."""

import math
import re
from collections.abc import Sequence


def extract_boxed(text: str) -> str | None:
    result = None
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start = match.end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                depth += 1
            elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
                depth -= 1
            if depth == 0:
                result = text[start:i]
                break
    return result


def normalize_answer(ans: str) -> str:
    ans = ans.replace("{,}", ",").replace(r"\,", ",")
    ans = ans.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\!", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = re.sub(r"\\text\{([^{}]*)\}", r"\1", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = re.sub(r"^[A-Za-z][A-Za-z_0-9]*=", "", ans)
    ans = ans.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    return ans.rstrip(".")


def _number(ans: str) -> float | None:
    if "," in ans:
        if not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", ans):
            return None
        ans = ans.replace(",", "")
    fraction = re.fullmatch(r"([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", ans)
    if fraction:
        ans = fraction[1] + fraction[2] + "/" + fraction[3]
    try:
        if "/" in ans:
            numerator, denominator = ans.split("/")
            value = float(numerator) / float(denominator)
        else:
            value = float(ans)
        return value if math.isfinite(value) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def is_equivalent(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False

    def elements(value: str) -> list[str] | None:
        value = value.replace("{,}", ",").replace(r"\,", ",")
        if "," in value and (
            re.search(r",\s", value)
            or not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", normalize_answer(value))
        ):
            return value.split(",")
        return None

    xs, ys = elements(pred), elements(gold)
    if xs is not None or ys is not None:
        return (
            xs is not None
            and ys is not None
            and len(xs) == len(ys)
            and all(is_equivalent(x, y) for x, y in zip(xs, ys, strict=True))
        )
    left, right = normalize_answer(pred), normalize_answer(gold)
    if left == right:
        return True
    a, b = _number(left), _number(right)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)


def grade_samples(samples: Sequence[str], gold: str) -> tuple[str | None, float]:
    if not samples:
        return None, 0.0
    answers = [extract_boxed(sample) for sample in samples]
    groups: list[list[str]] = []
    for answer in answers:
        if answer is None:
            continue
        for group in groups:
            if is_equivalent(answer, group[0]):
                group.append(answer)
                break
        else:
            groups.append([answer])
    majority = max(groups, key=len)[0] if groups else None
    return majority, sum(is_equivalent(a, gold) for a in answers) / len(answers)
