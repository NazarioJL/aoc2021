from __future__ import annotations

import argparse
import os.path
from enum import auto
from enum import Enum
from functools import reduce
from statistics import median
from typing import Callable
from typing import Literal
from typing import NamedTuple

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")

MATCH: dict[str, str] = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
}

COST_PART_1: dict[str, int] = {
    ")": 3,
    "]": 57,
    "}": 1197,
    ">": 25137,
}


COST_PART_2: dict[str, int] = {
    ")": 1,
    "]": 2,
    "}": 3,
    ">": 4,
}


CostFunctionTypeDef = Callable[[list[str]], int]
AggregateCostTypeDef = Callable[[list[int]], int]


def cost_1(brackets: list[str]) -> int:
    c = brackets[0]
    return COST_PART_1[c]


def cost_2(brackets: list[str]) -> int:
    return reduce(lambda acc, e: acc * 5 + COST_PART_2[e], brackets, 0)


def aggregate_cost_1(costs: list[int]) -> int:
    return sum(costs)


def aggregate_cost_2(costs: list[int]) -> int:
    # costs is guaranteed to have an odd number of elements
    return int(median(costs))


class ErrorType(Enum):
    NoError = auto()
    MissingClosing = auto()
    MissingOpening = auto()
    Mismatched = auto()


class ParseResult(NamedTuple):
    error_type: ErrorType
    error_brackets: list[str]


def parse(exp: str) -> ParseResult:
    stack = []

    for c in exp:
        if c in MATCH:
            stack.append(c)  # push open bracket
        else:
            if not stack:
                return ParseResult(
                    error_type=ErrorType.MissingOpening, error_brackets=[]
                )
            popped = stack.pop()  # get opening bracket
            expected = MATCH[popped]
            if c != expected:
                return ParseResult(error_type=ErrorType.Mismatched, error_brackets=[c])
    if stack:
        return ParseResult(
            error_type=ErrorType.MissingClosing,
            error_brackets=list(reversed([MATCH[s] for s in stack])),
        )
    else:
        return ParseResult(error_type=ErrorType.NoError, error_brackets=[])


COST_FUNCTIONS: dict[ErrorType, tuple[CostFunctionTypeDef, AggregateCostTypeDef]] = {
    ErrorType.Mismatched: (cost_1, aggregate_cost_1),
    ErrorType.MissingClosing: (cost_2, aggregate_cost_2),
}


def compute(s: str, error_type: ErrorType) -> int:
    cost, aggregate_cost = COST_FUNCTIONS[error_type]

    results = [parse(line) for line in s.splitlines()]
    costs = [cost(r.error_brackets) for r in results if r.error_type == error_type]

    return aggregate_cost(costs)


INPUT_S = """\
[({(<(())[]>[[{[]{<()<>>
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{
<{([{{}}[<[[[<>{}]]]>[]]
"""


@pytest.mark.parametrize(
    ("input_s", "error_type", "expected"),
    [
        (INPUT_S, ErrorType.Mismatched, 26397),
        (INPUT_S, ErrorType.MissingClosing, 288957),
    ],
)
def test(input_s: str, error_type: ErrorType, expected: int) -> None:
    assert compute(input_s, error_type) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        error_type = ErrorType.Mismatched if args.p == 1 else ErrorType.MissingClosing
        print(compute(f.read(), error_type))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
