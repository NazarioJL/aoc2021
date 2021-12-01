from __future__ import annotations

import argparse
import os.path

import pytest
from more_itertools import pairwise
from more_itertools import sliding_window

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def compute_2(s: str) -> int:
    numbers = [int(line) for line in s.splitlines()]
    result = 0
    curr = 0
    prev = 0

    for i, e in enumerate(numbers):
        if i <= 3 - 1:
            curr += e
        else:
            new_curr = curr + e - numbers[prev]
            if new_curr > curr:
                result += 1
            curr = new_curr
            prev += 1

    return result


def compute(s: str) -> int:
    numbers = [int(line) for line in s.splitlines()]
    return sum(
        1 if b > a else 0
        for a, b in pairwise(sum(w) for w in sliding_window(numbers, 3))
    )


INPUT_S = """\
199
200
208
210
200
207
240
269
260
263
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 5),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
