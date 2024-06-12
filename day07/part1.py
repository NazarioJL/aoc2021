from __future__ import annotations

import argparse
import os.path
from typing import Callable

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def compute(s: str, linear_cost: bool = False) -> int:
    crabs = [int(n_str) for n_str in s.split(",")]

    result = None
    result_pos = 0

    max_pos = max(crabs)
    min_pos = min(crabs)

    for new_pos in range(min_pos, max_pos + 1):
        total_fuel = 0
        for pos in crabs:
            diff = abs(new_pos - pos)
            used_fuel = diff if linear_cost else (diff * diff + diff) // 2
            total_fuel += used_fuel
        if result is None:
            result = total_fuel
        else:
            if total_fuel < result:
                result_pos = new_pos
            result = min(total_fuel, result)

    print(f"{result_pos=}")
    return result or 0


INPUT_S = """\
16,1,2,0,4,2,7,1,2,14
"""


@pytest.mark.parametrize(
    ("func", "input_s", "linear_cost", "expected"),
    [
        (compute, INPUT_S, True, 37),
        (compute, INPUT_S, False, 168),
    ],
)
def test(
    func: Callable[[str, bool], int], input_s: str, linear_cost: bool, expected: int
) -> None:
    assert func(input_s, linear_cost) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("-c", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), args.c))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
