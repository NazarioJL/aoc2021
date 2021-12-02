from __future__ import annotations

import argparse
import os.path
from typing import Tuple

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def compute(s: str) -> int:
    lines = s.splitlines()
    horizontal = 0
    depth = 0

    for line in lines:
        verb: str
        qty_str: str
        verb, qty_str = line.split()
        qty = int(qty_str)
        if verb.casefold() == "forward":
            horizontal += qty
        elif verb.casefold() == "down":
            depth += qty
        elif verb.casefold() == "up":
            depth -= qty
        else:
            raise ValueError(f"Unrecognized verb: '{verb}'")

    return horizontal * depth


INPUT_S = """\
forward 5
down 5
forward 8
up 3
down 8
forward 2
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 150),),
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
