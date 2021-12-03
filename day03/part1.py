from __future__ import annotations

import argparse
import os.path

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def compute(s: str, width: int = 12) -> int:
    count = 0
    freq = [0 for _ in range(width)]
    lookup = [1 << x for x in range(width)]
    numbers = [int(line, 2) for line in s.splitlines()]

    for n in numbers:
        for pos, b_mask in enumerate(lookup):
            if b_mask & n:
                freq[pos] += 1

        count += 1

    mask = 2 ** width - 1
    gamma = sum((1 if f > count / 2 else 0) * l for f, l in zip(freq, lookup))
    epsilon = gamma ^ mask

    return gamma * epsilon


INPUT_S = """\
00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 198),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s, 5) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
