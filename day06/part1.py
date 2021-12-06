from __future__ import annotations

import argparse
import os.path

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def compute(s: str, days: int) -> int:
    numbers = [int(line) for line in s.split(",")]
    count = [0 for _ in range(9)]

    for num in numbers:
        count[num] += 1

    print(count)

    for day in range(days):
        # print(f"Start: {day=}: {sum(count)}, {count=}")

        new_fish = count[0]
        new_parents = count[0]

        for idx in range(0, 8):
            count[idx] = count[idx + 1]

        count[8] = new_fish
        count[6] += new_parents
        # print(f"End: {day=}: {sum(count)}, {new_fish=}, {new_parents=} {count=}")




    return sum(count)


INPUT_S = """\
3,4,3,1,2
"""


@pytest.mark.parametrize(
    ("input_s", "days", "expected"),
    [(INPUT_S, 80, 5934), (INPUT_S, 256, 5934)]
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
