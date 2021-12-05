from __future__ import annotations

import argparse
import os.path
from typing import Iterable

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def get_points(x1: int, y1: int, x2: int, y2: int) -> Iterable[tuple[int, int]]:
    xs = (
        [x1 for _ in range(abs(y2 - y1) + 1)]
        if x1 == x2
        else get_nums_in_between(x1, x2)
    )
    ys = (
        [y1 for _ in range(abs(x2 - x1) + 1)]
        if y1 == y2
        else get_nums_in_between(y1, y2)
    )

    return zip(xs, ys)


def sign(n: int) -> int:
    return (n > 0) - (n < 0)


def get_nums_in_between(a: int, b: int) -> Iterable[int]:
    return range(a, b + sign(b - a), sign(b - a))


def print_points(points: dict[tuple[int, int], int]) -> None:
    max_x: int | None = None
    max_y: int | None = None
    min_x: int | None = None
    min_y: int | None = None

    for x, y in points:
        max_x = max(x, x) if max_x is None else max(max_x, x, x)
        max_y = max(y, y) if max_y is None else max(max_y, y, y)
        min_x = min(x, x) if min_x is None else min(min_x, x, x)
        min_y = min(y, y) if min_y is None else min(min_y, y, y)

    print()
    for y in range(min_y, max_y + 1):  # type: ignore
        s = ""
        for x in range(min_x, max_x + 1):  # type: ignore
            s += str(points.get((x, y), "."))
        print(s)


def compute(s: str, include_diag: bool = True) -> int:
    all_points: dict[tuple[int, int], int] = {}

    for line in s.splitlines():
        p1, p2 = line.split(" -> ")
        x1, y1 = (int(s) for s in p1.split(","))
        x2, y2 = (int(s) for s in p2.split(","))

        if not include_diag and not (x1 == x2 or y1 == y2):
            continue

        for x, y in get_points(x1, y1, x2, y2):
            if (x, y) not in all_points:
                all_points[x, y] = 1
            else:
                all_points[x, y] += 1

    result = sum(x > 1 for x in all_points.values())

    return result


INPUT_S = """\
0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2
"""


@pytest.mark.parametrize(
    ("input_s", "include_diag", "expected"),
    [
        (INPUT_S, False, 5),
        (INPUT_S, True, 12),
    ],
)
def test(input_s: str, include_diag: bool, expected: int) -> None:
    assert compute(input_s, include_diag) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("-d", action="store_true")
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), include_diag=args.d))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
