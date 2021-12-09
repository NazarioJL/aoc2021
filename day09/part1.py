from __future__ import annotations

import argparse
import os.path
from _heapq import heappush, heappushpop
from typing import Iterable

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def find_low_points(data: list[list[int]]) -> Iterable[tuple[tuple[int, int], int]]:
    rows = len(data)
    cols = len(data[0])

    for r, c in ((r, c) for r in range(rows) for c in range(cols)):
        curr = data[r][c]
        neighbors = [
            data[new_r][new_c]
            for new_r, new_c in (
                (r + r_off, c + c_off) for (r_off, c_off) in DIRECTIONS
            )
            if (0 <= new_r < rows and 0 <= new_c < cols)
        ]

        if neighbors and min(neighbors) > curr:
            yield (r, c), curr


def get_basin_size(row: int, col: int, data: list[list[int]]) -> int:
    rows = len(data)
    cols = len(data[0])

    to_visit = [(row, col)]
    size = 0
    visited = set()
    while to_visit:
        r, c = to_visit.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        size += 1

        for r_offset, c_offset in DIRECTIONS:
            new_r, new_c = r + r_offset, c + c_offset
            if (new_r < 0 or new_r >= rows) or (new_c < 0 or new_c >= cols):
                continue

            if data[new_r][new_c] >= 9:
                continue
            else:
                to_visit.append((new_r, new_c))

    return size


def compute(s: str) -> int:
    numbers: list[list[int]] = [
        [int(n_str) for n_str in line] for line in s.splitlines()
    ]
    low_points = find_low_points(numbers)
    return sum(val + 1 for (_, val) in low_points)


def compute_2(s: str) -> int:
    numbers: list[list[int]] = [
        [int(n_str) for n_str in line] for line in s.splitlines()
    ]

    low_points = find_low_points(numbers)
    top_3: list[int] = []

    for ((r, c), _) in low_points:
        size = get_basin_size(r, c, numbers)
        if len(top_3) < 3:
            heappush(top_3, size)
        else:
            if size > top_3[0]:
                heappushpop(top_3, size)

    return top_3[0] * top_3[1] * top_3[2]


INPUT_S = """\
2199943210
3987894921
9856789892
8767896789
9899965678
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 15),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 1134),),
)
def test_2(input_s: str, expected: int) -> None:
    assert compute_2(input_s) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        if args.p == 1:
            print(compute(f.read()))
        else:
            print(compute_2(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
