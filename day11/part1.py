from __future__ import annotations

import argparse
import os.path
from itertools import count

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


OFFSETS: set[tuple[int, int]] = {
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
}


def step(grid: list[list[int]]) -> int:
    rows = len(grid)
    cols = len(grid[0])

    flashing: list[tuple[int, int]] = []
    for r, c in ((r, c) for r in range(rows) for c in range(cols)):
        grid[r][c] += 1
        if grid[r][c] > 9:
            flashing.append((r, c))

    flashed: set[tuple[int, int]] = set()

    while flashing:
        r, c = flashing.pop()
        if (r, c) in flashed:
            continue

        flashed.add((r, c))

        for r_offset, c_offset in OFFSETS:
            new_r, new_c = r + r_offset, c + c_offset
            if 0 <= new_r < rows and 0 <= new_c < cols:
                grid[new_r][new_c] += 1
                if grid[new_r][new_c] > 9:
                    flashing.append((new_r, new_c))

    for r, c in flashed:
        grid[r][c] = 0

    return len(flashed)


def compute(s: str, part: int) -> int:
    grid = [[int(n_str) for n_str in line] for line in s.splitlines()]
    if part == 1:
        return part_1(grid)
    else:
        return part_2(grid)


def part_1(grid: list[list[int]]) -> int:
    return sum(step(grid) for _ in range(100))


def part_2(grid: list[list[int]]) -> int:
    total = len(grid) * len(grid[0])

    for step_count in count(1):
        flashes = step(grid=grid)
        if flashes == total:
            return step_count

    return -1


INPUT_S = """\
5483143223
2745854711
5264556173
6141336146
6357385478
4167524645
2176841721
6882881134
4846848554
5283751526
"""


@pytest.mark.parametrize(
    ("input_s", "part", "expected"),
    ((INPUT_S, 1, 1656), (INPUT_S, 2, 195)),
)
def test(input_s: str, part: int, expected: int) -> None:
    assert compute(input_s, part) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), args.p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
