from __future__ import annotations

import argparse
import os.path
from typing import NamedTuple

import pytest
from _pytest.capture import CaptureFixture

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


class Grid(NamedTuple):
    right: set[tuple[int, int]]
    down: set[tuple[int, int]]
    rows: int
    cols: int


def parse_grid(s: str) -> Grid:
    right: set[tuple[int, int]] = set()
    down: set[tuple[int, int]] = set()

    rows = s.splitlines()
    for r, line in enumerate(rows):
        for c, x in enumerate(line):
            if x == ">":
                right.add((r, c))
            elif x == "v":
                down.add((r, c))
            else:
                pass

    return Grid(right=right, down=down, rows=len(rows), cols=len(rows[0]))


def print_grid(grid: Grid) -> None:
    for r in range(grid.rows):
        buffer = []
        for c in range(grid.cols):
            if (r, c) in grid.right:
                buffer.append(">")
            elif (r, c) in grid.down:
                buffer.append("v")
            else:
                buffer.append(".")
        print("".join(buffer))


def step(grid: Grid) -> int:
    def get_index(r_: int, c_: int) -> tuple[int, int]:
        return r_ % grid.rows, c_ % grid.cols

    total_moves = 0
    moves: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for r, c in grid.right:
        next_move = get_index(r, c + 1)
        if next_move not in grid.right and next_move not in grid.down:
            moves.append(((r, c), next_move))

    total_moves += len(moves)
    for from_, to in moves:
        grid.right.remove(from_)
        grid.right.add(to)

    moves.clear()

    for r, c in grid.down:
        next_move = get_index(r + 1, c)
        if next_move not in grid.right and next_move not in grid.down:
            moves.append(((r, c), next_move))

    total_moves += len(moves)
    for from_, to in moves:
        grid.down.remove(from_)
        grid.down.add(to)

    return total_moves


def compute(s: str) -> int:
    grid = parse_grid(s)

    total_steps = 0
    while step(grid):
        total_steps += 1

    print_grid(grid)

    return total_steps + 1


INPUT_S1 = """\
...>...
.......
......>
v.....>
......>
.......
..vvv..
"""

OUTPUT_S1 = """\
>......
..v....
..>.v..
.>.v...
...>...
.......
v......
"""

INPUT_S2 = """\
v...>>.vv>
.vv>>.vv..
>>.>v>...v
>>v>>.>.v.
v>v.vv.v..
>.>>..v...
.vv..>.>v.
v.v..>>v.v
....v..v.>
"""

OUTPUT_S2 = """\
..>>v>vv..
..v.>>vv..
..>>v>>vv.
..>>>>>vv.
v......>vv
v>v....>>v
vvv.....>>
>vv......>
.>v.vv.v..
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S2, 58),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


@pytest.mark.parametrize(
    ("input_s", "step_count", "expected"),
    (
        (INPUT_S1, 4, OUTPUT_S1),
        (INPUT_S2, 57, OUTPUT_S2),
    ),
)
def test_step(
    input_s: str, step_count: int, expected: str, capsys: CaptureFixture[str]
) -> None:
    grid = parse_grid(input_s)
    for _ in range(step_count):
        step(grid)

    print_grid(grid)
    assert capsys.readouterr().out == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
