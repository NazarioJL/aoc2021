from __future__ import annotations

import argparse
import os.path
from functools import reduce
from itertools import count

import pytest
from _pytest.capture import CaptureFixture

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def print_paper(paper: set[tuple[int, int]]) -> None:
    cols, rows = reduce(
        lambda acc, e: (max(acc[0], e[0]), max(acc[1], e[1])),
        paper,
        (0, 0),
    )

    for r in range(rows + 1):
        line = ""
        for c in range(cols + 1):
            char = "#" if (c, r) in paper else "."
            line += char
        print(line)


def fold(paper: set[tuple[int, int]], axis: str, axis_val: int) -> set[tuple[int, int]]:
    result = set()
    if axis == "x":  # folding horizontally
        for x, y in paper:
            if x <= axis_val:
                result.add((x, y))
            else:
                # we need to get the mirror
                new_x = axis_val - abs(x - axis_val)
                result.add((new_x, y))
    else:  # must be "y", folding vertically
        for x, y in paper:
            if y <= axis_val:
                result.add((x, y))
            else:
                # we need to get the mirror
                new_y = axis_val - abs(y - axis_val)
                result.add((x, new_y))

    return result


def compute(s: str, folds: int) -> int:
    points: set[tuple[int, int]] = set()
    instructions: list[tuple[str, int]] = []
    for line in s.splitlines():
        if "," in line:
            x_str, y_str = line.split(",")
            x, y = int(x_str), int(y_str)
            points.add((x, y))
        elif "=" in line:
            _, _, inst = line.split()
            axis, val = inst.split("=")
            instructions.append((axis, int(val)))
        else:
            continue

    if folds == -1:
        it = zip(count(), instructions)
    else:
        it = zip(range(folds), instructions)

    for folds, (axis, axis_val) in it:
        points = fold(points, axis, axis_val)

    print_paper(points)
    return len(points)


INPUT_S = """\
6,10
0,14
9,10
0,3
10,4
4,11
6,0
6,12
4,1
0,13
10,12
3,4
3,0
8,4
1,10
2,14
8,10
9,0

fold along y=7
fold along x=5
"""


EXPECTED_OUTPUT_1 = """\
#.##..#..#.
#...#......
......#...#
#...#......
.#.#..#.###
"""


EXPECTED_OUTPUT_2 = """\
#####
#...#
#...#
#...#
#####
"""


@pytest.mark.parametrize(
    ("input_s", "folds", "expected", "expected_output"),
    (
        (INPUT_S, 1, 17, EXPECTED_OUTPUT_1),
        (INPUT_S, 2, 16, EXPECTED_OUTPUT_2),
    ),
)
def test(
    input_s: str,
    folds: int,
    expected: int,
    expected_output: str,
    capsys: CaptureFixture,
) -> None:
    assert compute(input_s, folds) == expected
    assert capsys.readouterr().out == expected_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--n", type=int, default=-1)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), args.n))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
