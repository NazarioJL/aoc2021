from __future__ import annotations

import argparse
import os.path
from heapq import heappop
from heapq import heappush

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


DIRECTIONS = (
    (0, 1),  # right
    (1, 0),  # down
    (0, -1),  # left
    (-1, 0),  # up
)


def get_lowest_risk_path(maze: list[list[int]], scale: int = 1) -> int:
    result = 0
    cost_to_node: dict[tuple[int, int], int] = {}

    orig_rows = len(maze)
    orig_cols = len(maze[0])
    rows = orig_rows * scale
    cols = orig_cols * scale

    end = (rows - 1, cols - 1)
    heuristic = rows + cols - 2

    def get_cost(row: int, col: int) -> int:
        section_r = row // orig_rows
        section_c = col // orig_cols
        mapped_r = row % orig_rows
        mapped_c = col % orig_cols

        val_at = maze[mapped_r][mapped_c]
        new_val = val_at + section_c + section_r - 1
        new_val = (new_val % 9) + 1

        return new_val

    heap: list[tuple[int, int, int, int]] = [(0, heuristic, 0, 0)]
    cost_to_node[0, 0] = 0

    while heap:
        cost, h, r, c = heappop(heap)
        if (r, c) == end:
            result = cost
            break
        else:
            for r_offset, c_offset in DIRECTIONS:
                new_r, new_c = r + r_offset, c + c_offset
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    h = (rows - new_r - 1) + (cols - new_c - 1)
                    # cost from r, c -> new_r, new_c = maze[new_r][new_c]
                    # total cost = new_cost + maze[new_r][new_c]
                    new_cost = cost + get_cost(new_r, new_c)

                    if (new_r, new_c) in cost_to_node:
                        prev_cost = cost_to_node[new_r, new_c]
                        if new_cost < prev_cost:
                            # only push if cheaper this time otherwise don't
                            heappush(heap, (new_cost, h, new_r, new_c))
                    else:
                        cost_to_node[new_r, new_c] = new_cost
                        heappush(heap, (new_cost, h, new_r, new_c))

    return result


def compute(s: str, scale: int) -> int:
    maze: list[list[int]] = []

    for line in s.splitlines():
        maze.append([int(c) for c in line])

    result = get_lowest_risk_path(maze, scale=scale)
    return result


INPUT_S = """\
1163751742
1381373672
2136511328
3694931569
7463417111
1319128137
1359912421
3125421639
1293138521
2311944581
"""


@pytest.mark.parametrize(
    ("input_s", "scale", "expected"),
    ((INPUT_S, 1, 40), (INPUT_S, 5, 315)),
)
def test(input_s: str, scale: int, expected: int) -> None:
    assert compute(input_s, scale) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--s", type=int, default=1)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), args.s))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    tuple
