from __future__ import annotations

import argparse
import os.path

import pytest

from support import timing

from day04.board import BingoBoard

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def compute(s: str) -> int:
    lines = s.splitlines()
    line_count = len(lines)
    drawn_nums = [int(n) for n in lines[0].split(",")]

    board_count = (line_count - 1) // 6
    boards_left: dict[int, BingoBoard] = {}
    boards_lookup: dict[int, list[BingoBoard]] = {}
    boards = []

    for board_idx in range(board_count):
        start = (6 * board_idx) + 1 + 1  # first line and empty space
        end = start + 5

        board = BingoBoard(board_id=board_idx, numbers={}, rows=[5, 5, 5, 5, 5], cols=[5, 5, 5, 5, 5])
        boards_left[board.board_id] = board
        board_lines = lines[start:end]
        boards.append(board)

        all_nums = []

        for row, bl in enumerate(board_lines):
            for col, n in enumerate(int(n_s) for n_s in bl.split()):
                all_nums.append(n)
                if n not in boards_lookup:
                    boards_lookup[n] = []
                boards_lookup[n].append(board)
                board.numbers[n] = row, col

    last_board: BingoBoard | None = None
    last_winning_num: int | None = None

    # TODO: so much repeated code, make more functions

    for num in drawn_nums:
        if last_board is not None:
            break
        if num in boards_lookup:
            for board in boards_lookup[num]:
                if board.board_id not in boards_left:
                    continue
                if board.mark_number(num):
                    del boards_left[board.board_id]
                    if len(boards_left) == 0:
                        last_board = board
                        last_winning_num = num

    if last_board is None:
        raise ValueError("No winning board found")

    return sum(last_board.numbers.keys() or [0]) * (last_winning_num or 0)


INPUT_S = """\
7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1

22 13 17 11  0
 8  2 23  4 24
21  9 14 16  7
 6 10  3 18  5
 1 12 20 15 19

 3 15  0  2 22
 9 18 13 17  5
19  8  7 25 23
20 11 10 24  4
14 21 16 12  6

14 21 17 24  4
10 16 15  9 19
18  8 23 26 20
22 11 13  6  5
 2  0 12  3  7
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 1924),),
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
