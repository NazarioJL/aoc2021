from __future__ import annotations

import argparse
import os.path
from functools import cache
from typing import NamedTuple

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


@cache
def get_next_score(turn: int, die_sides: int = 100) -> int:
    start = turn * 3
    s1 = (start % die_sides) + 1
    s2 = (start % die_sides) + 2
    s3 = (start % die_sides) + 3

    print(f"turn {turn} -> {s1}+{s2}+{s3} -> {s1 + s2 + s3}")

    score = s1 + s2 + s3

    return score


def part_1(p1_start: int, p2_start: int) -> int:
    p1_pos = p1_start - 1  # use 0 indexed
    p2_pos = p2_start - 1

    p1_score = 0
    p2_score = 0

    print()
    print(f"Starting positions, Player 1: {p1_pos + 1}, Player 2: {p2_pos + 1}")

    turn = 0

    MAX_SCORE = 1000

    while True:

        roll = get_next_score(turn)
        orig_pos = p1_pos

        p1_pos = (p1_pos + roll) % 10
        p1_score += p1_pos + 1

        print(
            f"Player 1 rolls:  was at: {orig_pos + 1}, moved to {p1_pos + 1}, "
            f"total score = {p1_score}"
        )

        if p1_score >= MAX_SCORE:
            print(
                f"Player 1 wins: {p1_pos=}, {p1_score=}, {p2_pos=}, {p2_score=}, {turn=}"
            )
            winner = 1
            break

        turn += 1
        orig_pos = p2_pos + 1
        roll = get_next_score(turn)
        p2_pos = (p2_pos + roll) % 10

        p2_score += p2_pos + 1

        print(
            f"Player 2 rolls:  was at: {orig_pos + 1}, moved to {p2_pos + 1}, "
            f"total score = {p2_score}"
        )

        if p2_score >= MAX_SCORE:
            print(
                f"Player 2 wins: {p1_pos=}, {p1_score=}, {p2_pos=}, {p2_score=}, {turn=}"
            )
            winner = 2
            break

        turn += 1

    if winner not in (1, 2):
        raise RuntimeError("No winner found")

    rolled_dies = (turn + 1) * 3
    if winner == 1:
        score = rolled_dies * p2_score
    else:
        score = rolled_dies * p1_score

    return score


def add_scores(
    s_1: tuple[int, int], s_2: tuple[int, int], s_3: tuple[int, int]
) -> tuple[int, int]:
    return s_1[0] + s_2[0] + s_3[0], s_1[1] + s_2[1] + s_3[1]


def part_2(p1_start: int, p2_start: int, winning_score: int = 21) -> int:
    memo: dict[tuple[int, int, int, int, int], tuple[int, int]] = {}

    cache_hits = 0

    all_rolls = [3, 4, 5, 6, 7, 8, 9]

    def roll(
        p1_score: int, p1_pos: int, p2_score: int, p2_pos: int, turn: int = 0
    ) -> tuple[int, int]:
        memoized: tuple[int, int] | None = memo.get(
            (p1_score, p1_pos, p2_score, p2_pos, turn)
        )

        if memoized is not None:
            nonlocal cache_hits
            cache_hits += 1
            return memoized

        if p1_score >= winning_score:
            return 1, 0
        if p2_score >= winning_score:
            return 0, 1

        if turn == 0:  # player 1's turn
            for roll in all_rolls
            p1_pos_1 = (p1_pos + 1) % 10
            p1_score_1 = p1_score + p1_pos_1 + 1

            p1_pos_2 = (p1_pos + 2) % 10
            p1_score_2 = p1_score + p1_pos_2 + 1

            p1_pos_3 = (p1_pos + 3) % 10
            p1_score_3 = p1_score + p1_pos_3 + 1

            outcome_1 = roll(p1_score_1, p1_pos_1, p2_score, p2_pos, 1)
            outcome_2 = roll(p1_score_2, p1_pos_2, p2_score, p2_pos, 1)
            outcome_3 = roll(p1_score_3, p1_pos_3, p2_score, p2_pos, 1)

            result = add_scores(outcome_1, outcome_2, outcome_3)

            memo[(p1_score, p1_pos, p2_score, p2_pos, turn)] = result

            return result
        else:  # player 2's turn
            p2_pos_1 = (p2_pos + 1) % 10
            p2_score_1 = p2_score + p2_pos_1 + 1

            p2_pos_2 = (p2_pos + 2) % 10
            p2_score_2 = p2_score + p2_pos_2 + 1

            p2_pos_3 = (p2_pos + 3) % 10
            p2_score_3 = p2_score + p2_pos_3 + 1

            outcome_1 = roll(p1_score, p1_pos, p2_score_1, p2_pos_1, 0)
            outcome_2 = roll(p1_score, p1_pos, p2_score_2, p2_pos_2, 0)
            outcome_3 = roll(p1_score, p1_pos, p2_score_3, p2_pos_3, 0)

            result = add_scores(outcome_1, outcome_2, outcome_3)

            memo[(p1_score, p1_pos, p2_score, p2_pos, turn)] = result

            return result

    p1_wins, p2_wins = roll(
        p1_score=0, p1_pos=p1_start - 1, p2_score=0, p2_pos=p2_start - 1, turn=0
    )

    print(f"{len(memo)}")
    print(f"{cache_hits=}")
    print(f"{(p1_wins, p2_wins)=}")

    return max(p1_wins, p2_wins)


def compute(s: str, part: int = 1) -> int:
    lines = s.splitlines()
    p1_pos = int(lines[0].split()[-1])
    p2_pos = int(lines[1].split()[-1])

    if part == 1:
        return part_1(p1_pos, p2_pos)
    else:
        return part_2(p1_pos, p2_pos)


INPUT_S = """\
Player 1 starting position: 4
Player 2 starting position: 8
"""


@pytest.mark.parametrize(
    ("input_s", "part", "expected"),
    ((INPUT_S, 1, 739785), (INPUT_S, 2, 444356092776315)),
)
def test(input_s: str, part: int, expected: int) -> None:
    assert compute(input_s, part) == expected


def test_part_2():
    result = part_2(1, 0, 5)
    print(result)


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
