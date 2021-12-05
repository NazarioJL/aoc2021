from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BingoBoard:
    board_id: int
    numbers: dict[int, tuple[int, int]]
    rows: list[int]
    cols: list[int]

    def mark_number(self, n: int) -> bool:
        r, c = self.numbers[n]
        self.rows[r] -= 1
        self.cols[c] -= 1

        del self.numbers[n]

        return self.rows[r] == 0 or self.cols[c] == 0

    def pretty_print(self) -> None:
        board = [[" *" for _ in range(5)] for _ in range(5)]
        for n, (r, c) in self.numbers.items():
            board[r][c] = f"{n:2}"

        print(f"==Board({self.board_id:04})==")
        for row in board:
            print(" ".join(row))
        print("===============")
