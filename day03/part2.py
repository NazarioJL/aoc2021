from __future__ import annotations

import argparse
import os.path
from itertools import cycle

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


TO_REMOVE_TRUTH_TABLE: set[tuple[bool, bool, bool]] = {
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
}


def compute(s: str, width: int = 12) -> int:
    lookup = [1 << x for x in range(width)]
    lookup.reverse()

    numbers = [int(line, 2) for line in s.splitlines()]

    def update_source(
        source: set[int],
        to_remove: list[int],
        freq_ones: list[int],
        freq_zeros: list[int],
    ) -> None:
        """Update source and frequency tables with material to remove"""
        for r in to_remove:
            source.remove(r)
            for tmp_idx, tmp_mask in enumerate(lookup):
                if tmp_mask & r:
                    freq_ones[tmp_idx] -= 1
                else:
                    freq_zeros[tmp_idx] -= 1

    def reduce_material(
        values: list[int],
        majority: bool = False,
    ) -> int:
        """Reduce material based on majority or minority strategy"""
        freq_ones = [0 for _ in range(width)]
        freq_zeros = [0 for _ in range(width)]
        material: set[int] = set()

        for value in values:
            material.add(value)
            for idx, b_mask in enumerate(lookup):
                if value & b_mask:
                    freq_ones[idx] += 1
                else:
                    freq_zeros[idx] += 1

        iteration = 0
        for idx in cycle(range(width)):
            b_mask = lookup[idx]
            to_remove = []

            ones_dominant = freq_ones[idx] >= freq_zeros[idx]
            for m in material:
                mask = not b_mask & m
                if (ones_dominant, majority, mask) in TO_REMOVE_TRUTH_TABLE:
                    to_remove.append(m)

            update_source(material, to_remove, freq_ones, freq_zeros)

            iteration += 1

            if len(material) == 1:
                break

        return material.pop()

    oxygen_left = reduce_material(numbers, True)
    co2_left = reduce_material(numbers, False)

    return oxygen_left * co2_left


INPUT_S = """\
00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 230),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s, 5) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
