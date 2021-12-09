from __future__ import annotations

import argparse
import os.path
from functools import reduce

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


DIGIT_TO_CODE = {  # Map of the numerical digit to its standard code
    0: "abcefg",
    1: "cf",  # unique
    2: "acdeg",
    3: "acdfg",
    4: "bcdf",  # unique
    5: "abdfg",
    6: "abdefg",
    7: "acf",  # unique
    8: "abcdefg",  # unique
    9: "abcdfg",
}


def normalize(s: str) -> str:
    """
    Normalize a string to be in the lowest lexicographic order. This is useful for
    when we care about the items in the string regardless of their relative order.

    >>> normalize("cba")
    'abc'

    :param s: the string to normalize lexicographically
    :return: the normalized string
    """
    return "".join(sorted(s))


def create_signature(code_map: dict[int, str]) -> dict[tuple[int, int, int, int], int]:
    """
    Creates a signature for each value in a dictionary of indexed codes. This makes an
    assumption that each value is unique and is strictly a permutation of a 7-segment
    display. The original display looks like the following:
        0: "abcefg",
        1: "cf",
        2: "acdeg",
        3: "acdfg",
        4: "bcdf",
        5: "abdfg",
        6: "abdefg",
        7: "acf",
        8: "abcdefg",
        9: "abcdfg"
    We can see the number '8' is represented by all 7 segments on.

     aaaa
    b    c
    b    c
     dddd
    e    f
    e    f
     gggg

    More information can be found -> https://adventofcode.com/2021/day/8

    The produced signature will be unique for each valid permutation of characters that
    represent a number in the 7-segment display. The produced signature will be a
    4-tuple with the following elements:
    - common characters with the number 1 (2 segments)
    - common characters with the number 7 (3 segments)
    - common characters with the number 4 (4 segments)
    - character count for the number being encoded

    Since 8 intersects with all numbers its signature will be: (2, 3, 4, 7)

    :param code_map: the code_map for which the signature is being coded for
    :return: a map of signatures to the original index provided in the code_map
    """
    fixed_length_codes = sorted(
        (code for code in code_map.values() if len(code) in {2, 4, 3}), key=len
    )  # get the codes that represent the number with unique length (1, 7, 4)
    return {
        tuple(  # type: ignore
            [
                *[
                    len(set(code).intersection(set(fixed)))
                    for fixed in fixed_length_codes
                ],
                len(code),
            ]
        ): n
        for n, code in code_map.items()
    }


def decode(
    code_map: dict[tuple[int, int, int, int], int], codes: list[str]
) -> dict[str, int]:
    result: dict[str, int] = {}
    index_to_code = {idx: normalize(code) for idx, code in enumerate(codes)}
    mixed_code_signatures = create_signature(index_to_code)
    for signature, idx in mixed_code_signatures.items():
        real_number = code_map[signature]
        mixed_code = index_to_code[idx]
        result[mixed_code] = real_number

    return result


def part_1(s: str) -> int:
    count = 0
    for line in s.splitlines():
        sides = line.split("|")
        right_side = sides[1].split()
        count += sum(len(code) in {2, 3, 4, 7} for code in right_side)

    return count


def part_2(s: str) -> int:
    master_code = create_signature(DIGIT_TO_CODE)
    result = 0

    for line in s.splitlines():
        sides = line.split("|")
        left_side = sides[0].split()
        right_side = sides[1].split()
        code_to_number = decode(master_code, left_side)
        decoded_nums = [code_to_number[normalize(c)] for c in right_side]
        number = reduce(lambda acc, e: acc * 10 + e, decoded_nums)
        result += number

    return result


def compute(s: str, part: int = 1) -> int:
    if part == 1:
        return part_1(s)
    elif part == 2:
        return part_2(s)
    else:
        raise ValueError("Bad part!")


INPUT_S = """\
be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 26),),
)
def test_part1(input_s: str, expected: int) -> None:
    assert part_1(input_s) == expected


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 61229),),
)
def test_part2(input_s: str, expected: int) -> None:
    assert part_2(input_s) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), part=args.p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
